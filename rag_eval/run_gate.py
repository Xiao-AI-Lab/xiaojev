"""Gated-RAG end-to-end evaluation.

Eval set: 293 non-train musique questions x 2 variants:
  ans   - original (gold docs present in the corpus-level candidate pool)
  unans - gold docids excluded from every pool stage (simulates missing evidence;
          the local gold set is fully answerable, so unanswerable cases can only
          be constructed this way)

Arms:
  nogate - round-1 top-5 always goes to the reader
  gate   - v3 answerability gate with tau calibrated on the calibration split
           (answered-set precision >= 0.90); gate failure triggers one retry
           round (BM25 pool widened 50->100, v3 rescored, reselect, re-gate);
           still failing -> refuse (no reader call)

Splits: calibration (98) tunes tau; test (101) is the primary report; dev (94)
is secondary. Reader: a local OpenAI-compatible server (we used qwen3.8-27b),
same prompt/schema as the RAG eval.

Outputs: gate_scores.jsonl, gate_predictions.jsonl, gate_metrics.json,
GATE_REPORT.md.

Configuration (environment variables):
  XIAOJEV_GATE_DIR      score/output directory (default: this rag_eval/ directory)
  XIAOJEV_CKPT_V3       v3 checkpoint dir (default: <repo>/ckpt/v3)
  XIAOJEV_READER_URL / XIAOJEV_READER_MODEL   reader endpoint (see run_qa.py)
"""
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from train import MODEL_PATH, StudentModel, load_ckpt  # noqa: E402

from rag_eval.gate import OUT, GatePipeline, calibrate_tau  # noqa: E402
from rag_eval.run_qa import call_reader, exact_match, token_f1  # noqa: E402
from rag_eval.score_v3 import score_items  # noqa: E402

CKPT = os.environ.get("XIAOJEV_CKPT_V3", str(ROOT / "ckpt" / "v3"))
TARGET_PRECISION = 0.90


def main():
    t_start = time.time()
    device = "cuda"
    pipe = GatePipeline()
    qids = sorted(pipe.questions)
    splits = {qid: pipe.questions[qid]["split"] for qid in qids}
    print(f"questions: {len(qids)}; index+pools ready ({time.time()-t_start:.0f}s)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    def score_counts():
        from collections import Counter
        c = Counter()
        if (OUT / "gate_scores.jsonl").exists():
            for line in open(OUT / "gate_scores.jsonl"):
                d = json.loads(line)
                c[(d["kind"], d.get("round"))] += 1
        return c

    counts = score_counts()
    scoring_done = (counts[("gate", 1)] == 586 and counts[("gate", 2)] == 586
                    and counts[("retry_rel", None)] > 0)

    def load_scores():
        p_gate, rel_retry = {}, defaultdict(dict)
        for line in open(OUT / "gate_scores.jsonl"):
            d = json.loads(line)
            v = d.get("p_yes", d.get("probs", [0.0])[0])
            if d["kind"] == "gate":
                p_gate[(d["qid"], d["variant"], d["round"])] = v
            elif d["kind"] == "retry_rel":
                rel_retry[(d["qid"], d["variant"])][d["docid"]] = v
        return p_gate, rel_retry

    if scoring_done:
        # timings measured in the first (interrupted) run, kept for the latency account
        t_gate_r1, t_retry_rel, t_gate_r2 = 18.0, 293.0, 18.0
        print(f"resume: v3 scoring complete {dict(counts)}; skipping GPU phases", flush=True)
        p_gate, rel_retry = load_scores()
    else:
        model = StudentModel().to(device)
        load_ckpt(model, CKPT)
        model.eval()
        print("v3 loaded", flush=True)
        with open(OUT / "gate_scores.jsonl", "w") as scores_f:
            # ---- phase 1: round-1 selection + gate scoring (586 items)
            t0 = time.time()
            gate_items = []
            for qid in qids:
                for variant in ("ans", "unans"):
                    gate_items.append(pipe.gate_item(qid, variant, 1,
                                                     pipe.topk_passages(qid, variant)))
            score_items(model, tok, gate_items, device, scores_f)
            t_gate_r1 = time.time() - t0
            print(f"phase1 gate scoring: {len(gate_items)} items, {t_gate_r1:.0f}s", flush=True)

            # ---- phase 2: retry relevance scoring for BM25 ranks 50-100 (all, so the
            # pipeline decision is a pure function of (p1, p2, tau))
            t0 = time.time()
            retry_items = pipe.retry_rel_items([(qid, v) for qid in qids for v in ("ans", "unans")])
            score_items(model, tok, retry_items, device, scores_f)
            t_retry_rel = time.time() - t0
            print(f"phase2 retry relevance: {len(retry_items)} items, {t_retry_rel:.0f}s", flush=True)

        # ---- phase 3: round-2 reselection + gate scoring
        _, rel_retry = load_scores()
        t0 = time.time()
        gate_items2 = []
        for qid in qids:
            for variant in ("ans", "unans"):
                ps = pipe.topk_passages(qid, variant, depth=100,
                                        extra_scores=rel_retry.get((qid, variant), {}))
                gate_items2.append(pipe.gate_item(qid, variant, 2, ps))
        with open(OUT / "gate_scores.jsonl", "a") as scores_f:
            score_items(model, tok, gate_items2, device, scores_f)
        t_gate_r2 = time.time() - t0
        print(f"phase3 round-2 gate: {len(gate_items2)} items, {t_gate_r2:.0f}s", flush=True)
        p_gate, rel_retry = load_scores()

    # top-k selections are pure functions of the scores
    topk_r1, topk_r2 = {}, {}
    for qid in qids:
        for variant in ("ans", "unans"):
            topk_r1[(qid, variant)] = pipe.topk_passages(qid, variant)
            topk_r2[(qid, variant)] = pipe.topk_passages(qid, variant, depth=100,
                                                         extra_scores=rel_retry.get((qid, variant), {}))

    # ---- tau calibration on the calibration split
    cal_records = [{"qid": qid, "variant": v,
                    "p1": p_gate[(qid, v, 1)], "p2": p_gate[(qid, v, 2)]}
                   for qid in qids if splits[qid] == "calibration"
                   for v in ("ans", "unans")]
    tau, curve = calibrate_tau(cal_records, TARGET_PRECISION)
    print(f"calibrated tau={tau:.2f} (target answered-precision {TARGET_PRECISION})", flush=True)

    def accepted(qid, variant):
        return (p_gate[(qid, variant, 1)] >= tau) or (p_gate[(qid, variant, 2)] >= tau)

    def final_passages(qid, variant):
        if p_gate[(qid, variant, 1)] >= tau:
            return topk_r1[(qid, variant)], 1
        return topk_r2[(qid, variant)], 2

    # ---- phase 4: reader calls
    # nogate arm: all 586 with round-1 top-5; gate arm: accepted only
    tasks = []
    for qid in qids:
        for variant in ("ans", "unans"):
            tasks.append((qid, variant, "nogate", topk_r1[(qid, variant)], 1))
            if accepted(qid, variant):
                ps, rnd = final_passages(qid, variant)
                tasks.append((qid, variant, "gate", ps, rnd))
    print(f"reader calls: {len(tasks)}", flush=True)
    # resume: keep existing predictions, only run missing (id, variant, arm)
    done_keys = set()
    if (OUT / "gate_predictions.jsonl").exists():
        for line in open(OUT / "gate_predictions.jsonl"):
            r = json.loads(line)
            done_keys.add((r["id"], r["variant"], r["arm"]))
    tasks = [t for t in tasks if (t[0], t[1], t[2]) not in done_keys]
    print(f"reader calls remaining after resume: {len(tasks)}", flush=True)
    preds_f = open(OUT / "gate_predictions.jsonl", "a")
    lock = threading.Lock()
    done = [0]
    t_reader = time.time()

    def work(task):
        qid, variant, arm, passages, rounds = task
        q = pipe.questions[qid]
        t_call = time.time()
        answer, usage = call_reader(q["question"], passages)
        dt = time.time() - t_call
        aliases = [q["answer"]] + q["answer_aliases"]
        rec = {"id": qid, "variant": variant, "arm": arm, "hop": q["hop"],
               "split": splits[qid], "rounds": rounds,
               "prediction": answer,
               "doc_ids": [p for p, _ in passages],
               "em": None if answer is None else exact_match(answer or "", aliases),
               "f1": None if answer is None else token_f1(answer or "", aliases),
               "empty": answer == "",
               "latency_s": round(dt, 3),
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens")}
        with lock:
            preds_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 100 == 0:
                preds_f.flush()
                print(f"{done[0]}/{len(tasks)} reader calls, {time.time()-t_reader:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, tasks))
    preds_f.close()

    # ---- metrics
    rows = [json.loads(l) for l in open(OUT / "gate_predictions.jsonl")]
    by = defaultdict(dict)
    for r in rows:
        by[(r["id"], r["variant"])][r["arm"]] = r

    def split_metrics(split):
        sub_q = [q for q in qids if splits[q] == split]
        out = {"n_questions": len(sub_q)}
        for variant in ("ans", "unans"):
            for arm in ("nogate", "gate"):
                recs = [by[(q, variant)][arm] for q in sub_q if arm in by.get((q, variant), {})]
                refused = len(sub_q) - len(recs) if arm == "gate" else 0
                ok = [r for r in recs if r["em"] is not None]
                m = {"answered": len(recs), "refused": refused,
                     "failures": len(recs) - len(ok)}
                if variant == "ans":
                    m["em"] = sum(r["em"] for r in ok) / len(sub_q)  # unanswered counts 0
                    m["f1"] = sum(r["f1"] for r in ok) / len(sub_q)
                    m["em_answered"] = sum(r["em"] for r in ok) / len(ok) if ok else None
                    m["f1_answered"] = sum(r["f1"] for r in ok) / len(ok) if ok else None
                else:
                    m["abstain_or_refused"] = (sum(1 for r in ok if r["empty"]) + refused) / len(sub_q)
                    m["hallucination_rate"] = sum(1 for r in ok if not r["empty"]) / len(sub_q)
                m["avg_rounds"] = (sum(r["rounds"] for r in recs)
                                   + refused * 2) / len(sub_q)
                m["mean_reader_latency_s"] = (sum(r["latency_s"] for r in ok) / len(ok)) if ok else None
                m["total_prompt_tokens"] = sum(r["prompt_tokens"] or 0 for r in recs)
                m["total_completion_tokens"] = sum(r["completion_tokens"] or 0 for r in recs)
                out[f"{variant}_{arm}"] = m
        # refusal confusion (gate arm): refused unanswerable = correct
        ref_unans = sum(1 for q in sub_q if "gate" not in by.get((q, "unans"), {}))
        ref_ans = sum(1 for q in sub_q if "gate" not in by.get((q, "ans"), {}))
        n_unans = n_ans = len(sub_q)
        tp, fp = ref_unans, ref_ans
        fn, tn = n_unans - ref_unans, n_ans - ref_ans
        out["refusal"] = {
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "answerable_kept": tn / n_ans,
        }
        return out

    metrics = {
        "tau": tau, "target_precision": TARGET_PRECISION,
        "calibration_curve": [c for c in curve if c["tau"] * 100 % 5 == 0],
        "timing_s": {"gate_r1": t_gate_r1, "retry_rel": t_retry_rel,
                     "gate_r2": t_gate_r2,
                     "v3_total": t_gate_r1 + t_retry_rel + t_gate_r2,
                     "note": "measured in the first run; reused on resume"},
        "v3_items": {"gate_r1": 586, "retry_rel": 29265, "gate_r2": 586},
        "test": split_metrics("test"),
        "dev": split_metrics("dev"),
        "calibration": split_metrics("calibration"),
    }
    with open(OUT / "gate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "calibration_curve"}, indent=2))
    print(f"total wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
