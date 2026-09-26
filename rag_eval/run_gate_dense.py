"""Dense first stage + v3 rerank top-5 + answerability gate (driver).

Same protocol as run_gate.py (BM25 version): 293 non-train questions x
{ans, unans} variants, tau calibrated on the 98-question calibration split at
answered-precision >= 0.90, frozen, then reported on test (primary) and dev.

Outputs: gate_dense_scores.jsonl, gate_dense_predictions.jsonl,
gate_dense_metrics.json. Resume-safe at both v3-scoring and reader levels.

Configuration (environment variables):
  XIAOJEV_GATE_DIR   score/output directory (default: this rag_eval/ directory)
  XIAOJEV_CKPT_V3    v3 checkpoint dir (default: <repo>/ckpt/v3)
"""
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
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

from rag_eval.gate import OUT, calibrate_tau  # noqa: E402
from rag_eval.gate_dense import DenseGatePipeline  # noqa: E402
from rag_eval.run_qa import call_reader, exact_match, token_f1  # noqa: E402
from rag_eval.score_v3 import score_items  # noqa: E402

CKPT = os.environ.get("XIAOJEV_CKPT_V3", str(ROOT / "ckpt" / "v3"))
TARGET_PRECISION = 0.90
SCORES = OUT / "gate_dense_scores.jsonl"
PREDS = OUT / "gate_dense_predictions.jsonl"


def main():
    t_start = time.time()
    device = "cuda"
    pipe = DenseGatePipeline()
    qids = sorted(pipe.questions)
    splits = {qid: pipe.questions[qid]["split"] for qid in qids}
    print(f"questions: {len(qids)}; dense pools ready ({time.time()-t_start:.0f}s)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    def score_counts():
        c = Counter()
        if SCORES.exists():
            for line in open(SCORES):
                d = json.loads(line)
                c[(d["kind"], d.get("round"))] += 1
        return c

    def load_scores():
        p_gate, rel_retry = {}, defaultdict(dict)
        for line in open(SCORES):
            d = json.loads(line)
            v = d.get("p_yes", d.get("probs", [0.0])[0])
            if d["kind"] == "gate":
                p_gate[(d["qid"], d["variant"], d["round"])] = v
            elif d["kind"] == "retry_rel":
                rel_retry[(d["qid"], d["variant"])][d["docid"]] = v
        return p_gate, rel_retry

    counts = score_counts()
    scoring_done = (counts[("gate", 1)] == 586 and counts[("gate", 2)] == 586
                    and counts[("retry_rel", None)] > 0)
    if scoring_done:
        print(f"resume: v3 scoring complete {dict(counts)}", flush=True)
        p_gate, rel_retry = load_scores()
        t_gate_r1 = t_retry_rel = t_gate_r2 = None
    else:
        model = StudentModel().to(device)
        load_ckpt(model, CKPT)
        model.eval()
        print("v3 loaded", flush=True)
        with open(SCORES, "w") as sf:
            t0 = time.time()
            items = [pipe.gate_item(qid, v, 1, pipe.topk_passages(qid, v))
                     for qid in qids for v in ("ans", "unans")]
            score_items(model, tok, items, device, sf)
            t_gate_r1 = time.time() - t0
            print(f"phase1 gate r1: {len(items)} items, {t_gate_r1:.0f}s", flush=True)

            t0 = time.time()
            retry_items = pipe.retry_rel_items([(q, v) for q in qids for v in ("ans", "unans")])
            score_items(model, tok, retry_items, device, sf)
            t_retry_rel = time.time() - t0
            print(f"phase2 retry rel: {len(retry_items)} items, {t_retry_rel:.0f}s", flush=True)

        _, rel_retry = load_scores()
        t0 = time.time()
        items2 = [pipe.gate_item(qid, v, 2,
                                 pipe.topk_passages(qid, v, depth=100,
                                                    extra_scores=rel_retry.get((qid, v), {})))
                  for qid in qids for v in ("ans", "unans")]
        with open(SCORES, "a") as sf:
            score_items(model, tok, items2, device, sf)
        t_gate_r2 = time.time() - t0
        print(f"phase3 gate r2: {len(items2)} items, {t_gate_r2:.0f}s", flush=True)
        p_gate, rel_retry = load_scores()

    topk_r1 = {(q, v): pipe.topk_passages(q, v) for q in qids for v in ("ans", "unans")}
    topk_r2 = {(q, v): pipe.topk_passages(q, v, depth=100,
                                          extra_scores=rel_retry.get((q, v), {}))
               for q in qids for v in ("ans", "unans")}

    cal_records = [{"qid": q, "variant": v, "p1": p_gate[(q, v, 1)], "p2": p_gate[(q, v, 2)]}
                   for q in qids if splits[q] == "calibration" for v in ("ans", "unans")]
    tau, curve = calibrate_tau(cal_records, TARGET_PRECISION)
    print(f"calibrated tau={tau:.2f}", flush=True)

    def accepted(q, v):
        return p_gate[(q, v, 1)] >= tau or p_gate[(q, v, 2)] >= tau

    def final_passages(q, v):
        return (topk_r1[(q, v)], 1) if p_gate[(q, v, 1)] >= tau else (topk_r2[(q, v)], 2)

    tasks = []
    for q in qids:
        for v in ("ans", "unans"):
            tasks.append((q, v, "nogate", topk_r1[(q, v)], 1))
            if accepted(q, v):
                ps, rnd = final_passages(q, v)
                tasks.append((q, v, "gate", ps, rnd))
    done_keys = set()
    if PREDS.exists():
        for line in open(PREDS):
            r = json.loads(line)
            done_keys.add((r["id"], r["variant"], r["arm"]))
    tasks = [t for t in tasks if (t[0], t[1], t[2]) not in done_keys]
    print(f"reader calls remaining: {len(tasks)}", flush=True)

    preds_f = open(PREDS, "a")
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
               "prediction": answer, "doc_ids": [p for p, _ in passages],
               "em": None if answer is None else exact_match(answer or "", aliases),
               "f1": None if answer is None else token_f1(answer or "", aliases),
               "empty": answer == "", "latency_s": round(dt, 3),
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens")}
        with lock:
            preds_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 100 == 0:
                preds_f.flush()
                print(f"{done[0]}/{len(tasks)}, {time.time()-t_reader:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, tasks))
    preds_f.close()

    rows = [json.loads(l) for l in open(PREDS)]
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
                    m["em"] = sum(r["em"] for r in ok) / len(sub_q)
                    m["f1"] = sum(r["f1"] for r in ok) / len(sub_q)
                    m["em_answered"] = sum(r["em"] for r in ok) / len(ok) if ok else None
                    m["f1_answered"] = sum(r["f1"] for r in ok) / len(ok) if ok else None
                else:
                    m["abstain_or_refused"] = (sum(1 for r in ok if r["empty"]) + refused) / len(sub_q)
                    m["hallucination_rate"] = sum(1 for r in ok if not r["empty"]) / len(sub_q)
                m["avg_rounds"] = (sum(r["rounds"] for r in recs) + refused * 2) / len(sub_q)
                m["mean_reader_latency_s"] = (sum(r["latency_s"] for r in ok) / len(ok)) if ok else None
                m["total_prompt_tokens"] = sum(r["prompt_tokens"] or 0 for r in recs)
                m["total_completion_tokens"] = sum(r["completion_tokens"] or 0 for r in recs)
                out[f"{variant}_{arm}"] = m
        ref_unans = sum(1 for q in sub_q if "gate" not in by.get((q, "unans"), {}))
        ref_ans = sum(1 for q in sub_q if "gate" not in by.get((q, "ans"), {}))
        n = len(sub_q)
        out["refusal"] = {
            "precision": ref_unans / (ref_unans + ref_ans) if ref_unans + ref_ans else None,
            "recall": ref_unans / n,
            "answerable_kept": (n - ref_ans) / n,
        }
        return out

    # gate AUC per split (round-1 and after-retry)
    def auc(pairs):
        pairs = sorted(pairs, key=lambda x: -x[0])
        pos = sum(l for _, l in pairs)
        neg = len(pairs) - pos
        rs = sum(i + 1 for i, (_, l) in enumerate(pairs) if l)
        return 1.0 - (rs - pos * (pos + 1) / 2) / (pos * neg)

    aucs = {}
    for split in ("calibration", "dev", "test"):
        for rnd in ("r1", "r12"):
            pairs = []
            for q in qids:
                if splits[q] != split:
                    continue
                for v in ("ans", "unans"):
                    s = p_gate[(q, v, 1)] if rnd == "r1" else max(p_gate[(q, v, 1)], p_gate[(q, v, 2)])
                    pairs.append((s, 1 if v == "ans" else 0))
            aucs[f"{split}_{rnd}"] = round(auc(pairs), 4)

    metrics = {
        "tau": tau, "target_precision": TARGET_PRECISION, "first_stage": "dense_nv_embed_v2_top50",
        "calibration_curve": [c for c in curve if c["tau"] * 100 % 5 == 0],
        "gate_auc": aucs,
        "timing_s": {"gate_r1": t_gate_r1, "retry_rel": t_retry_rel, "gate_r2": t_gate_r2},
        "test": split_metrics("test"),
        "dev": split_metrics("dev"),
        "calibration": split_metrics("calibration"),
    }
    with open(OUT / "gate_dense_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["test"], indent=2))
    print(f"total wall time: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
