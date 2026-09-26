"""Iterative RAG with the 4B LoRA answerability gate (task 4 follow-up).

Re-runs ONLY the gate phase of run_iterative.py: retrieval, 4B relevance
scores and 27B subqueries are reused verbatim from the v3-gate probe run
(iter_retrieval.jsonl / iter_scores.jsonl / iter_subqueries.jsonl), so the
evidence trajectories are identical and any tau's arm trajectory is again a
pure function of the recorded per-round gate probabilities.

Gate model: ckpt/qwen3_4b_lora_v1/step2500 (answerability channel evaluated in
local_runs/lora4b_20260924/evaluation/answerability_scores.jsonl). tau is
recalibrated on the same 98 calibration questions with the same
answered-precision >= 0.90 rule. Reader calls are reused from
iter_predictions.jsonl when (arm, rounds, doc_ids) match exactly (the reader
is deterministic: temperature 0, fixed seed); only changed trajectories hit
8020.

Outputs (separate from the v3-gate run): iter_scores_4bgate.jsonl,
iter_predictions_4bgate.jsonl, iter_metrics_4bgate.json. Resume-safe at the
gate-item and prediction level. GPU: CUDA_VISIBLE_DEVICES=3.
"""
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os  # noqa: E402

from rag_eval.common import PROP_ANSWERABLE, load_corpus, load_questions, passage_block  # noqa: E402
from rag_eval.fusion import fuse_rankings  # noqa: E402
from rag_eval.iterative_rag import ARMS, calibrate_tau_iter, decide  # noqa: E402
from rag_eval.run_iterative import (FUSION_CONFIG, MAX_ROUNDS, NONTRAIN, POOL,  # noqa: E402
                                    TARGET_PRECISION, TOPK, VARIANTS, load_keys)
from rag_eval.run_qa import call_reader, exact_match, token_f1  # noqa: E402
from rag_eval.score_v3 import score_items  # noqa: E402

OUT = Path(os.environ.get("XIAOJEV_RAG_EVAL_DIR", Path(__file__).resolve().parent))
GATE_CKPT = os.environ.get(
    "XIAOJEV_CKPT_4B", str(ROOT / "ckpt" / "qwen3_4b_lora_v1" / "step2500")
)  # 4B LoRA as the gate
CONCURRENCY = 8

RETR = OUT / "iter_retrieval.jsonl"
SUBQ = OUT / "iter_subqueries.jsonl"
SCORES_V3 = OUT / "iter_scores.jsonl"
PREDS_V3 = OUT / "iter_predictions.jsonl"
SCORES = OUT / "iter_scores_4bgate.jsonl"
PREDS = OUT / "iter_predictions_4bgate.jsonl"
METRICS = OUT / "iter_metrics_4bgate.json"


def main():
    t_start = time.time()
    timings = {}
    questions = {q["id"]: q for q in load_questions() if q["split"] in NONTRAIN}
    qids = sorted(questions)
    splits = {qid: questions[qid]["split"] for qid in qids}
    corpus = load_corpus()
    by_docid = {c["docid"]: c for c in corpus}
    key2doc = {(c["title"], re.sub(r"\s+", " ", c["text"]).strip()): c["docid"]
               for c in corpus}
    gold = {}
    for qid, q in questions.items():
        gold[qid] = {key2doc[(t, re.sub(r"\s+", " ", x).strip())]
                     for idx, t, x in q["paragraphs"] if idx in set(q["gold_idx"])}

    # ---- reuse probe artifacts from the v3-gate run
    retr = {}
    for line in open(RETR):
        d = json.loads(line)
        retr[(d["qid"], d["variant"], d["round"])] = d["top"]
    rel = defaultdict(dict)
    for line in open(SCORES_V3):
        d = json.loads(line)
        if d["kind"] == "relevance":
            rel[(d["qid"], d["variant"], d["round"])][d["docid"]] = d["p_yes"]
    subq_usage = {}
    for line in open(SUBQ):
        d = json.loads(line)
        subq_usage[(d["qid"], d["variant"], d["round"])] = {
            "prompt_tokens": d.get("prompt_tokens") or 0,
            "completion_tokens": d.get("completion_tokens") or 0}
    print(f"probe artifacts loaded: {len(retr)} retrievals, "
          f"{sum(len(v) for v in rel.values())} relevance scores", flush=True)

    def select_evidence():
        """Identical to run_iterative.select_evidence (pure)."""
        ev_after = {}
        for qid in qids:
            for v in VARIANTS:
                seen = set(gold[qid]) if v == "unans" else set()
                ev = []
                for r in range(1, MAX_ROUNDS + 1):
                    cand = [d for d in retr[(qid, v, r)][:POOL] if d not in seen]
                    sc = rel.get((qid, v, r), {})
                    order = fuse_rankings(cand, {d: sc[d] for d in cand}, **FUSION_CONFIG)
                    top = order[:TOPK]
                    seen.update(top)
                    ev.extend(top)
                    ev_after[(qid, v, r)] = list(ev)
        return ev_after

    ev_after = select_evidence()

    # ---- 4B gate phase (item-level resume)
    scored = {k for k in load_keys(SCORES, ("kind", "qid", "variant", "round"))
              if k[0] == "gate4b"}
    items = []
    for qid in qids:
        q = questions[qid]
        for v in VARIANTS:
            for r in range(1, MAX_ROUNDS + 1):
                if ("gate4b", qid, v, r) in scored:
                    continue
                state = (f"Question: {q['question']}\n\n" + "\n".join(
                    passage_block(by_docid[d]["title"], by_docid[d]["text"])
                    for d in ev_after[(qid, v, r)]))
                row = {"id": f"i4gate_{qid}_{v}_r{r}", "primitive": "noul",
                       "state": state, "proposition": PROP_ANSWERABLE,
                       "candidates": ["yes", "no"]}
                items.append((row, {"kind": "gate4b", "qid": qid, "variant": v,
                                    "round": r}))
    if items:
        t0 = time.time()
        from lora4b_model import load as load_lora4b  # training/lora4b_model.py (needs peft)
        model, tok = load_lora4b(GATE_CKPT)
        model.eval()
        print(f"4B LoRA gate loaded ({time.time()-t_start:.0f}s)", flush=True)
        with open(SCORES, "a") as sf:
            score_items(model, tok, items, "cuda", sf)
        timings["gate4b"] = round(time.time() - t0, 1)
        print(f"4B gate scoring: {len(items)} items, {timings['gate4b']}s", flush=True)
    else:
        print("resume: 4B gate scoring complete", flush=True)

    p_gate = {}
    for line in open(SCORES):
        d = json.loads(line)
        if d["kind"] == "gate4b":
            p_gate[(d["qid"], d["variant"], d["round"])] = d.get(
                "p_yes", d.get("probs", [0.0])[0])
    gate_probs = {(qid, v): [p_gate[(qid, v, r)] for r in range(1, MAX_ROUNDS + 1)]
                  for qid in qids for v in VARIANTS}

    # ---- tau calibration on the calibration split only (same rule as v3 run)
    cal_records = [{"qid": qid, "variant": v, "probs": gate_probs[(qid, v)]}
                   for qid in qids if splits[qid] == "calibration" for v in VARIANTS]
    tau, curve = calibrate_tau_iter(cal_records, TARGET_PRECISION)
    print(f"calibrated tau={tau:.2f} (target answered-precision "
          f"{TARGET_PRECISION})", flush=True)

    # ---- trajectories per arm + reader calls (reuse v3-run predictions when
    # the deterministic reader call would be identical)
    v3_preds = {}
    if PREDS_V3.exists():
        for line in open(PREDS_V3):
            r = json.loads(line)
            v3_preds[(r["id"], r["variant"], r["arm"])] = r

    traj = {}
    tasks, reused = [], []
    for qid in qids:
        for v in VARIANTS:
            for arm_name, arm in ARMS.items():
                n, stop = decide(gate_probs[(qid, v)], arm, tau)
                traj[(qid, v, arm_name)] = {"rounds": n, "stop": stop}
                if stop == "exhaust_refuse":
                    continue
                doc_ids = ev_after[(qid, v, n)]
                old = v3_preds.get((qid, v, arm_name))
                if (old is not None and old["rounds"] == n
                        and old["doc_ids"] == doc_ids and old["em"] is not None):
                    reused.append((qid, v, arm_name, n, stop, old))
                else:
                    tasks.append((qid, v, arm_name, n, stop))
    done_keys = load_keys(PREDS, ("id", "variant", "arm"))
    reused = [t for t in reused if (t[0], t[1], t[2]) not in done_keys]
    tasks = [t for t in tasks if (t[0], t[1], t[2]) not in done_keys]
    print(f"reader: {len(reused)} reused from v3 run, {len(tasks)} fresh calls",
          flush=True)

    preds_f = open(PREDS, "a")
    lock = threading.Lock()

    def make_rec(qid, v, arm, n, stop, answer, usage, dt):
        q = questions[qid]
        aliases = [q["answer"]] + q["answer_aliases"]
        prop = {"prompt_tokens": 0, "completion_tokens": 0}
        for r in range(1, n):
            u = subq_usage.get((qid, v, r))
            if u:
                prop["prompt_tokens"] += u["prompt_tokens"]
                prop["completion_tokens"] += u["completion_tokens"]
        return {"id": qid, "variant": v, "arm": arm, "hop": q["hop"],
                "split": splits[qid], "rounds": n, "stop": stop,
                "gate_probs": gate_probs[(qid, v)][:n],
                "prediction": answer,
                "doc_ids": ev_after[(qid, v, n)],
                "em": None if answer is None else exact_match(answer or "", aliases),
                "f1": None if answer is None else token_f1(answer or "", aliases),
                "empty": answer == "",
                "latency_s": round(dt, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "proposer_prompt_tokens": prop["prompt_tokens"],
                "proposer_completion_tokens": prop["completion_tokens"]}

    with lock:
        for qid, v, arm, n, stop, old in reused:
            rec = dict(old)
            rec.update({"rounds": n, "stop": stop,
                        "gate_probs": gate_probs[(qid, v)][:n],
                        "reused_from_v3_run": True})
            preds_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        preds_f.flush()

    done = [0]
    t_reader = time.time()

    def work(task):
        qid, v, arm, n, stop = task
        q = questions[qid]
        passages = [(d, by_docid[d]["text"]) for d in ev_after[(qid, v, n)]]
        t_call = time.time()
        answer, usage = call_reader(q["question"], passages)
        dt = time.time() - t_call
        rec = make_rec(qid, v, arm, n, stop, answer, usage, dt)
        with lock:
            preds_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 100 == 0:
                preds_f.flush()
                print(f"{done[0]}/{len(tasks)} reader calls, "
                      f"{time.time()-t_reader:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(work, tasks))
    preds_f.close()
    timings["reader"] = round(time.time() - t_reader, 1)

    # ---- metrics (same shape as run_iterative.py)
    rows = [json.loads(l) for l in open(PREDS)]
    by = defaultdict(dict)
    for r in rows:
        by[(r["id"], r["variant"])][r["arm"]] = r

    def split_metrics(split):
        sub_q = [q for q in qids if splits[q] == split]
        out = {"n_questions": len(sub_q)}
        for v in VARIANTS:
            for arm in ARMS:
                recs = [by[(q, v)][arm] for q in sub_q if arm in by.get((q, v), {})]
                refused = len(sub_q) - len(recs)
                ok = [r for r in recs if r["em"] is not None]
                m = {"answered": len(recs), "refused": refused,
                     "failures": len(recs) - len(ok)}
                if v == "ans":
                    m["em"] = sum(r["em"] for r in ok) / len(sub_q)
                    m["f1"] = sum(r["f1"] for r in ok) / len(sub_q)
                    m["em_answered"] = sum(r["em"] for r in ok) / len(ok) if ok else None
                    m["f1_answered"] = sum(r["f1"] for r in ok) / len(ok) if ok else None
                else:
                    m["abstain_or_refused"] = (sum(1 for r in ok if r["empty"])
                                               + refused) / len(sub_q)
                    m["hallucination_rate"] = (sum(1 for r in ok if not r["empty"])
                                               / len(sub_q))
                m["avg_rounds"] = (sum(r["rounds"] for r in recs)
                                   + sum(traj[(q, v, arm)]["rounds"]
                                         for q in sub_q
                                         if arm not in by.get((q, v), {}))) / len(sub_q)
                m["mean_reader_latency_s"] = (sum(r["latency_s"] for r in ok) / len(ok)
                                              if ok else None)
                m["total_prompt_tokens"] = (sum(r["prompt_tokens"] or 0 for r in recs)
                                            + sum(r["proposer_prompt_tokens"] for r in recs))
                m["total_completion_tokens"] = (sum(r["completion_tokens"] or 0 for r in recs)
                                                + sum(r["proposer_completion_tokens"] for r in recs))
                out[f"{v}_{arm}"] = m
        ref_unans = sum(1 for q in sub_q if "gated_refuse" not in by.get((q, "unans"), {}))
        ref_ans = sum(1 for q in sub_q if "gated_refuse" not in by.get((q, "ans"), {}))
        n = len(sub_q)
        out["refusal"] = {
            "precision": ref_unans / (ref_unans + ref_ans) if ref_unans + ref_ans else None,
            "recall": ref_unans / n,
            "answerable_kept": (n - ref_ans) / n,
        }
        # gate trajectory distribution (gated arm): per-round acceptance counts
        traj_dist = {}
        for v in VARIANTS:
            d = {f"accept_r{r}": 0 for r in range(1, MAX_ROUNDS + 1)}
            d["exhaust"] = 0
            for q in sub_q:
                t = traj[(q, v, "gated")]
                if t["stop"] == "gate_accept":
                    d[f"accept_r{t['rounds']}"] += 1
                else:
                    d["exhaust"] += 1
            traj_dist[v] = d
        out["gate_trajectory"] = traj_dist
        return out

    metrics = {
        "tau": tau, "target_precision": TARGET_PRECISION,
        "gate": str(GATE_CKPT), "gate_kind": "gate4b",
        "note": "retrieval/relevance/subquery artifacts reused from the v3-gate "
                "probe run; only the gate phase was re-scored",
        "fusion_config": FUSION_CONFIG,
        "arms": {name: dict(arm.__dict__) for name, arm in ARMS.items()},
        "calibration_curve": [c for c in curve if c["tau"] * 100 % 5 == 0],
        "timing_s": timings,
        "test": split_metrics("test"),
        "dev": split_metrics("dev"),
        "calibration": split_metrics("calibration"),
    }
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["test"], indent=2))
    print(f"total wall time: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
