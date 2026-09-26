"""Calibration-gated iterative RAG evaluation (task 4 driver).

Protocol identical to run_gate_dense.py (293 non-train musique questions x
{ans, unans}; unans bans the question's gold docids from every candidate
stage; tau calibrated on the 98-question calibration split at answered-set
precision >= 0.90, frozen, then reported on test (primary) and dev).

Pipeline per round (probe mode: every question runs all MAX_ROUNDS rounds so
that any tau's trajectory is a pure function of the recorded per-round gate
probabilities):
  dense top-50 (round 1: original question; later rounds: 27B subquery)
  -> 4B LoRA relevance scoring -> weighted-RRF fusion (frozen fusion4b config)
  -> top-5 accumulated into the evidence set (dedup across rounds)
  -> v3 answerability gate on the accumulated evidence

Arms (derived from the same probe artifacts): single / fixed2 / gated /
gated_refuse (see iterative_rag.ARMS). Reader is called only where the arm's
policy answers; gated_refuse refusals are implicit (no prediction row).

Outputs: iter_retrieval.jsonl, iter_subqueries.jsonl, iter_scores.jsonl,
iter_predictions.jsonl, iter_metrics.json. Every phase is resume-safe
(item-level for scores, key-level elsewhere). GPU: CUDA_VISIBLE_DEVICES=3.
"""
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "rag_eval"), str(ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag_eval.common import PROP_ANSWERABLE, PROP_RELEVANCE, load_corpus, load_questions, passage_block  # noqa: E402
from rag_eval.fusion import fuse_rankings  # noqa: E402
from rag_eval.iterative_rag import (ARMS, DenseRetriever, SubqueryProposer,  # noqa: E402
                                    calibrate_tau_iter, decide)
from rag_eval.run_qa import call_reader, exact_match, token_f1  # noqa: E402
from rag_eval.score_v3 import score_items  # noqa: E402

OUT = Path(os.environ.get("XIAOJEV_RAG_EVAL_DIR", Path(__file__).resolve().parent))
NONTRAIN = ("dev", "calibration", "test")
RERANK_CKPT = os.environ.get(
    "XIAOJEV_CKPT_4B", str(ROOT / "ckpt" / "qwen3_4b_lora_v1" / "step2500")
)  # 4B LoRA, fusion rerank
GATE_CKPT = os.environ.get(
    "XIAOJEV_CKPT_V3", str(ROOT / "ckpt" / "v3")
)  # 0.6B, answerability gate
FUSION_CONFIG = json.load(open(OUT / "fusion4b_config.json"))["config"]
MAX_ROUNDS = 3
TOPK = 5
POOL = 50
TARGET_PRECISION = 0.90
CONCURRENCY = 8
VARIANTS = ("ans", "unans")

RETR = OUT / "iter_retrieval.jsonl"
SUBQ = OUT / "iter_subqueries.jsonl"
SCORES = OUT / "iter_scores.jsonl"
PREDS = OUT / "iter_predictions.jsonl"
METRICS = OUT / "iter_metrics.json"


def load_keys(path, fields):
    keys = set()
    if path.exists():
        for line in open(path):
            d = json.loads(line)
            keys.add(tuple(d.get(f) for f in fields))
    return keys


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
    print(f"questions: {len(qids)} ({time.time()-t_start:.0f}s)", flush=True)

    retriever = DenseRetriever(corpus)
    proposer = SubqueryProposer()
    print(f"dense index ready ({time.time()-t_start:.0f}s)", flush=True)

    models = {}

    def rerank_model():
        if "rerank" not in models:
            from lora4b_model import load as load_lora4b  # training/lora4b_model.py (needs peft)
            models["rerank"] = load_lora4b(RERANK_CKPT)
            print(f"4B LoRA loaded ({time.time()-t_start:.0f}s)", flush=True)
        return models["rerank"]

    def gate_model():
        if "gate" not in models:
            from transformers import AutoTokenizer
            from train import MODEL_PATH, StudentModel, load_ckpt
            tok = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = StudentModel().to("cuda")
            load_ckpt(model, str(GATE_CKPT))
            model.eval()
            models["gate"] = (model, tok)
            print(f"v3 gate loaded ({time.time()-t_start:.0f}s)", flush=True)
        return models["gate"]

    subqueries = {}  # (qid, variant, round) -> subquery text (None allowed)

    def load_subqueries():
        subqueries.clear()
        if SUBQ.exists():
            for line in open(SUBQ):
                d = json.loads(line)
                subqueries[(d["qid"], d["variant"], d["round"])] = d["subquery"]

    retr = {}  # (qid, variant, round) -> raw dense top-(POOL+overshoot) docids

    def load_retr():
        retr.clear()
        if RETR.exists():
            for line in open(RETR):
                d = json.loads(line)
                retr[(d["qid"], d["variant"], d["round"])] = d["top"]

    def retrieval_phase_pv(rnd):
        todo = []
        for qid in qids:
            for v in VARIANTS:
                if (qid, v, rnd) not in retr:
                    todo.append((qid, v))
        if not todo:
            return
        t0 = time.time()
        queries = [questions[qid]["question"] if rnd == 1
                   else subqueries.get((qid, v, rnd - 1)) or questions[qid]["question"]
                   for qid, v in todo]
        top = retriever.retrieve_batch(queries, k=POOL)
        with open(RETR, "a") as f:
            for (qid, v), query, docids in zip(todo, queries, top):
                f.write(json.dumps({"qid": qid, "variant": v, "round": rnd,
                                    "query": query, "top": docids}) + "\n")
        load_retr()
        timings[f"retrieval_r{rnd}"] = round(time.time() - t0, 1)
        print(f"retrieval r{rnd}: {len(todo)} pipelines, "
              f"{timings[f'retrieval_r{rnd}']}s", flush=True)

    def relevance_phase(rnd):
        """4B LoRA relevance scoring of the round's candidates (gold-banned
        docids excluded for unans; item-level resume)."""
        scored = load_keys(SCORES, ("kind", "qid", "variant", "round", "docid"))
        items = []
        for qid in qids:
            q = questions[qid]
            for v in VARIANTS:
                banned = gold[qid] if v == "unans" else set()
                for d in retr[(qid, v, rnd)][:POOL]:
                    if d in banned or ("relevance", qid, v, rnd, d) in scored:
                        continue
                    c = by_docid[d]
                    row = {"id": f"irel_{qid}_{v}_r{rnd}_{d}", "primitive": "noul",
                           "state": f"Question: {q['question']}\n\n"
                                    + passage_block(c["title"], c["text"]),
                           "proposition": PROP_RELEVANCE, "candidates": ["yes", "no"]}
                    items.append((row, {"kind": "relevance", "qid": qid, "variant": v,
                                        "round": rnd, "docid": d}))
        if not items:
            return
        t0 = time.time()
        model, tok = rerank_model()
        with open(SCORES, "a") as sf:
            score_items(model, tok, items, "cuda", sf)
        timings[f"relevance_r{rnd}"] = round(time.time() - t0, 1)
        print(f"relevance r{rnd}: {len(items)} items, "
              f"{timings[f'relevance_r{rnd}']}s", flush=True)

    def load_scores():
        rel, p_gate = defaultdict(dict), {}
        if SCORES.exists():
            for line in open(SCORES):
                d = json.loads(line)
                v = d.get("p_yes", d.get("probs", [0.0])[0])
                if d["kind"] == "relevance":
                    rel[(d["qid"], d["variant"], d["round"])][d["docid"]] = v
                elif d["kind"] == "gate":
                    p_gate[(d["qid"], d["variant"], d["round"])] = v
        return rel, p_gate

    def select_evidence(rel, upto):
        """Pure: fused top-k per round, excluding banned and previously
        accumulated docids. Returns evidence docids after each round."""
        ev_after = {}
        for qid in qids:
            for v in VARIANTS:
                seen = set(gold[qid]) if v == "unans" else set()
                ev = []
                for r in range(1, upto + 1):
                    cand = [d for d in retr[(qid, v, r)][:POOL] if d not in seen]
                    sc = rel.get((qid, v, r), {})
                    order = fuse_rankings(cand, {d: sc[d] for d in cand}, **FUSION_CONFIG)
                    top = order[:TOPK]
                    seen.update(top)
                    ev.extend(top)
                    ev_after[(qid, v, r)] = list(ev)
        return ev_after

    def gate_phase(rnd, ev_after):
        """v3 answerability scoring on the accumulated evidence (item-level
        resume)."""
        scored = load_keys(SCORES, ("kind", "qid", "variant", "round"))
        scored = {k for k in scored if k[0] == "gate"}
        items = []
        for qid in qids:
            q = questions[qid]
            for v in VARIANTS:
                if ("gate", qid, v, rnd) in scored:
                    continue
                state = (f"Question: {q['question']}\n\n" + "\n".join(
                    passage_block(by_docid[d]["title"], by_docid[d]["text"])
                    for d in ev_after[(qid, v, rnd)]))
                row = {"id": f"igate_{qid}_{v}_r{rnd}", "primitive": "noul",
                       "state": state, "proposition": PROP_ANSWERABLE,
                       "candidates": ["yes", "no"]}
                items.append((row, {"kind": "gate", "qid": qid, "variant": v,
                                    "round": rnd}))
        if not items:
            return
        t0 = time.time()
        model, tok = gate_model()
        with open(SCORES, "a") as sf:
            score_items(model, tok, items, "cuda", sf)
        timings[f"gate_r{rnd}"] = round(time.time() - t0, 1)
        print(f"gate r{rnd}: {len(items)} items, {timings[f'gate_r{rnd}']}s", flush=True)

    def subquery_phase(rnd, ev_after):
        """27B follow-up subqueries after round rnd (used by round rnd+1)."""
        existing = load_keys(SUBQ, ("qid", "variant", "round"))
        todo = [(qid, v) for qid in qids for v in VARIANTS
                if (qid, v, rnd) not in existing]
        if not todo:
            return
        t0 = time.time()
        f = open(SUBQ, "a")
        lock = threading.Lock()
        done = [0]

        def work(qv):
            qid, v = qv
            ps = [{"docid": d, "title": by_docid[d]["title"],
                   "text": by_docid[d]["text"]} for d in ev_after[(qid, v, rnd)]]
            subquery, usage = proposer.propose(questions[qid]["question"], ps, rnd)
            rec = {"qid": qid, "variant": v, "round": rnd, "subquery": subquery,
                   "prompt_tokens": usage.get("prompt_tokens"),
                   "completion_tokens": usage.get("completion_tokens")}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[0] += 1
                if done[0] % 100 == 0:
                    f.flush()
                    print(f"  subqueries r{rnd}: {done[0]}/{len(todo)}, "
                          f"{time.time()-t0:.0f}s", flush=True)

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            list(ex.map(work, todo))
        f.close()
        load_subqueries()
        timings[f"subquery_r{rnd}"] = round(time.time() - t0, 1)
        print(f"subquery r{rnd}: {len(todo)} calls, "
              f"{timings[f'subquery_r{rnd}']}s", flush=True)

    # ---- probe phases: run all rounds for every question
    load_subqueries()
    load_retr()
    for rnd in range(1, MAX_ROUNDS + 1):
        retrieval_phase_pv(rnd)
        relevance_phase(rnd)
        rel, _ = load_scores()
        ev_after = select_evidence(rel, rnd)
        gate_phase(rnd, ev_after)
        if rnd < MAX_ROUNDS:
            subquery_phase(rnd, ev_after)

    rel, p_gate = load_scores()
    ev_after = select_evidence(rel, MAX_ROUNDS)
    gate_probs = {(qid, v): [p_gate[(qid, v, r)] for r in range(1, MAX_ROUNDS + 1)]
                  for qid in qids for v in VARIANTS}

    # ---- tau calibration on the calibration split only
    cal_records = [{"qid": qid, "variant": v, "probs": gate_probs[(qid, v)]}
                   for qid in qids if splits[qid] == "calibration" for v in VARIANTS]
    tau, curve = calibrate_tau_iter(cal_records, TARGET_PRECISION)
    print(f"calibrated tau={tau:.2f} (target answered-precision "
          f"{TARGET_PRECISION})", flush=True)

    # ---- trajectories per arm (pure) + reader calls where the arm answers
    traj = {}
    tasks = []
    for qid in qids:
        for v in VARIANTS:
            for arm_name, arm in ARMS.items():
                n, stop = decide(gate_probs[(qid, v)], arm, tau)
                traj[(qid, v, arm_name)] = {"rounds": n, "stop": stop}
                if stop != "exhaust_refuse":
                    tasks.append((qid, v, arm_name, n, stop))
    done_keys = load_keys(PREDS, ("id", "variant", "arm"))
    tasks = [t for t in tasks if (t[0], t[1], t[2]) not in done_keys]
    print(f"reader calls remaining after resume: {len(tasks)}", flush=True)

    # proposer token usage per (qid, variant): rounds 1..n-1 feed the trajectory
    subq_usage = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    if SUBQ.exists():
        for line in open(SUBQ):
            d = json.loads(line)
            k = (d["qid"], d["variant"], d["round"])
            subq_usage[k] = {"prompt_tokens": d.get("prompt_tokens") or 0,
                             "completion_tokens": d.get("completion_tokens") or 0}

    preds_f = open(PREDS, "a")
    lock = threading.Lock()
    done = [0]
    t_reader = time.time()

    def work(task):
        qid, v, arm, n, stop = task
        q = questions[qid]
        passages = [(d, by_docid[d]["text"]) for d in ev_after[(qid, v, n)]]
        t_call = time.time()
        answer, usage = call_reader(q["question"], passages)
        dt = time.time() - t_call
        aliases = [q["answer"]] + q["answer_aliases"]
        prop = {"prompt_tokens": 0, "completion_tokens": 0}
        for r in range(1, n):
            u = subq_usage.get((qid, v, r))
            if u:
                prop["prompt_tokens"] += u["prompt_tokens"]
                prop["completion_tokens"] += u["completion_tokens"]
        rec = {"id": qid, "variant": v, "arm": arm, "hop": q["hop"],
               "split": splits[qid], "rounds": n, "stop": stop,
               "gate_probs": gate_probs[(qid, v)][:n],
               "prediction": answer,
               "doc_ids": [p for p, _ in passages],
               "em": None if answer is None else exact_match(answer or "", aliases),
               "f1": None if answer is None else token_f1(answer or "", aliases),
               "empty": answer == "",
               "latency_s": round(dt, 3),
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens"),
               "proposer_prompt_tokens": prop["prompt_tokens"],
               "proposer_completion_tokens": prop["completion_tokens"]}
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

    # ---- metrics
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
        # refusal confusion (gated_refuse arm): refused unanswerable = correct
        ref_unans = sum(1 for q in sub_q if "gated_refuse" not in by.get((q, "unans"), {}))
        ref_ans = sum(1 for q in sub_q if "gated_refuse" not in by.get((q, "ans"), {}))
        n = len(sub_q)
        out["refusal"] = {
            "precision": ref_unans / (ref_unans + ref_ans) if ref_unans + ref_ans else None,
            "recall": ref_unans / n,
            "answerable_kept": (n - ref_ans) / n,
        }
        return out

    metrics = {
        "tau": tau, "target_precision": TARGET_PRECISION,
        "first_stage": "dense_nv_embed_v2_top50_per_round",
        "reranker": str(RERANK_CKPT), "gate": str(GATE_CKPT),
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
