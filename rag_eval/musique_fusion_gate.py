"""Musique reference point for the transfer pipeline: frozen 4B-RRF fusion top-5
-> v3 answerability gate (single round, no retry). Same code path as
transfer.py so the comparison is exact.

Uses existing fusion4b_rankings.json (frozen w=0.5/c=1). tau calibrated on the
musique calibration split (98) at answered-precision 0.90; test (101) reported.

Outputs: transfer/musique_fusiongate_scores.jsonl, _predictions.jsonl,
_metrics.json. Resume-safe. GPU: CUDA_VISIBLE_DEVICES=3.
"""
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag_eval.common import PROP_ANSWERABLE, load_corpus, load_questions, passage_block  # noqa: E402
from rag_eval.gate import calibrate_tau  # noqa: E402
from rag_eval.run_qa import call_reader, exact_match, token_f1  # noqa: E402

RAG_EVAL = Path(os.environ.get("XIAOJEV_RAG_EVAL_DIR", Path(__file__).resolve().parent))
OUT = Path(os.environ.get("XIAOJEV_TRANSFER_DIR", RAG_EVAL / "transfer"))
CKPT_V3 = os.environ.get("XIAOJEV_CKPT_V3", str(ROOT / "ckpt" / "v3"))
NONTRAIN = ("dev", "calibration", "test")
TARGET = 0.90


def main():
    questions = {q["id"]: q for q in load_questions() if q["split"] in NONTRAIN}
    corpus = {c["docid"]: c for c in load_corpus()}
    rankings = json.load(open(RAG_EVAL / "fusion4b_rankings.json"))
    # gold docids via bm25_corpus.jsonl (already mapped)
    gold = {r["id"]: set(r["gold_docids"])
            for r in map(json.loads, open(RAG_EVAL / "bm25_corpus.jsonl"))
            if r["id"] in questions}
    qids = sorted(questions)
    splits = {q: questions[q]["split"] for q in qids}

    # ---- v3 gate scores (single round)
    spath = OUT / "musique_fusiongate_scores.jsonl"
    want = {(q, v) for q in qids for v in ("ans", "unans")}
    have = set()
    if spath.exists():
        for r in map(json.loads, open(spath)):
            have.add((r["qid"], r["variant"]))
    todo = sorted(want - have)
    if todo:
        from train import MODEL_PATH, StudentModel, load_ckpt
        from rag_eval.score_v3 import score_items
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = StudentModel().to("cuda")
        load_ckpt(model, CKPT_V3)
        model.eval()
        items = []
        for qid, variant in todo:
            docs = rankings[qid]
            if variant == "unans":
                docs = [d for d in docs if d not in gold[qid]]
            state = (f"Question: {questions[qid]['question']}\n\n"
                     + "\n".join(passage_block(corpus[d]["title"], corpus[d]["text"])
                                 for d in docs[:5]))
            row = {"id": f"mqfg_{qid}_{variant}", "primitive": "noul", "state": state,
                   "proposition": PROP_ANSWERABLE, "candidates": ["yes", "no"]}
            items.append((row, {"kind": "gate", "qid": qid, "variant": variant, "round": 1}))
        t0 = time.time()
        with open(spath, "a") as f:
            score_items(model, tok, items, "cuda", f)
        print(f"gate scoring: {len(items)} items, {time.time()-t0:.0f}s", flush=True)
    p = {(r["qid"], r["variant"]): r.get("p_yes", r.get("probs", [0.0])[0])
         for r in map(json.loads, open(spath))}

    cal = [{"qid": q, "variant": v, "p1": p[(q, v)], "p2": p[(q, v)]}
           for q in qids if splits[q] == "calibration" for v in ("ans", "unans")]
    tau, curve = calibrate_tau(cal, TARGET)
    print(f"tau={tau:.2f}", flush=True)

    # ---- reader: hallucination and answered quality
    ppath = OUT / "musique_fusiongate_predictions.jsonl"
    done = set()
    if ppath.exists():
        for r in map(json.loads, open(ppath)):
            done.add((r["id"], r["arm"]))
    tasks = []
    for qid in qids:
        ans_ps = [(d, corpus[d]["text"]) for d in rankings[qid][:5]]
        unans_order = [d for d in rankings[qid] if d not in gold[qid]]
        unans_ps = [(d, corpus[d]["text"]) for d in unans_order[:5]]
        for arm, ps in (("nogate_ans", ans_ps), ("nogate_unans", unans_ps)):
            if (qid, arm) not in done:
                tasks.append((qid, arm, ps))
        if p[(qid, "ans")] >= tau and (qid, "gate_ans") not in done:
            tasks.append((qid, "gate_ans", ans_ps))
        if p[(qid, "unans")] >= tau and (qid, "gate_unans") not in done:
            tasks.append((qid, "gate_unans", unans_ps))
    print(f"reader calls remaining: {len(tasks)}", flush=True)
    lock = threading.Lock()
    pf = open(ppath, "a")
    n = [0]
    t0 = time.time()

    def work(task):
        qid, arm, ps = task
        q = questions[qid]
        t_call = time.time()
        answer, usage = call_reader(q["question"], ps)
        aliases = [q["answer"]] + q["answer_aliases"]
        rec = {"id": qid, "arm": arm, "split": splits[qid],
               "variant": "unans" if "unans" in arm else "ans",
               "prediction": answer,
               "em": None if answer is None else exact_match(answer or "", aliases),
               "f1": None if answer is None else token_f1(answer or "", aliases),
               "empty": answer == "", "latency_s": round(time.time() - t_call, 3),
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens")}
        with lock:
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[0] += 1
            if n[0] % 100 == 0:
                pf.flush()
                print(f"reader {n[0]}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, tasks))
    pf.close()

    preds = [json.loads(l) for l in open(ppath)]

    def auc(pairs):
        pairs = sorted(pairs, key=lambda x: -x[0])
        pos = sum(l for _, l in pairs)
        neg = len(pairs) - pos
        rs = sum(i + 1 for i, (_, l) in enumerate(pairs) if l)
        return 1.0 - (rs - pos * (pos + 1) / 2) / (pos * neg)

    def arm(arm_name, split):
        rs = [r for r in preds if r["arm"] == arm_name
              and (split == "all" or r["split"] == split)]
        ok = [r for r in rs if r["em"] is not None]
        m = {"n": len(rs)}
        if rs and rs[0]["variant"] == "ans":
            m["em"] = sum(r["em"] for r in ok) / len(ok) if ok else None
            m["f1"] = sum(r["f1"] for r in ok) / len(ok) if ok else None
        else:
            m["hallucination_rate"] = sum(1 for r in ok if not r["empty"]) / len(rs) if rs else None
        return m

    metrics = {"tau": tau, "pipeline": "dense top-50 -> 4B-RRF fusion (0.5/1) top-5 -> v3 gate",
               "calibration_curve": [c for c in curve if c["tau"] * 100 % 5 == 0]}
    for split in ("all", "calibration", "dev", "test"):
        sub = [q for q in qids if split == "all" or splits[q] == split]
        pairs = [(p[(q, "ans")], 1) for q in sub] + [(p[(q, "unans")], 0) for q in sub]
        acc_ans = [q for q in sub if p[(q, "ans")] >= tau]
        acc_unans = [q for q in sub if p[(q, "unans")] >= tau]
        answered = len(acc_ans) + len(acc_unans)
        metrics[split] = {
            "n": len(sub), "auc": round(auc(pairs), 4),
            "coverage": len(acc_ans) / len(sub),
            "answered_precision": len(acc_ans) / answered if answered else None,
            "refusal_recall": 1 - len(acc_unans) / len(sub),
            "nogate_unans": arm("nogate_unans", split),
            "gate_unans": arm("gate_unans", split),
            "nogate_ans": arm("nogate_ans", split),
            "gate_ans": arm("gate_ans", split),
        }
    with open(OUT / "musique_fusiongate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
