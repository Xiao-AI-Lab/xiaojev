"""Cross-dataset transfer: hotpotqa / 2wikimultihopqa.

Frozen musique configuration, zero tuning on the new datasets:
  dense (NV-Embed-v2) top-50 -> 4B LoRA fusion rerank (w=0.5, c=1, frozen on
  musique calibration) -> v3 answerability gate (tau recalibrated ONLY on the
  target dataset's own calibration subset, target answered-precision 0.90;
  single round, no retry arm).

Eval questions: semantic-hash non-train split of each dataset (~300, includes
its calibration/dev/test subsets) — identical protocol to musique's 293.
Contamination guards asserted: (a) question ids disjoint from musique's qid
space; (b) no question from the dataset's semantic TRAIN split (v3/v4/4B saw
semantic_v1 items derived from those); (c) corpus built only from eval
questions' own paragraphs.

Arms per dataset:
  A retrieval: dense R@k vs frozen-fusion R@k (paired bootstrap)
  B QA: dense top-4 vs fusion top-4 -> 27B reader (EM/F1, paired bootstrap)
  C gate: hallucination rate without/with gate on synthesized unanswerable
     variants (gold docs excluded at pool stage), gate AUC, coverage at tau

Artifacts in rag_eval/transfer/{ds}_*.jsonl/json; resume at every artifact.
Run: CUDA_VISIBLE_DEVICES=3 python transfer.py
"""
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "data"), str(ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os  # noqa: E402

from make_semanticdata import split_of as sem_split_of  # noqa: E402
from rag_eval.common import PROP_ANSWERABLE, PROP_RELEVANCE, passage_block, snippet  # noqa: E402
from rag_eval.metrics import agg, rank_metrics  # noqa: E402
from rag_eval.dense_retrieval import QUERY_PREFIX, embed  # noqa: E402
from rag_eval.fusion import fuse_rankings  # noqa: E402
from rag_eval.gate import calibrate_tau  # noqa: E402
from rag_eval.run_qa import call_reader, exact_match, token_f1  # noqa: E402

OUT = Path(os.environ.get(
    "XIAOJEV_TRANSFER_DIR", Path(__file__).resolve().parent / "transfer"
))
DATA = Path(os.environ["XIAOJEV_RAG_DATA"]) if os.environ.get("XIAOJEV_RAG_DATA") else None
CKPT_4B = os.environ.get(
    "XIAOJEV_CKPT_4B", str(ROOT / "ckpt" / "qwen3_4b_lora_v1" / "step2500")
)
CKPT_V3 = os.environ.get("XIAOJEV_CKPT_V3", str(ROOT / "ckpt" / "v3"))
FROZEN = dict(reranker_weight=0.5, rank_constant=1)  # musique fusion4b_config.json
TARGET_PRECISION = 0.90
DATASETS = ("hotpotqa", "2wikimultihopqa")


def rag_data_root():
    if DATA is None:
        raise SystemExit(
            "Set XIAOJEV_RAG_DATA to the RAG datasets root (hotpotqa/, "
            "2wikimultihopqa/, musique/ with raw/ + gold.jsonl + questions.jsonl)."
        )
    return DATA


def load_ds(name):
    root = rag_data_root()
    raw = {r.get("_id") or r["id"]: r for r in json.load(open(root / name / "raw" / f"{name}.json"))}
    gold = {g["id"]: g for g in map(json.loads, open(root / name / "gold.jsonl"))}
    questions = []
    for qid, r in raw.items():
        split = sem_split_of(name, qid)
        if split == "train":
            continue  # contaminated by semantic_v1 training
        paras = [(i, t, " ".join(sents)) for i, (t, sents) in enumerate(r["context"])]
        gold_titles = {t for t, _ in gold[qid]["supporting_facts"]}
        gold_idx = sorted(i for i, t, _ in paras if t in gold_titles)
        questions.append({"id": qid, "question": r["question"], "paragraphs": paras,
                          "gold_idx": gold_idx, "answer": gold[qid].get("answer") or "",
                          "answer_aliases": gold[qid].get("answer_aliases") or [],
                          "split": split})
    return questions


def phase_corpus(name, questions):
    """Union of eval questions' paragraphs as the retrieval corpus."""
    path = OUT / f"{name}_corpus.jsonl"
    if path.exists():
        docs = [json.loads(l) for l in open(path)]
    else:
        docs = []
        n = 0
        for q in questions:
            for pidx, title, text in q["paragraphs"]:
                docs.append({"docid": f"{name}:{n}", "qid": q["id"], "pidx": pidx,
                             "title": title, "text": text})
                n += 1
        with open(path, "w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    by_docid = {d["docid"]: d for d in docs}
    gold_docids = {}
    for q in questions:
        gold_docids[q["id"]] = {f"{name}:{n}" for n, d in enumerate(docs)
                                if d["qid"] == q["id"] and d["pidx"] in set(q["gold_idx"])}
    return docs, by_docid, gold_docids


def phase_dense(name, questions, docs):
    """NV-Embed-v2 encode corpus + queries; top-50 per question. Cached."""
    path = OUT / f"{name}_dense.jsonl"
    if path.exists():
        return {r["id"]: r for r in map(json.loads, open(path))}
    t0 = time.time()
    doc_vecs = embed([f"{d['title']}\n{d['text']}" for d in docs])
    doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    qlist = sorted(q["id"] for q in questions)
    qmap = {q["id"]: q for q in questions}
    qv = embed([QUERY_PREFIX + qmap[q]["question"] for q in qlist])
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    out = {}
    with open(path, "w") as f:
        for i, qid in enumerate(qlist):
            sims = qv[i] @ doc_vecs.T
            top = np.argpartition(-sims, 50)[:50]
            top = top[np.argsort(-sims[top])]
            rec = {"id": qid, "split": qmap[qid]["split"],
                   "top50": [docs[j]["docid"] for j in top],
                   "sims": [round(float(sims[j]), 6) for j in top]}
            out[qid] = rec
            f.write(json.dumps(rec) + "\n")
    print(f"  [{name}] dense encoded {len(docs)} docs + {len(qlist)} queries, "
          f"top-50 in {time.time()-t0:.0f}s", flush=True)
    return out


def phase_4b(name, questions, dense, by_docid):
    """4B LoRA relevance scoring of dense top-50. Resume by record count."""
    path = OUT / f"{name}_4b_scores.jsonl"
    need = {}
    done = defaultdict(set)
    if path.exists():
        for r in map(json.loads, open(path)):
            done[r["qid"]].add(r["docid"])
    for q in questions:
        missing = [d for d in dense[q["id"]]["top50"] if d not in done.get(q["id"], set())]
        if missing:
            need[q["id"]] = missing
    if not need:
        print(f"  [{name}] 4B scores complete", flush=True)
    else:
        print(f"  [{name}] 4B scoring: {len(need)} questions remaining", flush=True)
        from lora4b_model import load, predict  # training/lora4b_model.py (needs peft)
        model, tok = load(CKPT_4B)
        t0 = time.time()
        with open(path, "a") as f:
            for n, q in enumerate(questions):
                qid = q["id"]
                if qid not in need:
                    continue
                docs = need[qid]
                rows = [{"id": d, "primitive": "noul", "candidates": ["yes", "no"],
                         "state": f"Question: {q['question']}\n\n"
                                  f"{passage_block(by_docid[d]['title'], by_docid[d]['text'])}",
                         "proposition": PROP_RELEVANCE} for d in docs]
                probs = predict(model, tok, rows)
                for d in docs:
                    raw = probs[d].tolist()
                    f.write(json.dumps({"qid": qid, "docid": d,
                                        "probs": [round(v, 6) for v in raw]}) + "\n")
                if (n + 1) % 25 == 0:
                    f.flush()
                    print(f"  [{name}] 4b {n+1}/{len(questions)} ({time.time()-t0:.0f}s)",
                          flush=True)
        del model
        import torch
        torch.cuda.empty_cache()
    scores = defaultdict(dict)
    for r in map(json.loads, open(path)):
        scores[r["qid"]][r["docid"]] = r["probs"][0]
    return scores


def phase_retrieval(name, questions, dense, scores, gold_docids):
    """Frozen fusion vs dense; per split; paired bootstrap on delta R@5."""
    import random
    fused_rankings = {}
    records = []
    for q in questions:
        qid = q["id"]
        docs = dense[qid]["top50"]
        fused = fuse_rankings(docs, scores[qid], **FROZEN)
        fused_rankings[qid] = fused
        gold = gold_docids[qid]
        records.append({"id": qid, "split": q["split"],
                        "dense": rank_metrics(docs, gold, (1, 4, 5, 10, 20, 50)),
                        "fusion": rank_metrics(fused, gold, (1, 4, 5, 10, 20, 50))})
    with open(OUT / f"{name}_fusion_rankings.json", "w") as f:
        json.dump(fused_rankings, f)
    report = {}
    for split in ("all", "calibration", "dev", "test"):
        sub = [r for r in records if split == "all" or r["split"] == split]
        diffs = [r["fusion"]["r@5"] - r["dense"]["r@5"] for r in sub]
        rng = random.Random(20260926)
        boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(5000))
        report[split] = {"n": len(sub),
                         "dense": agg([r["dense"] for r in sub]),
                         "fusion": agg([r["fusion"] for r in sub]),
                         "delta_r5": sum(diffs) / len(diffs),
                         "paired_bootstrap_delta_r5_95pct": [boot[125], boot[4874]],
                         "improved": sum(d > 0 for d in diffs),
                         "worsened": sum(d < 0 for d in diffs)}
    return report, fused_rankings


def phase_gate_scores(name, questions, fused_rankings, by_docid, gold_docids):
    """v3 answerability on fusion top-5, ans + unans (gold excluded). Single round."""
    path = OUT / f"{name}_gate_scores.jsonl"
    want = {(q["id"], v) for q in questions for v in ("ans", "unans")}
    have = set()
    if path.exists():
        for r in map(json.loads, open(path)):
            have.add((r["qid"], r["variant"]))
    todo = sorted(want - have)
    if not todo:
        print(f"  [{name}] gate scores complete", flush=True)
    else:
        from train import MODEL_PATH, StudentModel, load_ckpt
        from rag_eval.score_v3 import score_items
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = StudentModel().to("cuda")
        load_ckpt(model, CKPT_V3)
        model.eval()
        items = []
        for qid, variant in todo:
            q = next(q for q in questions if q["id"] == qid)
            docs = fused_rankings[qid]
            if variant == "unans":
                docs = [d for d in docs if d not in gold_docids[qid]]
            top5 = docs[:5]
            state = (f"Question: {q['question']}\n\n"
                     + "\n".join(passage_block(by_docid[d]["title"], by_docid[d]["text"])
                                 for d in top5))
            row = {"id": f"transfer_gate_{qid}_{variant}", "primitive": "noul",
                   "state": state, "proposition": PROP_ANSWERABLE,
                   "candidates": ["yes", "no"]}
            items.append((row, {"kind": "gate", "qid": qid, "variant": variant, "round": 1}))
        t0 = time.time()
        with open(path, "a") as f:
            score_items(model, tok, items, "cuda", f)
        print(f"  [{name}] gate scoring: {len(items)} items, {time.time()-t0:.0f}s", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()
    p = {}
    for r in map(json.loads, open(path)):
        p[(r["qid"], r["variant"])] = r.get("p_yes", r.get("probs", [0.0])[0])
    return p


def phase_reader(name, questions, dense, fused_rankings, by_docid, gold_docids, p_gate, tau):
    path = OUT / f"{name}_predictions.jsonl"
    done = set()
    if path.exists():
        for r in map(json.loads, open(path)):
            done.add((r["id"], r["arm"]))
    qmap = {q["id"]: q for q in questions}

    def passages(order, qid, k):
        return [(d, by_docid[d]["text"]) for d in order[:k]]

    tasks = []
    for q in questions:
        qid = q["id"]
        unans_order = [d for d in fused_rankings[qid] if d not in gold_docids[qid]]
        plan = [
            ("qa_dense", passages(dense[qid]["top50"], qid, 4), "ans"),
            ("qa_fusion", passages(fused_rankings[qid], qid, 4), "ans"),
            ("gate_nogate_ans", passages(fused_rankings[qid], qid, 5), "ans"),
            ("gate_nogate_unans", passages(unans_order, qid, 5), "unans"),
        ]
        if p_gate[(qid, "ans")] >= tau:
            plan.append(("gate_gate_ans", passages(fused_rankings[qid], qid, 5), "ans"))
        if p_gate[(qid, "unans")] >= tau:
            plan.append(("gate_gate_unans", passages(unans_order, qid, 5), "unans"))
        for arm, ps, variant in plan:
            if (qid, arm) not in done:
                tasks.append((qid, arm, variant, ps))
    print(f"  [{name}] reader calls remaining: {len(tasks)}", flush=True)
    lock = threading.Lock()
    preds_f = open(path, "a")
    done_n = [0]
    t0 = time.time()

    def work(task):
        qid, arm, variant, ps = task
        q = qmap[qid]
        t_call = time.time()
        answer, usage = call_reader(q["question"], ps)
        dt = time.time() - t_call
        aliases = [q["answer"]] + q["answer_aliases"]
        rec = {"id": qid, "arm": arm, "variant": variant, "split": q["split"],
               "prediction": answer, "doc_ids": [p for p, _ in ps],
               "em": None if answer is None else exact_match(answer or "", aliases),
               "f1": None if answer is None else token_f1(answer or "", aliases),
               "empty": answer == "", "latency_s": round(dt, 3),
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens")}
        with lock:
            preds_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done_n[0] += 1
            if done_n[0] % 100 == 0:
                preds_f.flush()
                print(f"  [{name}] reader {done_n[0]}/{len(tasks)} ({time.time()-t0:.0f}s)",
                      flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, tasks))
    preds_f.close()
    return [json.loads(l) for l in open(path)]


def summarize(name, questions, retrieval, p_gate, tau, preds):
    import random
    splits = {q["id"]: q["split"] for q in questions}
    qids = sorted(splits)
    out = {"tau": tau, "retrieval": retrieval}

    def auc(pairs):
        pairs = sorted(pairs, key=lambda x: -x[0])
        pos = sum(l for _, l in pairs)
        neg = len(pairs) - pos
        rs = sum(i + 1 for i, (_, l) in enumerate(pairs) if l)
        return 1.0 - (rs - pos * (pos + 1) / 2) / (pos * neg)

    gate = {}
    for split in ("all", "calibration", "dev", "test"):
        sub = [q for q in qids if split == "all" or splits[q] == split]
        pairs = [(p_gate[(q, "ans")], 1) for q in sub] + [(p_gate[(q, "unans")], 0) for q in sub]
        acc_ans = [q for q in sub if p_gate[(q, "ans")] >= tau]
        acc_unans = [q for q in sub if p_gate[(q, "unans")] >= tau]
        answered = len(acc_ans) + len(acc_unans)
        gate[split] = {"n": len(sub), "auc": round(auc(pairs), 4),
                       "coverage": len(acc_ans) / len(sub),
                       "answered_precision": len(acc_ans) / answered if answered else None,
                       "refusal_recall": 1 - len(acc_unans) / len(sub)}
    out["gate"] = gate

    def arm_metrics(arm, split, variant):
        rs = [r for r in preds
              if r["arm"] == arm and (split == "all" or r["split"] == split)
              and r["variant"] == variant]
        ok = [r for r in rs if r["em"] is not None]
        m = {"n": len(rs), "failures": len(rs) - len(ok)}
        if variant == "ans":
            m["em"] = sum(r["em"] for r in ok) / len(ok) if ok else None
            m["f1"] = sum(r["f1"] for r in ok) / len(ok) if ok else None
        else:
            m["hallucination_rate"] = sum(1 for r in ok if not r["empty"]) / len(rs) if rs else None
        m["prompt_tokens"] = sum(r["prompt_tokens"] or 0 for r in rs)
        m["completion_tokens"] = sum(r["completion_tokens"] or 0 for r in rs)
        m["mean_latency_s"] = sum(r["latency_s"] for r in ok) / len(ok) if ok else None
        return m

    qa = {}
    for split in ("all", "calibration", "dev", "test"):
        qa[split] = {"dense_top4": arm_metrics("qa_dense", split, "ans"),
                     "fusion_top4": arm_metrics("qa_fusion", split, "ans")}
    # paired bootstrap on EM difference (all)
    rs_d = {r["id"]: r for r in preds if r["arm"] == "qa_dense"}
    rs_f = {r["id"]: r for r in preds if r["arm"] == "qa_fusion"}
    common = sorted(set(rs_d) & set(rs_f))
    diffs = [rs_f[q]["em"] - rs_d[q]["em"] for q in common
             if rs_d[q]["em"] is not None and rs_f[q]["em"] is not None]
    rng = random.Random(20260926)
    boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(5000))
    out["qa"] = qa
    out["qa_delta_em"] = {"mean": sum(diffs) / len(diffs), "n": len(diffs),
                          "paired_bootstrap_95pct": [boot[125], boot[4874]]}
    out["gate_reader"] = {
        s: {"nogate_ans": arm_metrics("gate_nogate_ans", s, "ans"),
            "gate_ans": arm_metrics("gate_gate_ans", s, "ans"),
            "nogate_unans": arm_metrics("gate_nogate_unans", s, "unans"),
            "gate_unans": arm_metrics("gate_gate_unans", s, "unans")}
        for s in ("all", "test")}
    return out


def run_dataset(name):
    print(f"=== {name} ===", flush=True)
    questions = load_ds(name)
    musique_qids = {json.loads(l)["id"]
                    for l in open(rag_data_root() / "musique" / "questions.jsonl")}
    assert not {q["id"] for q in questions} & musique_qids, "qid space overlap with musique"
    assert all(sem_split_of(name, q["id"]) != "train" for q in questions)
    print(f"  eval questions: {len(questions)} "
          f"(cal {sum(q['split']=='calibration' for q in questions)}, "
          f"dev {sum(q['split']=='dev' for q in questions)}, "
          f"test {sum(q['split']=='test' for q in questions)}); "
          f"qid disjoint from musique asserted", flush=True)
    docs, by_docid, gold_docids = phase_corpus(name, questions)
    print(f"  corpus: {len(docs)} paragraphs", flush=True)
    dense = phase_dense(name, questions, docs)
    scores = phase_4b(name, questions, dense, by_docid)
    retrieval, fused_rankings = phase_retrieval(name, questions, dense, scores, gold_docids)
    print(f"  retrieval: all dense R@5={retrieval['all']['dense']['r@5']:.4f} "
          f"fusion R@5={retrieval['all']['fusion']['r@5']:.4f}", flush=True)
    p_gate = phase_gate_scores(name, questions, fused_rankings, by_docid, gold_docids)
    cal_records = [{"qid": q["id"], "variant": v,
                    "p1": p_gate[(q["id"], v)], "p2": p_gate[(q["id"], v)]}
                   for q in questions if q["split"] == "calibration"
                   for v in ("ans", "unans")]
    tau, curve = calibrate_tau(cal_records, TARGET_PRECISION)
    print(f"  calibrated tau={tau:.2f}", flush=True)
    preds = phase_reader(name, questions, dense, fused_rankings, by_docid, gold_docids,
                         p_gate, tau)
    metrics = summarize(name, questions, retrieval, p_gate, tau, preds)
    metrics["frozen_fusion"] = FROZEN
    metrics["target_precision"] = TARGET_PRECISION
    metrics["calibration_curve"] = [c for c in curve if c["tau"] * 100 % 5 == 0]
    metrics["n_questions"] = len(questions)
    metrics["split_counts"] = dict(Counter(q["split"] for q in questions))
    with open(OUT / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [{name}] done", flush=True)
    return metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for name in DATASETS:
        all_metrics[name] = run_dataset(name)
    with open(OUT / "transfer_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
