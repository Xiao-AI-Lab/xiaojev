"""Experiment 2: learned score-level fusion (logistic regression) vs manual RRF.

Features per (question, candidate): (dense cosine sim min-max normalized within
the question, xiaojev P(relevant)). Two variants: v4 scores and 4B LoRA scores.
Fit on the 98 calibration questions ONLY (4900 samples), frozen, then evaluated
on dev/test/nontrain. Stability: 5-fold CV within the calibration questions.
No sklearn -> 3-parameter IRLS implemented with numpy.

Outputs: fusion_logistic_metrics.json (+ dense_sims_top50.jsonl cache).

Configuration (environment variables):
  XIAOJEV_RAG_EVAL_DIR      working/output directory (default: this rag_eval/)
  XIAOJEV_V4_DENSE_SCORES   v4 dense top-50 scores JSONL (required)
  XIAOJEV_4B_DENSE_SCORES   4B LoRA dense top-50 scores JSONL (required)
  XIAOJEV_DENSE_INDEX_ROOT  dense index root for cosine features
                            (see rag_eval/dense_retrieval.py)
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.common import load_corpus, load_questions  # noqa: E402
from rag_eval.metrics import agg, rank_metrics  # noqa: E402
from rag_eval.dense_retrieval import HIPPO, QUERY_PREFIX, embed  # noqa: E402
from rag_eval.fusion import fuse_rankings  # noqa: E402

OUT = Path(os.environ.get("XIAOJEV_RAG_EVAL_DIR", Path(__file__).resolve().parent))
SIMS_CACHE = OUT / "dense_sims_top50.jsonl"
SCORE_FILES = {
    "v4": os.environ.get("XIAOJEV_V4_DENSE_SCORES"),
    "4b": os.environ.get("XIAOJEV_4B_DENSE_SCORES"),
}
RRF_FROZEN = {
    "v4": dict(reranker_weight=0.4, rank_constant=1),   # v4_repair fusion_config.json
    "4b": json.loads((OUT / "fusion4b_config.json").read_text())["config"],
}


def dense_sims(dense, questions):
    """docid -> cosine sim for each question's top-50; cached."""
    if SIMS_CACHE.exists():
        return {r["id"]: dict(zip(r["top50"], r["sims"]))
                for r in map(json.loads, SIMS_CACHE.open())}
    corpus = load_corpus()
    key2rowdoc = {}
    inputs = json.load(open(HIPPO / "index" / "inputs.json"))["chunk"]
    key2doc = {(c["title"], re.sub(r"\s+", " ", c["text"]).strip()): c["docid"] for c in corpus}
    row_of_doc = {}
    for i, c in enumerate(inputs):
        title, _, text = c["content"].partition("\n")
        row_of_doc[key2doc[(title, re.sub(r"\s+", " ", text).strip())]] = i
    vecs = np.array(np.load(HIPPO / "index" / "chunk_vectors.npy", mmap_mode="r"), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    qlist = sorted(dense)
    qv = embed([QUERY_PREFIX + questions[q]["question"] for q in qlist])
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    out = {}
    with open(SIMS_CACHE, "w") as f:
        for i, q in enumerate(qlist):
            docids = dense[q]["top50"]
            sims = [float(qv[i] @ vecs[row_of_doc[d]]) for d in docids]
            out[q] = dict(zip(docids, sims))
            f.write(json.dumps({"id": q, "top50": docids,
                                "sims": [round(s, 6) for s in sims]}) + "\n")
    print(f"dense sims computed for {len(qlist)} questions -> {SIMS_CACHE}", flush=True)
    return out


def fit_lr(X, y, l2=1e-4, iters=100):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (p - y) + l2 * w
        W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def build_dataset(qids, dense, sims, scores):
    """X: [bias, dense_sim_minmax, p_relevant]; y: is_gold; groups: qid order."""
    X, y, groups = [], [], []
    for q in qids:
        docids = dense[q]["top50"]
        s = np.array([sims[q][d] for d in docids])
        lo, hi = s.min(), s.max()
        s_norm = (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)
        gold = set(dense[q]["gold_docids"])
        for d, sn in zip(docids, s_norm):
            X.append([1.0, sn, scores[q][d]])
            y.append(1.0 if d in gold else 0.0)
            groups.append(q)
    return np.array(X), np.array(y), groups


def lr_rankings(qids, dense, sims, scores, w):
    out = {}
    for q in qids:
        docids = dense[q]["top50"]
        s = np.array([sims[q][d] for d in docids])
        lo, hi = s.min(), s.max()
        s_norm = (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)
        X = np.column_stack([np.ones(len(docids)), s_norm,
                             [scores[q][d] for d in docids]])
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        out[q] = [d for d, _ in sorted(zip(docids, p), key=lambda t: -t[1])]
    return out


def eval_rankings(rankings, qids, dense):
    rows = [rank_metrics(rankings[q], set(dense[q]["gold_docids"]), (1, 4, 5, 10, 20, 50))
            for q in qids]
    return agg(rows)


def main():
    missing = [k for k, v in SCORE_FILES.items() if not v]
    if missing:
        raise SystemExit(
            "Missing score files for: "
            + ", ".join(missing)
            + ". Set XIAOJEV_V4_DENSE_SCORES and XIAOJEV_4B_DENSE_SCORES "
              "(artifacts of the v4_acceptance / lora4b runs; not committed to git)."
        )
    questions = {q["id"]: q for q in load_questions()}
    dense = {r["id"]: r for r in map(json.loads, (OUT / "dense_corpus.jsonl").open())}
    sims = dense_sims(dense, questions)
    calibration = sorted(q for q in dense if questions[q]["split"] == "calibration")
    assert len(calibration) == 98
    splits = {s: sorted(q for q in dense if s == "nontrain" or questions[q]["split"] == s)
              for s in ("calibration", "dev", "test", "nontrain")}

    report = {"features": ["bias", "dense_sim_minmax_per_q", "xiaojev_p_relevant"],
              "fit_split": "calibration (98 questions, 4900 samples)", "models": {}}
    for tag, score_file in SCORE_FILES.items():
        scores = defaultdict(dict)
        for r in map(json.loads, Path(score_file).open()):
            scores[r["qid"]][r["docid"]] = r["probs"][0]
        X, y, _ = build_dataset(calibration, dense, sims, scores)
        w = fit_lr(X, y)
        # 5-fold CV within calibration questions
        rng = np.random.RandomState(20260926)
        folds = rng.permutation(calibration)
        fold_r5, fold_coefs = [], []
        for k in range(5):
            val = sorted(folds[k::5])
            tr = [q for q in calibration if q not in set(val)]
            Xk, yk, _ = build_dataset(tr, dense, sims, scores)
            wk = fit_lr(Xk, yk)
            fold_coefs.append(wk.tolist())
            fold_r5.append(eval_rankings(lr_rankings(val, dense, sims, scores, wk),
                                         val, dense)["r@5"])
        model = {"coefficients": dict(zip(report["features"], [round(float(c), 4) for c in w])),
                 "cv5_calibration_r5": {"mean": float(np.mean(fold_r5)),
                                        "std": float(np.std(fold_r5)),
                                        "folds": [round(v, 4) for v in fold_r5],
                                        "coef_spread": [[round(float(c), 3) for c in fc]
                                                        for fc in fold_coefs]},
                 "splits": {}}
        lr_all = {s: eval_rankings(lr_rankings(ids, dense, sims, scores, w), ids, dense)
                  for s, ids in splits.items()}
        for s, ids in splits.items():
            rrf_rows = [rank_metrics(fuse_rankings(dense[q]["top50"], scores[q],
                                                   **RRF_FROZEN[tag]),
                                     set(dense[q]["gold_docids"]), (1, 4, 5, 10, 20, 50))
                        for q in ids]
            model["splits"][s] = {"rrf": agg(rrf_rows), "logistic": lr_all[s],
                                  "delta_lr_minus_rrf_r5": lr_all[s]["r@5"] - agg(rrf_rows)["r@5"]}
        report["models"][tag] = model
        print(f"[{tag}] coef={model['coefficients']} cv5 r5={np.mean(fold_r5):.4f}+-{np.std(fold_r5):.4f}",
              flush=True)
        for s in ("test", "nontrain"):
            m = model["splits"][s]
            print(f"  {s}: RRF r@5={m['rrf']['r@5']:.4f}  LR r@5={m['logistic']['r@5']:.4f}  "
                  f"delta={m['delta_lr_minus_rrf_r5']:+.4f}", flush=True)

    with open(OUT / "fusion_logistic_metrics.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
