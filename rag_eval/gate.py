"""Answerability-gated RAG pipeline for xiaojev v3 (reusable module).

Pipeline: BM25 top-D (corpus-level) -> v3 relevance-channel rerank -> top-k
-> v3 answerability-channel gate on the selected k passages. If P(answerable)
< tau, one retry round widens the BM25 pool (depth 100), re-ranks with freshly
scored candidates, re-selects top-k and re-gates; still below tau -> refuse.

Unanswerable variants simulate missing evidence by excluding the question's
gold docids from every candidate pool stage.

Used by run_gate.py; scoring itself goes through score_v3.score_items.

Configuration (environment variables):
  XIAOJEV_GATE_DIR  directory for score caches and outputs
                    (default: this rag_eval/ directory)
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.common import PROP_ANSWERABLE, PROP_RELEVANCE, load_corpus, load_questions, passage_block  # noqa: E402
from rag_eval.bm25_rank import build_index, score_query  # noqa: E402

OUT = Path(os.environ.get("XIAOJEV_GATE_DIR", Path(__file__).resolve().parent))
NONTRAIN = ("dev", "calibration", "test")
TOPK = 5
POOL_ROUND1 = 50
POOL_ROUND2 = 100


class GatePipeline:
    def __init__(self, pool_depth=POOL_ROUND2):
        self.questions = {q["id"]: q for q in load_questions() if q["split"] in NONTRAIN}
        self.corpus = load_corpus()
        self.by_docid = {c["docid"]: c for c in self.corpus}
        postings, idf, doc_len, avgdl = build_index(self.corpus)
        self.index = (postings, idf, doc_len, avgdl)
        # per-question gold docids + BM25 ranked pool to `pool_depth`
        from rag_eval.common import tokens
        import re
        key2doc = {(c["title"], re.sub(r"\s+", " ", c["text"]).strip()): c["docid"]
                   for c in self.corpus}
        self.gold_docids = {}
        self.bm25_ranked = {}
        for qid, q in self.questions.items():
            self.gold_docids[qid] = {
                key2doc[(t, re.sub(r"\s+", " ", x).strip())]
                for idx, t, x in q["paragraphs"] if idx in set(q["gold_idx"])}
            sc = score_query(tokens(q["question"]), *self.index)
            top = sorted(sc, key=lambda d: -sc[d])[:pool_depth]
            self.bm25_ranked[qid] = [self.corpus[d]["docid"] for d in top]
        # v3 relevance scores for BM25 top-50 (from the earlier cascade run)
        self.rel = defaultdict(dict)
        for line in open(OUT / "v3_scores.jsonl"):
            d = json.loads(line)
            if d["kind"] == "cascade":
                self.rel[d["qid"]][d["docid"]] = d["p_yes"]

    def pool(self, qid, variant, depth):
        cand = self.bm25_ranked[qid][:depth]
        if variant == "unans":
            cand = [d for d in cand if d not in self.gold_docids[qid]]
        return cand

    def rerank(self, qid, variant, depth=POOL_ROUND1, extra_scores=None):
        """v3 relevance rerank of the BM25 pool; returns ranked docids (best first)."""
        cand = self.pool(qid, variant, depth)
        sc = dict(self.rel.get(qid, {}))
        if extra_scores:
            sc.update(extra_scores)
        return sorted(cand, key=lambda d: -sc.get(d, 0.0))

    def topk_passages(self, qid, variant, k=TOPK, depth=POOL_ROUND1, extra_scores=None):
        ranked = self.rerank(qid, variant, depth, extra_scores)
        return [(d, self.by_docid[d]["text"]) for d in ranked[:k] if d in self.by_docid]

    def gate_item(self, qid, variant, round_, passages):
        """Answerability noul item over the selected passages (training format)."""
        q = self.questions[qid]
        state = (f"Question: {q['question']}\n\n"
                 + "\n".join(passage_block(self.by_docid[d]["title"], self.by_docid[d]["text"])
                             for d, _ in passages))
        row = {"id": f"gate_{qid}_{variant}_r{round_}", "primitive": "noul",
               "state": state, "proposition": PROP_ANSWERABLE, "candidates": ["yes", "no"]}
        meta = {"kind": "gate", "qid": qid, "variant": variant, "round": round_}
        return row, meta

    def retry_rel_items(self, qids_variants):
        """Relevance items for BM25 ranks 50..100 (new candidates only)."""
        items = []
        for qid, variant in qids_variants:
            q = self.questions[qid]
            known = set(self.pool(qid, variant, POOL_ROUND1))
            for d in self.pool(qid, variant, POOL_ROUND2):
                if d in known:
                    continue
                c = self.by_docid[d]
                row = {"id": f"rrel_{qid}_{variant}_{d}", "primitive": "noul",
                       "state": f"Question: {q['question']}\n\n{passage_block(c['title'], c['text'])}",
                       "proposition": PROP_RELEVANCE, "candidates": ["yes", "no"]}
                items.append((row, {"kind": "retry_rel", "qid": qid, "variant": variant,
                                    "docid": d}))
        return items


def calibrate_tau(records, target_precision=0.90):
    """records: [{qid, variant, p1, p2}] on the calibration split.
    Pipeline accepts iff p1 >= tau or p2 >= tau. Choose the tau with the
    highest answerable coverage among those meeting the answered-set
    precision target. Returns (tau, curve)."""
    curve = []
    best = None
    for tau100 in range(1, 100):
        tau = tau100 / 100
        acc = [r for r in records if r["p1"] >= tau or r["p2"] >= tau]
        if not acc:
            curve.append({"tau": tau, "coverage": 0.0, "precision": None})
            continue
        prec = sum(1 for r in acc if r["variant"] == "ans") / len(acc)
        cov = sum(1 for r in acc if r["variant"] == "ans") / max(1, sum(1 for r in records if r["variant"] == "ans"))
        curve.append({"tau": tau, "coverage": cov, "precision": prec, "n_answered": len(acc)})
        if prec >= target_precision and (best is None or cov > best[1]):
            best = (tau, cov, prec)
    if best is None:  # target unreachable: take max precision, then max coverage
        feas = [c for c in curve if c["precision"] is not None]
        best_c = max(feas, key=lambda c: (c["precision"], c["coverage"]))
        best = (best_c["tau"], best_c["coverage"], best_c["precision"])
    return best[0], curve
