"""Dense-first-stage variant of the gated pipeline.

Identical to gate.py's GatePipeline except the candidate pool comes from
NV-Embed-v2 dense retrieval (top-100, cached to dense_corpus100.jsonl) instead
of BM25. Round 1 uses ranks 1-50 (scores reused from v3_scores_dense.jsonl);
retry round widens to ranks 51-100 with fresh v3 relevance scoring.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.common import PROP_ANSWERABLE, load_corpus, load_questions, passage_block  # noqa: E402
from rag_eval.dense_retrieval import HIPPO, QUERY_PREFIX, embed  # noqa: E402
from rag_eval.gate import NONTRAIN, POOL_ROUND1, POOL_ROUND2, OUT  # noqa: E402

DENSE100 = OUT / "dense_corpus100.jsonl"


class DenseGatePipeline:
    def __init__(self, pool_depth=POOL_ROUND2):
        self.questions = {q["id"]: q for q in load_questions() if q["split"] in NONTRAIN}
        self.corpus = load_corpus()
        self.by_docid = {c["docid"]: c for c in self.corpus}
        key2doc = {(c["title"], re.sub(r"\s+", " ", c["text"]).strip()): c["docid"]
                   for c in self.corpus}
        self.gold_docids = {}
        for qid, q in self.questions.items():
            self.gold_docids[qid] = {
                key2doc[(t, re.sub(r"\s+", " ", x).strip())]
                for idx, t, x in q["paragraphs"] if idx in set(q["gold_idx"])}
        self.dense_ranked = self._dense_top100()
        # v3 relevance scores for dense top-50 (cascade run)
        self.rel = defaultdict(dict)
        for line in open(OUT / "v3_scores_dense.jsonl"):
            d = json.loads(line)
            if d["kind"] == "dense_cascade":
                self.rel[d["qid"]][d["docid"]] = d.get("p_yes", d.get("probs", [0.0])[0])

    def _dense_top100(self):
        if DENSE100.exists():
            return {d["id"]: d["top100"] for d in map(json.loads, open(DENSE100))}
        inputs = json.load(open(HIPPO / "index" / "inputs.json"))["chunk"]
        key2doc = {(c["title"], re.sub(r"\s+", " ", c["text"]).strip()): c["docid"]
                   for c in self.corpus}
        row_docid = []
        for c in inputs:
            title, _, text = c["content"].partition("\n")
            row_docid.append(key2doc[(title, re.sub(r"\s+", " ", text).strip())])
        vecs = np.array(np.load(HIPPO / "index" / "chunk_vectors.npy", mmap_mode="r"),
                        dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        qlist = sorted(self.questions)
        qv = embed([QUERY_PREFIX + self.questions[q]["question"] for q in qlist])
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)
        out = {}
        for i, qid in enumerate(qlist):
            sims = qv[i] @ vecs.T
            top = np.argpartition(-sims, 100)[:100]
            top = top[np.argsort(-sims[top])]
            out[qid] = [row_docid[j] for j in top]
        with open(DENSE100, "w") as f:
            for qid in qlist:
                f.write(json.dumps({"id": qid, "top100": out[qid]}) + "\n")
        print(f"dense top-100 computed for {len(qlist)} questions -> {DENSE100}", flush=True)
        return out

    def pool(self, qid, variant, depth):
        cand = self.dense_ranked[qid][:depth]
        if variant == "unans":
            cand = [d for d in cand if d not in self.gold_docids[qid]]
        return cand

    def rerank(self, qid, variant, depth=POOL_ROUND1, extra_scores=None):
        cand = self.pool(qid, variant, depth)
        sc = dict(self.rel.get(qid, {}))
        if extra_scores:
            sc.update(extra_scores)
        return sorted(cand, key=lambda d: -sc.get(d, 0.0))

    def topk_passages(self, qid, variant, k=5, depth=POOL_ROUND1, extra_scores=None):
        ranked = self.rerank(qid, variant, depth, extra_scores)
        return [(d, self.by_docid[d]["text"]) for d in ranked[:k] if d in self.by_docid]

    def gate_item(self, qid, variant, round_, passages):
        q = self.questions[qid]
        state = (f"Question: {q['question']}\n\n"
                 + "\n".join(passage_block(self.by_docid[d]["title"], self.by_docid[d]["text"])
                             for d, _ in passages))
        row = {"id": f"gated_{qid}_{variant}_r{round_}", "primitive": "noul",
               "state": state, "proposition": PROP_ANSWERABLE, "candidates": ["yes", "no"]}
        return row, {"kind": "gate", "qid": qid, "variant": variant, "round": round_}

    def retry_rel_items(self, qids_variants):
        from rag_eval.common import PROP_RELEVANCE
        items = []
        for qid, variant in qids_variants:
            q = self.questions[qid]
            known = set(self.pool(qid, variant, POOL_ROUND1))
            for d in self.pool(qid, variant, POOL_ROUND2):
                if d in known:
                    continue
                c = self.by_docid[d]
                row = {"id": f"drel_{qid}_{variant}_{d}", "primitive": "noul",
                       "state": f"Question: {q['question']}\n\n{passage_block(c['title'], c['text'])}",
                       "proposition": PROP_RELEVANCE, "candidates": ["yes", "no"]}
                items.append((row, {"kind": "retry_rel", "qid": qid, "variant": variant,
                                    "docid": d}))
        return items
