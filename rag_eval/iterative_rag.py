"""Calibration-gated iterative RAG (reusable module).

Each round: dense top-50 -> xiaojev fusion rerank -> top-k accumulated into
the evidence set E -> answerability gate P(answerable | question, E):
  p >= tau            -> the 27B reader answers from E, stop
  round == max_rounds -> per-arm fallback: answer anyway or refuse
  else                -> the 27B model proposes a follow-up subquery (prompt:
                         question + current evidence summary + what is still
                         missing) and the next round retrieves with it

Evaluation arms (ARMS):
  single        one retrieval round, no gate (existing dense+fusion baseline)
  fixed2        two rounds with a subquery in between, no gate
  gated         gate-controlled, up to 3 rounds, answer on exhaustion
  gated_refuse  gate-controlled, up to 3 rounds, refuse on exhaustion

Every model-touching step (retrieval / rerank / gate / reader / subquery) is
an injected callable, so unit tests run with fakes (no GPU, no 8020). The
adapters at the bottom wire the real components (NV-Embed-v2 dense retrieval,
xiaojev noul scoring, weighted-RRF fusion, qwen3.8-27b reader/proposer).
run_iterative.py is the evaluation driver.
"""
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _p in (str(ROOT), str(HERE), str(ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag_eval.common import PROP_ANSWERABLE, PROP_RELEVANCE, passage_block
from rag_eval.fusion import fuse_rankings

import os

OUT = Path(os.environ.get("XIAOJEV_RAG_EVAL_DIR", HERE))
NONTRAIN = ("dev", "calibration", "test")
TOPK = 5
POOL = 50
TARGET_PRECISION = 0.90


@dataclass(frozen=True)
class ArmConfig:
    max_rounds: int
    gated: bool
    on_exhaust: str = "answer"  # "answer" | "refuse"; only used by gated arms
    top_k: int = TOPK
    pool: int = POOL


ARMS = {
    "single": ArmConfig(max_rounds=1, gated=False),
    "fixed2": ArmConfig(max_rounds=2, gated=False),
    "gated": ArmConfig(max_rounds=3, gated=True, on_exhaust="answer"),
    "gated_refuse": ArmConfig(max_rounds=3, gated=True, on_exhaust="refuse"),
}

STOP_UNGATED = "ungated"
STOP_ACCEPT = "gate_accept"
STOP_EXHAUST_ANSWER = "exhaust_answer"
STOP_EXHAUST_REFUSE = "exhaust_refuse"


class IterativeRAG:
    """Runs one question under one arm. Injected callables:

    retrieve(query, k)   -> dense-ordered passages [{docid, title, text}]
    rerank(question, passages) -> xiaojev-fused ordering of the same passages
    gate(question, evidence)   -> P(answerable), required for gated arms
    read(question, evidence)   -> (answer_text, usage), required unless the
                                  arm can only refuse
    propose(question, evidence, round) -> (subquery, usage), required when
                                  max_rounds > 1
    """

    def __init__(self, retrieve, rerank, gate=None, read=None, propose=None,
                 config=ARMS["gated"], tau=None):
        if config.gated:
            if gate is None:
                raise ValueError("gated arms need a gate callable")
            if tau is None:
                raise ValueError("gated arms need tau")
        if config.on_exhaust == "answer" and read is None:
            raise ValueError("this arm can answer: read callable required")
        if not config.gated and read is None:
            raise ValueError("ungated arms always answer: read callable required")
        if config.max_rounds > 1 and propose is None:
            raise ValueError("multi-round arms need a propose callable")
        self.retrieve = retrieve
        self.rerank = rerank
        self.gate = gate
        self.read = read
        self.propose = propose
        self.config = config
        self.tau = tau

    def run(self, question, exclude_docids=()):
        """exclude_docids: docids banned from every candidate stage (the unans
        variant bans the question's gold docids to simulate missing evidence)."""
        cfg = self.config
        seen = set(exclude_docids)
        evidence, rounds = [], []
        answer, reader_usage = None, {}
        proposer_usage = []
        subquery = None
        stop = None
        t0 = time.time()
        for rnd in range(1, cfg.max_rounds + 1):
            query = question if rnd == 1 else subquery
            cand = [p for p in self.retrieve(query, cfg.pool) if p["docid"] not in seen]
            top = self.rerank(question, cand)[: cfg.top_k]
            seen.update(p["docid"] for p in top)
            evidence.extend(top)
            rec = {"round": rnd, "query": query,
                   "top_docids": [p["docid"] for p in top]}
            if cfg.gated:
                p = self.gate(question, evidence)
                rec["p_answerable"] = p
                if p >= self.tau:
                    answer, reader_usage = self.read(question, evidence)
                    stop = STOP_ACCEPT
                    rounds.append(rec)
                    break
            rounds.append(rec)
            if rnd == cfg.max_rounds:
                if cfg.gated and cfg.on_exhaust == "refuse":
                    stop = STOP_EXHAUST_REFUSE
                else:
                    answer, reader_usage = self.read(question, evidence)
                    stop = STOP_EXHAUST_ANSWER if cfg.gated else STOP_UNGATED
            else:
                subquery, usage = self.propose(question, evidence, rnd)
                proposer_usage.append({"round": rnd, "subquery": subquery, **usage})
        return {"stop": stop, "rounds": rounds, "n_rounds": len(rounds),
                "gate_probs": [r.get("p_answerable") for r in rounds],
                "evidence_docids": [p["docid"] for p in evidence],
                "answer": answer, "reader_usage": reader_usage,
                "proposer_usage": proposer_usage,
                "latency_s": round(time.time() - t0, 3)}


def decide(gate_probs, arm, tau):
    """Pure trajectory decision for probe-mode evaluation: gate_probs[r] is
    P(answerable) after round r+1 on the evidence accumulated through that
    round. Returns (n_rounds, stop) matching IterativeRAG.run."""
    if not arm.gated:
        return arm.max_rounds, STOP_UNGATED
    for i, p in enumerate(gate_probs[: arm.max_rounds], 1):
        if p >= tau:
            return i, STOP_ACCEPT
    n = min(arm.max_rounds, len(gate_probs))
    return n, STOP_EXHAUST_ANSWER if arm.on_exhaust == "answer" else STOP_EXHAUST_REFUSE


def calibrate_tau_iter(records, target_precision=TARGET_PRECISION):
    """records: [{qid, variant, probs: [p_round1, ...]}] on the calibration
    split; a question is accepted iff any round's p >= tau. Same selection
    rule as gate.calibrate_tau: among taus meeting the answered-set precision
    target, take the one with the highest answerable coverage.
    Returns (tau, curve)."""
    curve = []
    best = None
    n_ans = max(1, sum(1 for r in records if r["variant"] == "ans"))
    for tau100 in range(1, 100):
        tau = tau100 / 100
        acc = [r for r in records if any(p >= tau for p in r["probs"])]
        if not acc:
            curve.append({"tau": tau, "coverage": 0.0, "precision": None})
            continue
        prec = sum(1 for r in acc if r["variant"] == "ans") / len(acc)
        cov = sum(1 for r in acc if r["variant"] == "ans") / n_ans
        curve.append({"tau": tau, "coverage": cov, "precision": prec,
                      "n_answered": len(acc)})
        if prec >= target_precision and (best is None or cov > best[1]):
            best = (tau, cov, prec)
    if best is None:  # target unreachable: take max precision, then max coverage
        feas = [c for c in curve if c["precision"] is not None]
        best_c = max(feas, key=lambda c: (c["precision"], c["coverage"]))
        best = (best_c["tau"], best_c["coverage"], best_c["precision"])
    return best[0], curve


# ---------------------------------------------------------------------------
# real adapters (GPU / network); imported lazily so the core stays testable
# ---------------------------------------------------------------------------

class DenseRetriever:
    """NV-Embed-v2 dense retrieval over the reused HippoRAGv2 chunk index
    (same vectors and docid mapping as dense_retrieval.py / gate_dense.py)."""

    def __init__(self, corpus):
        import numpy as np

        from rag_eval.dense_retrieval import HIPPO, QUERY_PREFIX, embed, norm_text

        self.np = np
        self.embed = embed
        self.prefix = QUERY_PREFIX
        self.by_docid = {c["docid"]: c for c in corpus}
        key2doc = {(c["title"], norm_text(c["text"])): c["docid"] for c in corpus}
        inputs = json.load(open(HIPPO / "index" / "inputs.json"))["chunk"]
        self.row_docid = []
        for c in inputs:
            title, _, text = c["content"].partition("\n")
            self.row_docid.append(key2doc[(title, norm_text(text))])
        vecs = np.array(np.load(HIPPO / "index" / "chunk_vectors.npy", mmap_mode="r"),
                        dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        self.vecs = vecs

    def retrieve_batch(self, queries, k=POOL, overshoot=16):
        """Top-(k+overshoot) docids per query, dense order. The overshoot
        margin lets callers still take k after banning docids."""
        np = self.np
        qv = self.embed([self.prefix + q for q in queries])
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)
        sims = qv @ self.vecs.T
        out = []
        for row in sims:
            top = np.argpartition(-row, k + overshoot)[: k + overshoot]
            top = top[np.argsort(-row[top])]
            out.append([self.row_docid[j] for j in top])
        return out

    def retrieve(self, query, k=POOL, overshoot=16):
        """Top-k passage dicts in dense order (single-query wrapper)."""
        docids = self.retrieve_batch([query], k, overshoot)[0][:k]
        return [{"docid": d, "title": self.by_docid[d]["title"],
                 "text": self.by_docid[d]["text"]} for d in docids]


class XiaojevScorer:
    """Batched noul yes/no scoring with a loaded xiaojev model (any ckpt that
    train.encode_row/collate/microbatch support: v3 StudentModel or 4B LoRA)."""

    def __init__(self, model, tok, device="cuda", budget=65536):
        self.model = model
        self.tok = tok
        self.device = device
        self.budget = budget

    def score(self, rows):
        """rows: [{state, proposition}] -> list of P(yes), order preserved."""
        import torch
        import torch.nn.functional as F

        from train import collate, encode_row, microbatch

        paths, slices = [], []
        for row in rows:
            full = {"primitive": "noul", "candidates": ["yes", "no"], **row}
            _, _, rp = encode_row(self.tok, full, 8192)
            slices.append((len(paths), len(rp)))
            paths.extend(rp)
        out = [None] * len(rows)
        with torch.no_grad():
            for batch in microbatch(slices, paths, self.budget):
                mb = [paths[i] for r in batch
                      for i in range(slices[r][0], slices[r][0] + slices[r][1])]
                tokens, mask, lengths = collate(mb, self.tok.pad_token_id, self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(tokens, mask, lengths)
                off = 0
                for r in batch:
                    _, k = slices[r]
                    probs = F.softmax(logits[off: off + k].float(), dim=-1)
                    out[r] = round(float(probs[0]), 6)
                    off += k
        return out

    def relevance(self, question, passages):
        rows = [{"state": f"Question: {question}\n\n"
                          + passage_block(p["title"], p["text"]),
                 "proposition": PROP_RELEVANCE} for p in passages]
        return self.score(rows)

    def answerable(self, question, passages):
        state = (f"Question: {question}\n\n"
                 + "\n".join(passage_block(p["title"], p["text"]) for p in passages))
        return self.score([{"state": state, "proposition": PROP_ANSWERABLE}])[0]


class FusionReranker:
    """xiaojev relevance + dense order -> weighted-RRF fused ordering."""

    def __init__(self, scorer, config):
        self.scorer = scorer
        self.config = config  # {reranker_weight, rank_constant}

    def rerank(self, question, passages):
        docids = [p["docid"] for p in passages]
        scores = dict(zip(docids, self.scorer.relevance(question, passages)))
        by_id = {p["docid"]: p for p in passages}
        return [{**by_id[d], "xiaojev_p_relevant": scores[d]}
                for d in fuse_rankings(docids, scores, **self.config)]


class ReaderClient:
    """qwen3.8-27b reader, same prompt/schema as the RAG eval (run_qa.py)."""

    def read(self, question, passages):
        from rag_eval.run_qa import call_reader

        return call_reader(question, [(p["docid"], p["text"]) for p in passages])


SUBQUERY_SCHEMA = {"name": "followup_subquery", "strict": True, "schema": {
    "type": "object", "properties": {"subquery": {"type": "string"}},
    "required": ["subquery"], "additionalProperties": False}}

SUBQUERY_SYSTEM = "You are a retrieval planner for multi-hop question answering."


def build_subquery_prompt(question, evidence):
    blocks = "\n".join(passage_block(p["title"], p["text"]) for p in evidence)
    return (f"Question: {question}\n\n"
            f"Evidence gathered so far:\n{blocks}\n\n"
            "The evidence above is not sufficient to answer the question. "
            "Identify what information is still missing and write one focused "
            "follow-up search query that would retrieve a passage containing "
            "the missing information. Do not answer the question. "
            "Return exactly one JSON object with one field: "
            "{\"subquery\": \"<follow-up query>\"}.")


class SubqueryProposer:
    """Follow-up subquery generator (same endpoint settings as the reader)."""

    def __init__(self, api=None, model=None):
        self.api = api or (
            os.environ.get("XIAOJEV_READER_URL", "http://127.0.0.1:8020/v1").rstrip("/")
            + "/chat/completions"
        )
        self.model = model or os.environ.get("XIAOJEV_READER_MODEL", "qwen3.8-27b")

    def propose(self, question, evidence, round_, retries=3):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SUBQUERY_SYSTEM},
                         {"role": "user", "content": build_subquery_prompt(question, evidence)}],
            "temperature": 0, "max_tokens": 1024, "seed": 20260917,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": SUBQUERY_SCHEMA},
        }
        for attempt in range(retries):
            try:
                req = request.Request(self.api, data=json.dumps(payload).encode(),
                                      headers={"Content-Type": "application/json"})
                with request.urlopen(req, timeout=180) as resp:
                    d = json.loads(resp.read())
                obj = json.loads(d["choices"][0]["message"]["content"])
                return obj.get("subquery", "").strip(), d.get("usage", {})
            except Exception as e:
                if attempt == retries - 1:
                    return None, {"error": repr(e)}
                time.sleep(2 * (attempt + 1))
