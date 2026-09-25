"""End-to-end QA: retrieval arms -> local LLM reader -> EM/F1.

Arms (all on the 293 non-train questions, top-4 passages each):
  bm25_inpool   BM25 ranking of the question's own 20 paragraphs
  v3_inpool     v3 relevance-channel ranking of the same 20 paragraphs
  v3_cascade    BM25 corpus top-50 reranked by v3 relevance channel
  gold          gold paragraphs only (reader upper bound)

Reader prompt mirrors the HippoRAGv2/LineageRAG shared reader (empty guides),
JSON-schema answer, temperature 0, max_tokens 2048, seed 20260917.
Writes qa_predictions.jsonl and qa_metrics.json.

Configuration (environment variables):
  XIAOJEV_GATE_DIR      score/output directory (default: this rag_eval/ directory)
  XIAOJEV_READER_URL    OpenAI-compatible base URL of the reader
                        (default: http://127.0.0.1:8020/v1; we served qwen3.8-27b)
  XIAOJEV_READER_MODEL  served reader model name (default: qwen3.8-27b)
"""
import json
import os
import re
import string
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.common import load_corpus, load_questions  # noqa: E402

OUT = Path(os.environ.get("XIAOJEV_GATE_DIR", Path(__file__).resolve().parent))
API = os.environ.get("XIAOJEV_READER_URL", "http://127.0.0.1:8020/v1").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("XIAOJEV_READER_MODEL", "qwen3.8-27b")
NONTRAIN = ("dev", "calibration", "test")
TOPK = 4
CONCURRENCY = 8

SCHEMA = {"name": "grounded_short_answer", "strict": True, "schema": {
    "type": "object", "properties": {"answer": {"type": "string"}},
    "required": ["answer"], "additionalProperties": False}}

SYSTEM = "You are a long-document QA reader. Give concise answers from the context."
USER_TAIL = (
    "Question: QUESTION_PLACEHOLDER\n\n"
    "Answer with the shortest exact phrase supported by the context passages. "
    "When the question asks for a list or order, include every supported item. "
    "Return only answer values; never repeat the question wording. "
    "For who or name questions return only the person or role; for where return "
    "only the location; for when return only the time or event phrase; for why or "
    "how return only the cause or mechanism. Never attach the predicate copied "
    "from the question to an otherwise sufficient answer value.\n"
    "Do not explain or write an analysis. Return exactly one JSON object with "
    "one field: {\"answer\": \"<short answer phrase>\"}. The answer field must contain "
    "only the answer values, never reasoning, a step-by-step breakdown, or a preamble. "
    "If the passages do not support an answer, use an empty answer string."
)


def normalize_answer(text):
    lowered = str(text).lower()
    nopunc = "".join(c for c in lowered if c not in string.punctuation)
    noart = re.sub(r"\b(a|an|the)\b", " ", nopunc)
    return " ".join(noart.split())


def exact_match(pred, aliases):
    n = normalize_answer(pred)
    return float(any(n == normalize_answer(a) for a in aliases))


def _f1(pred, ans):
    pt, at = normalize_answer(pred).split(), normalize_answer(ans).split()
    if not pt or not at:
        return float(pt == at)
    overlap = sum((Counter(pt) & Counter(at)).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pt), overlap / len(at)
    return 2 * p * r / (p + r)


def token_f1(pred, aliases):
    return max((_f1(pred, a) for a in aliases), default=0.0)


def call_reader(question, passages, retries=3):
    ctx = "\n\n".join(f"[{i+1}] source_doc_id={pid}\n{text}" for i, (pid, text) in enumerate(passages))
    user = f"Context passages:\n{ctx}\n\n" + USER_TAIL.replace("QUESTION_PLACEHOLDER", question)
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0, "max_tokens": 2048, "seed": 20260917,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
    }
    for attempt in range(retries):
        try:
            req = request.Request(API, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
            with request.urlopen(req, timeout=180) as resp:
                d = json.loads(resp.read())
            content = d["choices"][0]["message"]["content"]
            obj = json.loads(content)
            return obj.get("answer", "").strip(), d.get("usage", {})
        except Exception as e:
            if attempt == retries - 1:
                return None, {"error": repr(e)}
            time.sleep(2 * (attempt + 1))


def main():
    questions = {q["id"]: q for q in load_questions() if q["split"] in NONTRAIN}
    corpus = {c["docid"]: c for c in load_corpus()}

    # rankings
    bm25_in = {}
    for line in open(OUT / "bm25_inpool.jsonl"):
        d = json.loads(line)
        if d["id"] in questions:
            bm25_in[d["id"]] = d["ranking"]
    bm25_top50 = {}
    for line in open(OUT / "bm25_corpus.jsonl"):
        d = json.loads(line)
        if d["id"] in questions:
            bm25_top50[d["id"]] = d["top50"]
    rel = defaultdict(dict)
    cascade = defaultdict(dict)
    for line in open(OUT / "v3_scores.jsonl"):
        d = json.loads(line)
        if d["kind"] == "relevance":
            rel[d["qid"]][d["pidx"]] = d["p_yes"]
        elif d["kind"] == "cascade":
            cascade[d["qid"]][d["docid"]] = d["p_yes"]

    def para_text(qid, pidx):
        return next(t for i, _, t in questions[qid]["paragraphs"] if i == pidx)

    def arms_for(qid):
        q = questions[qid]
        bm = bm25_in[qid][:TOPK]
        v3 = sorted(rel[qid], key=lambda i: -rel[qid][i])[:TOPK]
        cas_sc = cascade.get(qid, {})
        cas = sorted(bm25_top50.get(qid, []), key=lambda d: -cas_sc.get(d, 0.0))[:TOPK]
        return {
            "bm25_inpool": [(f"p{i}", para_text(qid, i)) for i in bm],
            "v3_inpool": [(f"p{i}", para_text(qid, i)) for i in v3],
            "v3_cascade": [(d, corpus[d]["text"]) for d in cas if d in corpus],
            "gold": [(f"p{i}", para_text(qid, i)) for i in q["gold_idx"]],
        }

    tasks = []
    for qid in sorted(questions):
        for arm, passages in arms_for(qid).items():
            tasks.append((qid, arm, passages))
    print(f"{len(tasks)} reader calls ({len(questions)} questions x {len(tasks)//len(questions)} arms)")

    preds_f = open(OUT / "qa_predictions.jsonl", "w")
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(task):
        qid, arm, passages = task
        q = questions[qid]
        answer, usage = call_reader(q["question"], passages)
        aliases = [q["answer"]] + q["answer_aliases"]
        rec = {
            "id": qid, "arm": arm, "hop": q["hop"], "split": q["split"],
            "prediction": answer,
            "doc_ids": [pid for pid, _ in passages],
            "em": None if answer is None else exact_match(answer or "", aliases),
            "f1": None if answer is None else token_f1(answer or "", aliases),
        }
        with lock:
            preds_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 100 == 0:
                preds_f.flush()
                print(f"{done[0]}/{len(tasks)} calls, {time.time()-t0:.0f}s", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(work, tasks))
    preds_f.close()

    # metrics
    rows = [json.loads(l) for l in open(OUT / "qa_predictions.jsonl")]
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    metrics = {}
    for arm, rs in sorted(by_arm.items()):
        ok = [r for r in rs if r["em"] is not None]
        m = {"n": len(rs), "failures": len(rs) - len(ok),
             "em": sum(r["em"] for r in ok) / len(ok) if ok else None,
             "f1": sum(r["f1"] for r in ok) / len(ok) if ok else None}
        for h in (2, 3, 4):
            sub = [r for r in ok if r["hop"] == h]
            if sub:
                m[f"hop{h}"] = {"n": len(sub),
                                "em": sum(r["em"] for r in sub) / len(sub),
                                "f1": sum(r["f1"] for r in sub) / len(sub)}
        metrics[arm] = m
    with open(OUT / "qa_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
