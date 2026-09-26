"""QA arm for experiment 1: frozen 4B-RRF fusion top-4 -> reader.
Same reader/prompt/protocol as run_qa.py; 293 non-train questions.
Writes qa_predictions_fusion4b.jsonl + qa_metrics_fusion4b.json."""
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.common import load_corpus, load_questions  # noqa: E402
from rag_eval.run_qa import OUT, CONCURRENCY, call_reader, exact_match, token_f1  # noqa: E402

NONTRAIN = ("dev", "calibration", "test")
TOPK = 4


def main():
    questions = {q["id"]: q for q in load_questions() if q["split"] in NONTRAIN}
    corpus = {c["docid"]: c for c in load_corpus()}
    rankings = json.load(open(OUT / "fusion4b_rankings.json"))

    tasks = []
    for qid in sorted(questions):
        passages = [(d, corpus[d]["text"]) for d in rankings[qid][:TOPK] if d in corpus]
        tasks.append((qid, passages))
    print(f"{len(tasks)} reader calls (fusion4b top-{TOPK})")
    preds_f = open(OUT / "qa_predictions_fusion4b.jsonl", "w")
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(task):
        qid, passages = task
        q = questions[qid]
        answer, usage = call_reader(q["question"], passages)
        aliases = [q["answer"]] + q["answer_aliases"]
        rec = {"id": qid, "arm": "fusion4b", "hop": q["hop"], "split": q["split"],
               "prediction": answer, "doc_ids": [p for p, _ in passages],
               "em": None if answer is None else exact_match(answer or "", aliases),
               "f1": None if answer is None else token_f1(answer or "", aliases),
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens")}
        with lock:
            preds_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 100 == 0:
                preds_f.flush()
                print(f"{done[0]}/{len(tasks)}, {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(work, tasks))
    preds_f.close()

    rows = [json.loads(l) for l in open(OUT / "qa_predictions_fusion4b.jsonl")]
    ok = [r for r in rows if r["em"] is not None]
    metrics = {"fusion4b": {"n": len(rows), "failures": len(rows) - len(ok),
                            "em": sum(r["em"] for r in ok) / len(ok),
                            "f1": sum(r["f1"] for r in ok) / len(ok)}}
    for h in (2, 3, 4):
        sub = [r for r in ok if r["hop"] == h]
        if sub:
            metrics["fusion4b"][f"hop{h}"] = {"n": len(sub),
                                              "em": sum(r["em"] for r in sub) / len(sub),
                                              "f1": sum(r["f1"] for r in sub) / len(sub)}
    # also per eval split for the red-line report
    for s in ("dev", "calibration", "test"):
        sub = [r for r in ok if r["split"] == s]
        metrics["fusion4b"][s] = {"n": len(sub),
                                  "em": sum(r["em"] for r in sub) / len(sub),
                                  "f1": sum(r["f1"] for r in sub) / len(sub)}
    with open(OUT / "qa_metrics_fusion4b.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
