"""Experiment 1: weighted-RRF fusion with 4B LoRA scores, protocol identical to
the v4 repair fusion (same candidate file, same grid, same 98 calibration
questions, same tie-break, same bootstrap seed). Only the reranker score file
changes: 4B LoRA dense top-50 scores.

Outputs: fusion4b_config.json / fusion4b_calibration_trials.json /
fusion4b_metrics.json / fusion4b_rankings.json in the rag_eval directory.

Configuration (environment variables):
  XIAOJEV_RAG_EVAL_DIR     working/output directory (default: this rag_eval/)
  XIAOJEV_4B_DENSE_SCORES  4B LoRA dense top-50 scores JSONL (required;
                           produced by the lora4b evaluation run)
"""
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.common import load_questions  # noqa: E402
from rag_eval.metrics import agg, rank_metrics  # noqa: E402
from rag_eval.fusion import fuse_rankings  # noqa: E402

OUT = Path(os.environ.get("XIAOJEV_RAG_EVAL_DIR", Path(__file__).resolve().parent))
SCORE_FILE = os.environ.get("XIAOJEV_4B_DENSE_SCORES")


def main():
    if not SCORE_FILE:
        raise SystemExit(
            "Set XIAOJEV_4B_DENSE_SCORES to the 4B LoRA dense top-50 scores JSONL "
            "(an artifact of the lora4b evaluation run; not committed to git)."
        )
    score_file = Path(SCORE_FILE)
    questions = {q["id"]: q for q in load_questions()}
    candidate_file = OUT / "dense_corpus.jsonl"
    dense = {r["id"]: r for r in map(json.loads, candidate_file.open())}
    scores = defaultdict(dict)
    for r in map(json.loads, score_file.open()):
        scores[r["qid"]][r["docid"]] = r["probs"][0]
    assert len(dense) == 293 and all(questions[q]["split"] != "train" for q in dense)
    calibration = sorted(q for q in dense if questions[q]["split"] == "calibration")
    assert len(calibration) == 98

    def rows(qids, config):
        result = []
        for q in qids:
            ranking = fuse_rankings(dense[q]["top50"], scores[q], **config)
            result.append(
                rank_metrics(ranking, set(dense[q]["gold_docids"]), (1, 4, 5, 10, 20, 50))
            )
        return result

    trials = []
    for constant in (1, 5, 10, 20, 60):
        for weight in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1):
            config = dict(reranker_weight=weight, rank_constant=constant)
            metric = agg(rows(calibration, config))
            trials.append(dict(config=config, calibration_r5=metric["r@5"]))
    selected = max(
        trials,
        key=lambda t: (
            t["calibration_r5"],
            -t["config"]["reranker_weight"],
            -t["config"]["rank_constant"],
        ),
    )
    frozen = {
        "method": "weighted reciprocal rank fusion",
        "config": selected["config"],
        "selection": "maximize calibration R@5; ties prefer smaller reranker weight, then rank constant",
        "calibration_ids": calibration,
        "calibration_r5": selected["calibration_r5"],
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (candidate_file, score_file)
        },
    }
    with (OUT / "fusion4b_config.json").open("w") as handle:
        json.dump(frozen, handle, indent=2)
    (OUT / "fusion4b_calibration_trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    config = selected["config"]
    report = {"frozen_config": config, "score_source": str(score_file), "splits": {}}
    for split in ("calibration", "dev", "test", "nontrain"):
        ids = sorted(q for q in dense if split == "nontrain" or questions[q]["split"] == split)
        baseline = rows(ids, dict(reranker_weight=0, rank_constant=1))
        fused = rows(ids, config)
        diffs = [f["r@5"] - b["r@5"] for b, f in zip(baseline, fused)]
        rng = random.Random(20260924)
        bootstrap = sorted(
            sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(5000)
        )
        report["splits"][split] = {
            "dense": agg(baseline),
            "fusion": agg(fused),
            "delta_r5": sum(diffs) / len(diffs),
            "paired_bootstrap_delta_r5_95pct": [bootstrap[125], bootstrap[4874]],
            "improved_questions": sum(d > 0 for d in diffs),
            "worsened_questions": sum(d < 0 for d in diffs),
        }
    rankings = {q: fuse_rankings(r["top50"], scores[q], **config) for q, r in dense.items()}
    (OUT / "fusion4b_rankings.json").write_text(json.dumps(rankings, indent=2) + "\n")
    (OUT / "fusion4b_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({s: {"dense_r5": v["dense"]["r@5"], "fusion_r5": v["fusion"]["r@5"],
                          "delta": v["delta_r5"], "ci": v["paired_bootstrap_delta_r5_95pct"]}
                      for s, v in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
