# Dense retrieval with xiaojev v4

The repair combines the dense retriever's order with the original v4 model's order. It uses weighted reciprocal-rank fusion, with dense weight 0.6, reranker weight 0.4, and rank constant 1. Parameters were selected on 98 calibration questions and frozen before evaluating dev/test.

```python
from rag_eval.dense_reranker import DenseReranker

reranker = DenseReranker(
    checkpoint="ckpt/v4",
    config_path="results/v4_repair/fusion_config.json",
)
# Passages are dictionaries with docid, title, text, in descending dense-score order.
ranked = reranker.rerank(question, passages[:50])
```

This interface scores the supplied candidates on CUDA; it requires the original v4 weights and the base tokenizer (`XIAOJEV_BASE_MODEL`, default `Qwen/Qwen3-0.6B`). It does not load gold labels or the frozen evaluation inputs. The returned `xiaojev_p_relevant` is the original model score; the fused ranking is not a calibrated probability.

## Recompute the reported metrics

From the repository root, using only Python's standard library:

```bash
python -m rag_eval.evaluate_fusion --verify-calibration --output results/recomputed_fusion.json
```

The committed `results/v4_repair/retrieval_inputs.jsonl` contains document IDs, frozen dense rankings, original v4 relevance scores, gold document IDs, and split labels for all 293 non-training questions. It contains no corpus text. This command reproduces the calibration choice, split metrics, and paired bootstrap interval; it does not rerun the embedding model or reader.

Independent test (101 questions): dense R@5 73.35%, original v4-only reordering 65.35%, fusion 77.31%. The paired bootstrap 95% interval for the gain over dense is +0.91 to +7.01 percentage points. The 293-question total includes calibration and must not be called an independent test result.

## Data preparation for the original v4 training

Set `XIAOJEV_RAG_DATA` to a directory with `musique/raw/musique.json`, `musique/gold.jsonl`, and `musique/corpus.jsonl`, using the normalized dataset format described in the main README.

```bash
python -m rag_eval.bm25_rank
python data/make_ragdata.py
python scripts/check_rag_data.py
```

BM25 outputs default to `results/retrieval`; override with `XIAOJEV_RETRIEVAL_DIR`. Dense hard negatives can reuse an NV-Embed-v2 index: `XIAOJEV_DENSE_INDEX_ROOT` must contain `index/inputs.json` with a `chunk` list of `{id, content}` records and `index/chunk_vectors.npy` in the same row order. The embedding endpoint is set with `XIAOJEV_EMBED_URL` (default `http://127.0.0.1:8019/v1/embeddings`). Without that index/service, the generator explicitly reports its BM25 fallback; that fallback does not exactly reproduce the original dense-negative training run.

QA scores and the small test-set QA improvement are recorded in [the release report](../docs/V4_REPAIR.md). Reader baselines are retained same-configuration runs, not fresh paired reruns.

## Experiments in this directory

Each report lists its committed code and summary JSONs; large score caches and
reader predictions regenerate at runtime (git-ignored `rag_eval/*.jsonl`).

- **[GATE_REPORT.md](GATE_REPORT.md)** — BM25 first stage + v3 answerability
  gate (v3 era). One line: hallucination on unanswerable questions 32.7% →
  1.0% and −83.7% reader tokens, but only 27.7% of answerable questions kept
  (BM25 top-5 evidence completeness is the bottleneck, not the gate).
- **[FUSION4B_REPORT.md](FUSION4B_REPORT.md)** — 4B weighted-RRF fusion
  (reranker weight 0.5, constant 1) and a learned-fusion control. One line:
  test R@5 **79.79%** vs dense 73.35% (+6.44 pp, 95% CI [+3.2, +9.8]) and QA
  EM 39.9 tying the oracle-pool reader; logistic learned fusion ties RRF
  (negative result — RRF stays the default).
- **[GATE_DENSE_REPORT.md](GATE_DENSE_REPORT.md)** — the gate re-run with a
  dense first stage. One line (inversion): coverage *drops* to 8.9% at the
  90% precision point even though the dense Pareto frontier is higher —
  because v3 reranking still damages the dense pool and the v3 gate
  underestimates short dense contexts; hallucination reaches 0.0%.
- **[ITER_REPORT.md](ITER_REPORT.md)** — calibration-gated iterative RAG,
  four arms. One line: iterative retrieval is a double-edged sword
  (answerable EM 40.6 → 52.5, unanswerable hallucination 53.5% → 62.4%);
  `gated_refuse` is the only arm that wins both sides (answered EM 52.1,
  hallucination 6.9%, −73% mixed-traffic tokens, answerable keep rate 70.3%
  vs the old BM25 gate's 27.7% — iteration rescued gate coverage with no
  retraining). On pure-answerable traffic the gate adds nothing and costs
  more: its entire value is in mixed traffic.
