# Dense first stage + answerability gate, re-tested (experiment 3)

**Red line: tau is calibrated on the 98 calibration questions only, at target
answered-precision >= 0.90, frozen before dev/test are computed; test is
reported once.**

Date: 2026-09-26. Code: [`gate_dense.py`](gate_dense.py) +
[`run_gate_dense.py`](run_gate_dense.py) — line-for-line the same protocol as
the BM25 version ([`run_gate.py`](run_gate.py)), with only the first stage
swapped to NV-Embed-v2 dense top-50 (retry round widens to dense top-100).
Same 293 MuSiQue non-training questions x {ans, unans} variants (unans = gold
documents removed, synthetic unanswerable). Reader: local qwen3.8-27b, same
settings as before. Gate model: `ckpt/v3`. All numbers:
[`gate_dense_metrics.json`](gate_dense_metrics.json).

## vs the BM25-stage gate (test, 101 questions/variant; BM25 column from `gate_metrics.json`)

| Metric | BM25 first stage (09-23) | dense first stage (this run) |
|---|---|---|
| Calibrated tau (precision >= 0.90) | 0.65 | **0.88** |
| Answerable keep rate (coverage) | 27.7% | **8.9%** |
| Refusal recall on unanswerable | 96.0% | **98.0%** |
| Hallucination rate (nogate → gate) | 32.7% → 1.0% | 32.7% → **0.0%** |
| Answered EM / F1 | 21.4 / 39.4 (28 questions) | 33.3 / 49.6 (only 9 questions, noisy) |
| Gate AUC (test, round 1) | 0.766 | 0.752 |
| nogate EM / F1 (answerable, top-5) | 16.8 / 26.9 | **24.8 / 33.2** |
| Mixed-stream reader prompt tokens | 187,812 → 30,667 (−83.7%) | 198,195 → 11,813 (**−94.0%**) |
| Avg retrieval rounds (ans/unans) | 1.76 / 1.96 | 1.91 / 1.98 |
| Reader mean latency | 4.3–5.0s | 5.8–6.4s (shared-server load window) |
| v3 scoring cost | 586+29,265+586 items / 18+293+18s | 586+29,276+586 items / 18+294+18s (same scale) |

## Calibration curves (calibration, 98 questions; answered precision / answerable coverage)

| tau | dense coverage | dense precision | BM25 coverage (same tau) | BM25 precision (same tau) |
|---|---|---|---|---|
| 0.05 | 0.898 | 0.677 | 0.765 | 0.664 |
| 0.30 | 0.531 | 0.754 | 0.439 | 0.811 |
| 0.50 | 0.429 | 0.808 | 0.316 | 0.861 |
| 0.65 | 0.337 | 0.825 | 0.245 | 0.923 |
| 0.88–0.90 | 0.082 | 1.000 | 0.041 | 1.000 |

**Key finding — the expected coverage jump did NOT happen at the 90%
precision operating point; coverage fell instead (27.7% → 8.9%).** At equal
tau, dense coverage is genuinely higher (tau = 0.5: 42.9% vs 31.6%) — the
Pareto frontier moves up as a whole. But the dense curve's high-precision
tail is worse: precision caps at ~83% for tau <= 0.85 and only reaches 100%
at tau >= 0.88, where just 8% coverage remains.

## Mechanism (why)

Splitting round-1 P(answerable) by whether the top-5 contains all gold
documents (293 questions):

| State | n | mean P | median P | P(>= 0.65) | P(>= 0.88) |
|---|---|---|---|---|---|
| answerable, top-5 gold-complete | 79 | 0.503 | 0.415 | 39.2% | 21.5% |
| answerable, top-5 gold-incomplete | 214 | 0.348 | 0.228 | 23.8% | 7.9% |
| synthetic unanswerable | 293 | 0.126 | 0.030 | 4.8% | 0.7% |

Two stacked bottlenecks:

1. **v3 reranking still hurts the dense pool:** dense-only top-5 All@5 =
   41.6%, but 27.0% after v3 reranking (the saturation problem reported
   09-23; the 4B/RRF fusion repairs exactly this, reaching 54.5% — see
   [FUSION4B_REPORT.md](FUSION4B_REPORT.md)).
2. **The gate systematically underestimates short dense contexts:** even with
   complete gold evidence, the median P is only ~0.42. v3's answerability
   channel was trained on 20-passage full/removal states; top-5 dense states
   are a different distribution, and the calibration did not transfer. This
   directly kills coverage in the high-precision band.

## Conclusions and next step

- The dense first stage itself works: nogate EM 16.8 → 24.8, and
  gold-completeness after v3 reranking 20.8% → 27.0%. Gate safety holds and
  is even cleaner under dense (hallucination 0.0%, refusal recall 98%).
- But coverage at the 90% precision operating point is choked by the v3
  gate's calibration mismatch. **The natural next combination: feed the
  fusion4b ordering (All@5 54.5%) into the gate, and retrain/recalibrate the
  answerability channel on short top-k contexts (v5)** — which is exactly the
  iterative-RAG experiment in [ITER_REPORT.md](ITER_REPORT.md).
- On cost, the gate's value is again prominent: −94% reader prompt tokens on
  the mixed stream, zero reader latency on refused questions.

## Artifacts

Committed: `gate_dense.py`, `run_gate_dense.py`,
[`gate_dense_metrics.json`](gate_dense_metrics.json) (full calibration curve +
per-split AUCs). Runtime (regenerable, git-ignored):
`gate_dense_scores.jsonl` (31,438 rows), `gate_dense_predictions.jsonl`
(695 reader predictions), `dense_corpus100.jsonl` (dense top-100 cache).
Resume-safe driver.
