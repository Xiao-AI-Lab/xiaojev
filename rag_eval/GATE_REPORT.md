# Answerability-gated RAG, end to end (MuSiQue, 293 non-training questions)

Date: 2026-09-23. Code: [`gate.py`](gate.py) (reusable pipeline) +
[`run_gate.py`](run_gate.py) (evaluation driver), scoring via
[`score_v3.py`](score_v3.py), reader calls via [`run_qa.py`](run_qa.py).
Model: `ckpt/v3` (Qwen3-0.6B + EOS head). Reader: local qwen3.8-27b
(temperature 0, seed 20260917, max_tokens 2048, JSON schema, same template as
the RAG eval). All numbers: [`gate_metrics.json`](gate_metrics.json).

## Design

```
BM25 top-50 -> v3 relevance-channel rerank, take top-5 -> v3 answerability gate P(answerable)
  P >= tau -> 27B reader answers
  P <  tau -> one retry: pool widened to BM25 top-100, new candidates scored,
              top-5 re-selected, re-gated; still < tau -> refuse (no reader call)
```

- Evaluation: 293 questions x 2 variants = 586 pipeline runs. `ans` = normal
  corpus; `unans` = the question's gold documents removed from every candidate
  stage, simulating missing evidence. **The local MuSiQue gold set is fully
  answerable, so unanswerable cases can only be constructed synthetically
  this way.**
- Splits reuse the semantic-data hash: calibration (98) sets tau, test (101)
  is the primary report, dev (94) is secondary.
- tau rule: sweep tau in [0.01, 0.99] on calibration, require answered-set
  answerable-precision >= 0.90, take the tau with the highest answerable
  coverage -> **tau = 0.65** (calibration: answered precision 92.3%,
  answerable coverage 24.5%).
- Arms: nogate (all 586 go straight to the reader) vs gate (above).

## Calibration curve (calibration, n=98 x 2)

| tau | answerable coverage | answered precision | n_answered |
|---|---|---|---|
| 0.05 | 0.765 | 0.664 | 113 |
| 0.20 | 0.500 | 0.778 | 63 |
| 0.30 | 0.439 | 0.811 | 53 |
| 0.50 | 0.316 | 0.861 | 36 |
| **0.65** | **0.245** | **0.923** | 26 |
| 0.70 | 0.184 | 0.947 | 19 |
| 0.85 | 0.071 | 1.000 | 7 |

## Main results (test split, 101 questions per variant)

| Variant x arm | answered | refused | EM | F1 | EM/F1 (answered only) | hallucination rate | abstain-or-refused |
|---|---|---|---|---|---|---|---|
| ans x nogate | 101 | 0 | 0.168 | 0.269 | 0.168 / 0.269 | — | — |
| ans x gate | 28 | 73 | 0.059* | 0.109* | **0.214 / 0.394** | — | — |
| unans x nogate | 101 | 0 | — | — | — | **32.7%** | 67.3% |
| unans x gate | 4 | 97 | — | — | — | **1.0%** | **99.0%** |

*The gate arm's full-set EM/F1 counts refusals as zero — that is the coverage
cost; the meaningful columns are "answered only" and the hallucination rate.

**Refusal quality (test):** refusal recall 96.0% (97/101 unanswerable
correctly refused); refusal precision 57.1% (of 170 refused, 97 truly
unanswerable); answerable-question keep rate 27.7%.
**Gate discrimination:** P(answerable) AUC = 0.766 (test, round 1) / 0.770
(including retry); calibration 0.808; dev 0.778. Lower than the 0.936 of the
simpler full-vs-drop_all probe, as expected: there the gate compared 20
passages with vs without gold; here it judges whether the *selected top-5*
suffice — and the BM25 first stage's top-5 often misses gold even for
answerable questions (BM25->v3 cascade All@5 is only 0.208), which the gate
correctly scores down.

## Cost account (test split, 202 mixed runs)

| Arm | reader prompt tokens | completion tokens | avg retrieval rounds | per-question latency |
|---|---|---|---|---|
| nogate | 187,812 | 1,737 | 1.0 | v3 relevance+gate ~0.6s + reader ~5.0s ≈ 5.6s |
| gate | **30,667 (−83.7%)** | **303 (−82.6%)** | 1.86 | answered: ~0.6–1.2s v3 + ~4.3–5.1s reader; refused: ~1.2s, no reader |

Measured v3-side throughput: gate items 31 ms each (586 in 18s), retry
relevance 10 ms each (29,265 in 293s) on one RTX 3090; per-question v3 cost
0.6s (one round) / 1.15s (two rounds). Refusals are 84% of the mixed stream
(170/202), so the gate arm averages ≈ **1.8s per question vs ≈ 5.6s** for
nogate.

## Interpretation

1. **The gate's core value holds:** hallucination rate on unanswerable
   questions drops 32.7% → 1.0% (33x), while answered-question quality *rises*
   (EM 16.8 → 21.4, F1 26.9 → 39.4) because evidence-poor questions are
   filtered out. tau is set directly from a small calibration set at a target
   precision — no training — a practical use of xiaojev's calibration.
2. **The cost is coverage:** only 27.7% of answerable questions are kept.
   The bottleneck is **not** gate discrimination (AUC 0.77 is adequate) but
   the **BM25 first stage's top-5 evidence completeness** (All@5 ≈ 0.21) —
   most refusals are *correct* refusals (top-5 genuinely cannot answer the
   question). A stronger dense first stage plus the retrained v4 reranker
   should raise coverage and precision together.
3. The retry round contributes 76% of retrieval rounds but rescues few
   questions (9 of the 28 gate-arm answers come from round 2); in production,
   trigger retries only for borderline score bands.

## Artifacts

- Committed: `gate.py`, `run_gate.py`, `score_v3.py`, `run_qa.py`,
  `gate_metrics.json` (every number above + the full calibration curve).
- Runtime (regenerable, git-ignored): `gate_scores.jsonl` (31,437 rows),
  `gate_predictions.jsonl` (665 reader predictions with latency/token usage).
  Both v3 scoring and reader calls resume from partial outputs.
