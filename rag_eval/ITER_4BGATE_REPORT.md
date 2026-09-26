# Iterative RAG with the 4B gate (vs the v3 gate in ITER_REPORT.md)

Date: 2026-09-26. Code: [`run_iterative_4bgate.py`](run_iterative_4bgate.py).
Probe reuse: retrieval, 4B relevance scoring, and subqueries are all reused
from the v3-gate run's artifacts (`iter_retrieval` / `iter_scores` /
`iter_subqueries.jsonl`) — **only the gate phase was re-scored**: 4B LoRA
(`ckpt/qwen3_4b_lora_v1/step2500`) produces P(answerable) for each round's
accumulated evidence, 3×586 = 1,758 items. Gate phase 534s + 234s of reader
calls on changed trajectories = 769s total (vs 7302s for the full v3-gate
run). Reader determinism (temperature 0, fixed seed): the 1,686 predictions
whose trajectories (rounds + evidence doc_ids) did not change were reused
directly from the v3 run; only 325 fresh reader calls, zero failures.
Protocol, four arms, and the tau rule (98 calibration questions, answered
precision >= 0.90, max coverage) are identical to the v3-gate run. Artifacts
are separate: `iter_scores_4bgate.jsonl` / `iter_predictions_4bgate.jsonl` /
[`iter_metrics_4bgate.json`](iter_metrics_4bgate.json); the v3 artifacts are
untouched.

## tau calibration: the 4B probability distribution is visibly healthier

**v3: τ = 0.64 → 4B: τ = 0.37.** v3's probabilities are compressed at the
bottom (answerable round-1 mean P very low; τ had to be hard-picked on a
compressed distribution); 4B's calibration curve is smooth and monotonic,
precision rising steadily from 0.69 (τ = 0.05) to 1.0 (τ ≥ 0.80) without the
v3 curve's jitters:

| tau | answerable coverage (4B) | answered precision (4B) | v3 coverage / precision |
|---|---|---|---|
| 0.05 | 0.969 | 0.693 | 0.990 / 0.571 |
| 0.20 | 0.847 | 0.874 | 0.939 / 0.687 |
| **0.37** | **0.806** | **0.898~0.90** | — |
| 0.40 | 0.786 | 0.917 | 0.847 / 0.798 |
| 0.65 | 0.684 | 0.957 | 0.663 / 0.903 |
| 0.80 | 0.530 | 1.000 | — |

(τ = 0.37 sits between the stored 5% grid points 0.35–0.40; coverage/precision
read from neighboring grid values in `iter_metrics_4bgate.json`.)

**Gate discrimination (AUC, P = max over 3 rounds):** 4B = 0.937
(calibration) / 0.933 (dev) / **0.914** (test); v3 = 0.903 / 0.878 / 0.882.
Round-1-only AUC: 4B 0.857 vs v3 0.815 (test). The 4B gate is better
everywhere.

## Question 1: do "one round is enough" questions stop at round one? — Yes

Gated-arm trajectories (test, 101 questions/variant):

| Variant | accept r1 | r2 | r3 | exhausted | avg_rounds |
|---|---|---|---|---|---|
| ans, v3 gate | 26 | 33 | 12 | 30 | 2.16 |
| ans, **4B gate** | **61** | 15 | 6 | 19 | **1.64** |
| unans, v3 gate | 1 | 7 | 2 | 91 | 2.91 |
| unans, **4B gate** | 7 | 3 | 1 | 90 | 2.83 |

Answerable round-1 acceptance 26/101 → **61/101** (2.3x), avg_rounds 2.16 →
1.64; unanswerable questions still run all 3 rounds 90/101 (leaked to reader:
10 → 11, flat). The gate finally separates "one round suffices" from "keep
retrieving".

## Question 2: four-arm comparison (test, 101/arm; v3 → 4B)

Answerable:

| Arm | EM | F1 | answered-only EM/F1 | answered/refused | avg_rounds | prompt tokens |
|---|---|---|---|---|---|---|
| single | 0.406 → 0.406 | 0.521 → 0.521 | same | 101/0 | 1.00 | 97.8k (unchanged) |
| fixed2 | 0.525 → 0.525 | 0.617 → 0.617 | same | 101/0 | 2.00 | 223.6k (unchanged) |
| gated | 0.515 → **0.465** | 0.618 → 0.581 | same | 101/0 | 2.16 → **1.64** | 255.1k → **185.3k (−27%)** |
| gated_refuse | 0.366 → 0.376 | 0.452 → 0.482 | 0.521/0.643 → 0.463/0.594 | 71/30 → **82/19** | 2.16 → 1.64 | 141.1k → 114.4k |

Unanswerable:

| Arm | hallucination | abstain-or-refused | avg_rounds | prompt tokens |
|---|---|---|---|---|
| single / fixed2 | 53.5% / 59.4% (unchanged) | — | — | unchanged |
| gated (answer on exhaust) | 62.4% → 63.4% | 37.6% → 36.6% | 2.91 → 2.83 | 366.3k → 353.3k |
| gated_refuse | **6.9% → 8.9%** | 93.1% → 91.1% | 2.91 → 2.83 | 26.0k → **18.9k** |

**Refusal quality (test):** precision 75.2% → **82.6%**; recall 90.1% →
89.1% (flat); answerable keep rate 70.3% → **81.2%** (+10.9 pp).
**Mixed traffic (202 pipelines) prompt tokens:** gated 621.4k → 538.6k
(−13%); gated_refuse 167.2k → 133.3k (−20%).

**dev corroboration:** gated EM 0.479 → 0.479 (unchanged — so test's −5 pp is
partly small-sample noise); gated_refuse answered-only EM 0.532 → 0.552,
hallucination 7.4% → 6.4% (4B is better across the board on dev); refusal
precision 72.4% → 76.3%, keep rate 66.0% → 71.3%; answerable round-1
acceptance 51/94, avg_rounds 2.18 → 1.78. calibration trends the same.

## Interpretation

1. **The 4B gate solves what the gate swap was meant to solve:** round-1
   acceptance 26 → 61/101, avg_rounds 2.16 → 1.64, gated-arm tokens −27% —
   the leftover "the gate saves no cost on pure-answerable traffic" issue is
   gone. τ = 0.37 sits on a smooth monotonic calibration curve (no longer
   hard-picked on a compressed distribution), and test/dev/calibration
   refusal metrics corroborate each other — the calibration transfers well.
2. **The cost, stated plainly: part of what earlier stopping buys is evidence
   quality.** gated-arm test EM 0.515 → 0.465 (dev flat): more questions stop
   at round 1, so the reader sees 5 passages instead of 10–15. The 4B gate's
   "answerable" judgment is right (AUC 0.914), but "answerable" is not
   "answered best" — v3's conservatism was accidentally acting as a retrieval
   deepener. If answer quality comes first, raise τ to 0.65 (calibration
   precision 0.957, coverage 0.684) or switch to an "accept, then still take
   one more round before answering" policy.
3. **The overall account favors 4B:** refusal precision +7.4 pp, answerable
   keep rate +10.9 pp, mixed-traffic tokens −13% to −20%, rounds −0.5; the
   price is hallucination +2.0 pp (6.9 → 8.9%; dev moves the *other* way,
   −1.0 pp) and answered-only EM wobble (test −5.8 pp / dev +2.0 pp).
4. **Conclusion: the 4B gate becomes the default.** Discrimination (AUC
   0.914 vs 0.882), calibration health, cost control, and keep rate are all
   better; the small hallucination give-back can be absorbed by a τ nudge
   (the curve is smooth: 0.37 → 0.50 moves precision 0.898 → 0.911) or by
   reader-side abstention (the 36.6% empty-answer rate is still there).
   The v3-gate results stay in `iter_metrics.json` / ITER_REPORT.md as the
   control.

## Artifacts

- Committed: `run_iterative_4bgate.py`,
  [`iter_metrics_4bgate.json`](iter_metrics_4bgate.json) (every number above +
  calibration curve + gated-arm trajectory distribution).
- Runtime (regenerable, git-ignored): `iter_scores_4bgate.jsonl` (1,758 4B
  gate scores, kind=gate4b), `iter_predictions_4bgate.jsonl` (2,011 rows:
  1,686 reused + 325 fresh reader calls, `reused_from_v3_run` flag).
  Resume conventions unchanged (item-level gate scores, key-level
  predictions).
