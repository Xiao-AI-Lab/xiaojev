# 4B rank fusion + learned fusion (experiments 1+2)

**Red line: every parameter choice (RRF weights/constant, logistic
coefficients) is fit/selected on the 98 calibration questions ONLY and frozen
before dev/test are computed; test is reported once, never selected on.**

Date: 2026-09-26. Evaluation set: 293 MuSiQue non-training questions
(dev 94 / calibration 98 / test 101), candidate pool = NV-Embed-v2 dense
top-50 (same frozen candidate file as the v4 repair fusion experiment;
sha256 in [`fusion4b_config.json`](fusion4b_config.json)). 4B scores: the
lora4b run's dense top-50 scores (reused per question, 0 missing, no GPU
rerun). v4 scores: the v4 acceptance run's file (not committed; set
`XIAOJEV_4B_DENSE_SCORES` / `XIAOJEV_V4_DENSE_SCORES` to re-run).

## Experiment 1: 4B weighted-RRF fusion (protocol identical to the v4 repair)

Same `fusion.py::fuse_rankings`, same grid (constants {1,5,10,20,60} x weights
{0..1.0 step 0.1}), maximize calibration R@5, ties prefer smaller weight then
smaller constant, 5000 paired bootstraps (seed 20260924).

**Frozen config: reranker_weight = 0.5, rank_constant = 1** (v4 fusion:
0.4/1). Calibration R@5: dense 69.64% → fusion **81.21%** (v4 fusion 76.87%).

| Set | dense R@5 | v4 fusion R@5 | 4B fusion R@5 | 4B Δ vs dense [95% CI] | improved / worsened |
|---|---:|---:|---:|---|---|
| calibration (98) | 69.64 | 76.87 | 81.21 | +11.56 [+8.0, +15.3] | 32 / 1 |
| dev (94) | 68.88 | 74.11 | 76.15 | +7.27 [+3.0, +11.5] | 24 / 5 |
| **test (101)** | **73.35** | **77.31** | **79.79** | **+6.44 [+3.2, +9.8]** | 21 / 3 |
| nontrain (293) | 70.68 | 76.14 | 79.10 | +8.42 [+6.2, +10.6] | 77 / 9 |

Other 4B-fusion test metrics: R@4 77.48 / R@10 85.40 / R@20 87.46 / All@5
54.46 / MRR 0.949. Standalone reranking without fusion: 4B alone 73.07%
nontrain R@5; v4 alone 65.35% test R@5 (score saturation persists in
standalone mode — fusion is what combines dense-order stability with the
model's semantic judgment).

**End-to-end QA (top-4, same 27B reader, 293 questions):**

| Arm | EM | F1 |
|---|---:|---:|
| gold upper bound (measured 09-23) | 63.1 | 75.1 |
| **4B fusion top-4** | **39.9** | **49.8** |
| 4B standalone rerank top-4 (lora4b results) | 32.1 | 42.2 |
| dense top-4 (measured 09-23) | 31.1 | 41.0 |
| dense→v3 rerank top-4 (measured 09-23, saturation regression) | 23.2 | 30.9 |

4B-fusion QA EM 39.9 ties the 09-23 oracle in-pool v3 (39.9, privileged
20-candidate pools) and exceeds HippoRAGv2's 37.2 on the same subset (its
top-5 full pipeline). Test subset: EM 40.6 / F1 49.7
([`qa_metrics_fusion4b.json`](qa_metrics_fusion4b.json)).

## Experiment 2: logistic learned fusion vs manual RRF

Features = [bias, dense cosine similarity (min-max normalized per question),
xiaojev P(relevant)]; samples = 98 questions x 50 candidates = 4,900;
3-parameter IRLS (hand-written numpy, no sklearn); 5-fold CV by question
inside calibration only.

| Score source | Coefficients (dense / xiaojev) | CV5 calibration R@5 | test R@5: RRF → LR | Δ |
|---|---|---|---|---|
| v4 | 5.21 / 3.84 | 77.57% ± 5.0% | 77.31% → **77.97%** | +0.66 pp |
| 4B | 4.89 / 4.05 | 80.56% ± 6.0% | 79.79% → **79.87%** | +0.08 pp |

nontrain: v4 LR 76.68% vs RRF 76.14% (+0.54 pp); 4B LR 78.41% vs RRF 79.10%
(−0.68 pp).

**Verdict (a negative result, kept as-is):** learned fusion ≈ manual RRF —
v4 slightly up, 4B flat. The coefficients show both signals are used (dense
and xiaojev carry comparable weight), meaning RRF already captures most of
the fusable information; xiaojev's calibrated probability values add no extra
ranking gain over rank order. **Sample-size caveat:** fitting 3 parameters on
98 questions is not high overfit risk, but the CV standard deviation of
±5–6 pp means fold-to-fold noise dwarfs any sub-0.7 pp difference — no
conclusion can be drawn from LR's tiny edge. Recommendation: keep RRF as the
default (training-free, robust); revisit LR if a future score source (e.g. a
saturation-fixed v5) changes the picture.

## Artifacts

- Experiment 1: `eval_fusion_4b.py`, [`fusion4b_config.json`](fusion4b_config.json),
  [`fusion4b_calibration_trials.json`](fusion4b_calibration_trials.json),
  [`fusion4b_metrics.json`](fusion4b_metrics.json),
  [`fusion4b_rankings.json`](fusion4b_rankings.json), `run_qa_fusion4b.py`,
  [`qa_metrics_fusion4b.json`](qa_metrics_fusion4b.json)
  (reader predictions `qa_predictions_fusion4b.jsonl` regenerate at runtime).
- Experiment 2: `eval_fusion_logistic.py`,
  [`fusion_logistic_metrics.json`](fusion_logistic_metrics.json),
  `dense_sims_top50.jsonl` (dense-similarity cache, runtime).
- Reused: `fusion.py` (same function as the v4 repair), the frozen dense
  candidate file, and the two score files (read-only).
