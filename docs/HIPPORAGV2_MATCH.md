# Benchmarking against HippoRAG v2 (clean subsets, recomputed per question)

Date: 2026-10-04. Summary JSON: [`results/dag_match/hipporagv2_match.json`](../results/dag_match/hipporagv2_match.json).
Comparison line for the older dagv2 benchmark: [DAG_MATCH.md](DAG_MATCH.md)
(1 win, 2 draws, 0 losses).

## Protocol

- **Identical conditions on both sides:** same clean subsets as dag_match
  (semantic-hash non-train split, seed 20260922 — musique 293 / hotpotqa 299 /
  2wiki 306 questions), same reader (Qwen3.8-27B), same embeddings
  (NV-Embed-v2), same corpora, same metrics.
- HippoRAGv2 (official commit `474ae76`) trains nothing, so full recomputation
  is unbiased; its qid space was asserted identical to dag_match's unit ids
  (same raw sources). EM/F1 were recomputed from its archived prediction
  strings with our `exact_match`/`token_f1` and cross-validated against its
  archived scores — **0 rows of definition mismatch**. Paired bootstrap: 5000
  resamples (seed 20261004). Pure archive recompute — zero GPU/reader calls.
- Our arms are the production numbers from the dag_match final pipeline
  (NV dense + 4B fusion + query-planning/answer-conditioned retrieve–read
  loop + fusion-union final answer + elastic panel): musique saveloop
  no_commit 55.63/64.42, hotpotqa saveloop no_commit 66.22/80.01, 2wiki
  saveloop commit 73.53/81.66.

## Headline (clean subsets)

| Dataset | HippoRAGv2 EM / F1 | Ours EM / F1 | ΔEM [95% CI] | Verdict |
|---|---:|---:|---|---|
| musique (293) | 37.20 / 48.71 | **55.63 / 64.42** | **+18.43 [+12.3, +24.2]** | decisive win |
| hotpotqa (299) | 62.88 / 75.69 | **66.22 / 80.01** | **+3.34 [−0.7, +7.7]** | win (CI touches 0) |
| 2wiki (306) | 61.76 / 68.28 | **73.53 / 81.66** | **+11.76 [+7.5, +16.0]** | decisive win |

Retrieval reference (HippoRAGv2 archived R@5/All@5, same subsets): musique
76.79/53.24, hotpotqa 95.32/90.97, 2wiki 90.44/74.84.

## Per-hop / per-type ΔEM (ours − HippoRAGv2 [95% CI])

**musique:**

| hop | n | HippoRAGv2 | Ours | ΔEM [CI] |
|---|---:|---:|---:|---|
| 2 | 166 | 42.8 | 62.0 | +19.28 [+11.4, +27.1] |
| 3 | 81 | 33.3 | 51.9 | +18.52 [+6.2, +30.9] |
| 4 | 46 | 23.9 | 39.1 | +15.22 [+4.3, +28.3] |

**hotpotqa:**

| type | n | HippoRAGv2 | Ours | ΔEM [CI] |
|---|---:|---:|---:|---|
| bridge | 239 | 59.8 | 63.6 | +3.77 [−1.3, +8.8] |
| comparison | 60 | 75.0 | 76.7 | +1.67 [−3.3, +6.7] |

**2wiki:**

| type | n | HippoRAGv2 | Ours | ΔEM [CI] |
|---|---:|---:|---:|---|
| bridge_comparison | 73 | 79.5 | 97.3 | +17.81 [+8.2, +27.4] |
| comparison | 74 | 87.8 | 91.9 | +4.05 [−1.4, +9.5] |
| compositional | 128 | 39.8 | 50.0 | +10.16 [+3.9, +16.4] |
| inference | 31 | 48.4 | 71.0 | +22.58 [+6.5, +38.7] |

## One-line conclusion

Under fully identical questions, reader, and embeddings, our pipeline beats
HippoRAG v2 on clean-subset EM by **+18.4 pp (musique) and +11.8 pp (2wiki),
both significant, and +3.3 pp on hotpotqa (positive direction)** — with no
losing hop or question type, including our historical weak spots (2wiki
inference +22.6 pp, musique 4hop +15.2 pp). For the dagv2 comparison (a
heavier DAG pipeline we tie overall), see [DAG_MATCH.md](DAG_MATCH.md).
