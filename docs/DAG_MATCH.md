# dag_match: our full stack vs a DAG-decomposition pipeline (dagv2)

**A mostly-negative, fully quantified benchmark line — recorded as-is.** Our
stack (NV-Embed-v2 dense + 4B fusion rerank + iterative assembly + 27B reader)
was benchmarked against dagv2 (planner decomposition + grounded multi-channel
node retrieval + node-chain answer injection + source-cited panel curation) on
three datasets' clean subsets. **Final score: 1 win, 4 losses.** Summary JSONs:
[`results/dag_match/`](../results/dag_match/) (per-battle `dag_metrics*.json`,
tau calibration records, and `contamination_audit.json`).

## Protocol (pinned down empirically before scoring)

- Per dataset, the same reader (27B, temperature 0, fixed seed, same prompt),
  the same embeddings, and per-dataset corpora; the **clean subset** = the
  semantic-hash non-train split (seed 20260922): musique 293, hotpotqa 299,
  2wiki 306 questions.
- dagv2's per-question scores were **recomputed from its archived
  request/response records** on the identical clean subsets; EM/F1
  normalization re-derived and verified equal (80/80 rows match); R@k/All@k =
  exact per-hop passage recall (100/100 verified); our score model is a
  round-invariant (qid, variant, docid) so cached scores replay exactly.
- Gate τ calibrated only on each clean-98 calibration subset (answered-set
  precision ≥ 0.90). Paired bootstrap: 5000 resamples per comparison.
- **Contamination disclosure:** dagv2 trains nothing, so clean subsets are
  unbiased for it. Our full-1000 auxiliary tables include our own training
  questions (musique 707/1000, hotpotqa 701/1000, 2wiki 694/1000) and are
  marked optimistic, coarse reference only — headline numbers are always the
  clean subsets.

## Scoreboard (clean subsets, EM vs dagv2)

| Dataset (n) | dagv2 | Our best arm | ΔEM [95% CI] | Result |
|---|---:|---:|---|---|
| musique (293) | 56.66 | 49.49 (fixed2_chain) | −7.17 [−12.3, −2.0] | **loss** |
| hotpotqa (299) | 63.21 | 66.56 (single_k20) | +3.34 [−1.0, +7.7] | **win (directional — CI crosses 0)** |
| 2wiki (306) | 73.53 | 68.63 (fixed3_k20) | −4.90 [−8.8, −1.0] | **loss** |
| 2wiki decfirst probe (306) | 73.53 | 61.44 (nochain) | −12.09 [−16.7, −7.5] | loss (mechanism probe) |
| 2wiki curated-panel probe (306) | 73.53 | 65.69 | −7.84 [−11.8, −3.9] | loss (mechanism probe) |

The hotpotqa win is real in direction but **not statistically settled** (all
arms +3.0–3.4, CIs cross 0): dense R@5 is already 94.8% there, iteration adds
nothing (single = fixed2 = fixed3), and the edge comes from the reader side
with evidence completeness ≈ 98% on both sides.

## Four-component attribution (isolated, per-component probes)

Each dagv2 component was ablated into our stack one at a time on frozen
trajectories:

| Component probe | Net EM contribution | Verdict |
|---|---:|---|
| Budget alignment (reader sees top-20 like dagv2) | **+0.3 pp** (fixed2 48.81 → 49.15) | budget only binds at tiny evidence (single 5→20 segments: +5.5); our k20 arms' evidence completeness (All@20 87.4–88.7%) already *exceeds* dagv2's 77.5% |
| Node-chain injection (intermediate answers into reader context) | **±0** (+0.34, CI crosses 0) | oracle-answer chain upper bound 54.95 still < dagv2 56.66 — the mechanism itself is not the gap; 2hop +6.0 / 4hop −8.7 cancel out |
| Upfront decomposition + multi-channel union retrieval | **−7.2 pp** (decfirst_nochain 61.44 vs fixed3_k20 68.63) | negative asset alone: upfront subqueries carry unresolved references ("the performer of X") that break retrieval; dagv2's nodes are *interleaved grounded* (each node retrieves with parent answers already resolved). Evidence-guided iterative subqueries ("what is missing") beat upfront decomposition |
| Source-priority panel curation | **−2.9 pp** (curpanel_nochain, CI excludes 0) | our proxy (title/answer-string mention) pins noise onto the panel (All@20 93.8% < 96.7% uncurated); dagv2 pins *explicitly cited* sources (boolean commitments from node answers) |

**Conclusion of the attribution:** dagv2's four mechanisms — decomposition ×
interleaved grounding × commitment-based citation curation × chain injection —
only work as a joint structure. No subset of them closes the gap; the combined
shortfall is ~5–12 pp EM in retrieval-limited multi-entity/deep-chain
territory.

## Capability boundary (the takeaway)

- **Where we win / tie:** retrieval-reachable questions (hotpotqa-type, dense
  R@5 > 94%) — we match or beat the heavy DAG pipeline at roughly **half the
  token cost** (~10k vs 12.9k LLM tokens/question); plus the refusal/gating
  and cost-sensitive scenarios from the gate reports (hallucination 32.7% →
  1%, reader tokens −84–94%).
- **Where dagv2's wall stands:** retrieval-limited, multi-entity-grounding
  multi-hop (2wiki bridge_comparison/inference, musique deep chains). Its
  planner decomposition + per-node grounded union retrieval is worth ~19 pp
  R@5 on 2wiki, and the reader-side structure a further ~5–7 pp EM.
- **Minimum credible change to enter that region** (not a subset of the four
  components): a node-answering loop with explicit citation commitments —
  nodes emit boolean source vectors, children inherit pinned sources, and
  queries are grounded in parent answers. That is dagv2's core design; the
  gap is now quantified, and single-point patches cannot close it.
- Also measured: our probes run 3 rounds unconditionally (offline arm
  arbitration); the online gated arm averages 1.51–1.56 rounds.

## What we did *not* do

- No hotpotqa/2wiki benchmark was started until musique was resolved; the
  first musique battle's failure gated them per protocol.
- No dagv2 component was judged from its paper claims — each was re-measured
  as an isolated probe on our frozen trajectories.
- No result above uses test questions for any selection; τ and configs come
  from calibration splits only.
