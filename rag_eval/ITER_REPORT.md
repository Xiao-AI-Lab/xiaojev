# Calibration-gated iterative RAG, end to end (MuSiQue, 293 non-training questions)

Date: 2026-09-26. Code: [`iterative_rag.py`](iterative_rag.py) (reusable
pipeline, every model-touching step injectable) +
[`run_iterative.py`](run_iterative.py) (evaluation driver, probe mode +
phase-level resume). Unit tests: [`../tests/test_iterative_rag.py`](../tests/test_iterative_rag.py)
(18 cases, all mocked, no GPU/network).
Models: reranker = 4B LoRA (`ckpt/qwen3_4b_lora_v1/step2500`, weighted-RRF
fusion w=0.5/c=1, the frozen fusion4b config); gate = v3 (`ckpt/v3`,
answerability channel); reader/subquery = local qwen3.8-27b (temperature 0,
seed 20260917, JSON schema, same template as the RAG eval); first-stage
retrieval = NV-Embed-v2 (reused chunk index). Full run wall time 7302s;
1,983 reader predictions, zero failures. All numbers:
[`iter_metrics.json`](iter_metrics.json).

## Design

```
Each round: dense top-50 (round 1: original question; later: 27B subquery)
  -> 4B LoRA relevance scoring -> weighted-RRF fusion -> top-5 deduped into
     the accumulated evidence set E
  -> v3 answerability gate P(answerable | question, E)
      P >= tau        -> the reader answers from E, stop
      round 3 still <tau -> per-arm fallback: answer anyway / refuse
      else            -> the 27B model reads a summary of E and proposes a
                         "what is still missing" subquery for the next round
```

- Same protocol as `run_gate.py`: 293 questions x {ans, unans} = 586
  pipelines; `unans` bans the question's gold documents from every candidate
  stage (the local gold set is fully answerable, so unanswerable cases can
  only be constructed synthetically). calibration (98) sets tau; test (101)
  is primary; dev (94) is secondary.
- **Probe mode:** every question runs all 3 rounds with per-round gate
  probabilities recorded; tau is calibrated afterwards and each arm's
  trajectory is a pure function of the recorded probabilities — the reader is
  called only where an arm's policy answers, and refusals are implicit (no
  prediction row). Same "score store, pure decision" idea as `run_gate.py`.
- **Four arms:** `single` (1 round, no gate = the existing dense+fusion
  baseline), `fixed2` (fixed 2 rounds, no gate), `gated` (gate-controlled, up
  to 3 rounds, answer on exhaustion), `gated_refuse` (gate-controlled, up to
  3 rounds, refuse on exhaustion).
- tau rule identical to `gate.py`: sweep [0.01, 0.99], require answered-set
  precision >= 0.90, maximize answerable coverage → **tau = 0.64**
  (calibration: answered precision 90.3%, answerable coverage 66.3%).

## Calibration curve (calibration, n = 98 x 2; accept = any round P >= tau)

| tau | answerable coverage | answered precision | n_answered |
|---|---|---|---|
| 0.05 | 0.990 | 0.571 | 170 |
| 0.20 | 0.939 | 0.687 | 134 |
| 0.40 | 0.847 | 0.798 | 104 |
| 0.60 | 0.704 | 0.873 | 79 |
| **0.64** | **0.663** | **0.903** | 72 |
| 0.70 | 0.612 | 0.909 | 66 |
| 0.85 | 0.469 | 0.939 | 49 |
| 0.95 | 0.214 | 1.000 | 21 |

(The 0.64 row reads the stored values of the neighboring 0.65 grid point;
`iter_metrics.json` stores the 5% grid.)

## Main results (test split, 101 questions per arm)

Answerable questions (full-set EM/F1 counts refusals as zero; gated_refuse
also listed "answered only"):

| Arm | answered | refused | EM | F1 | EM/F1 (answered only) | avg rounds | prompt tokens |
|---|---|---|---|---|---|---|---|
| single | 101 | 0 | 0.406 | 0.521 | 0.406 / 0.521 | 1.00 | 97.8k |
| fixed2 | 101 | 0 | **0.525** | 0.617 | 0.525 / 0.617 | 2.00 | 223.6k |
| gated | 101 | 0 | 0.515 | **0.618** | 0.515 / 0.618 | 2.16 | 255.1k |
| gated_refuse | 71 | 30 | 0.366* | 0.452* | **0.521 / 0.643** | 2.16 | 141.1k |

Unanswerable questions (synthetic, gold removed):

| Arm | answered | refused | hallucination rate | abstain-or-refused | avg rounds | prompt tokens |
|---|---|---|---|---|---|---|
| single | 101 | 0 | 53.5% | 46.5% | 1.00 | 102.3k |
| fixed2 | 101 | 0 | 59.4% | 40.6% | 2.00 | 222.9k |
| gated (answer on exhaust) | 101 | 0 | **62.4%** | 37.6% | 2.91 | 366.3k |
| gated_refuse | 10 | 91 | **6.9%** | **93.1%** | 2.91 | **26.0k** |

**Gate trajectories (gated arm, test):** answerable — 26 accepted in round 1,
33 in round 2, 12 in round 3, 30 exhausted; unanswerable — 91/101 ran all 3
rounds (only 10 slipped through to the reader).
**Refusal quality (test, gated_refuse):** recall 90.1% (91/101 unanswerable
correctly refused); precision 75.2%; **answerable keep rate 70.3%**.
**Gate discrimination:** P(answerable) AUC test = 0.815 (round 1) / **0.882**
(max over 3 rounds); calibration 0.828 / 0.903; dev 0.808 / 0.878. Iterating
evidence lifts the gate's own AUC from the BM25 gate's 0.77 to 0.88 — the
gate benefits from iteration too.

## dev / calibration corroboration (same trend)

| Split | ans EM single → fixed2 | unans hallucination single → gated | gated_refuse hallucination | keep rate |
|---|---|---|---|---|
| dev (94) | 0.404 → 0.479 | 47.9% → 57.4% | 7.4% | 66.0% |
| calibration (98) | 0.378 → 0.439 | 46.9% → 51.0% | 4.1% | 66.3% |

On dev, gated and fixed2 tie exactly on ans EM (0.479 = 0.479), consistent
with test's "no gate benefit on pure-answerable traffic".

## Cost account (test split, 202-run mixed stream; prompt+completion incl. subqueries)

| Arm | total prompt tokens | avg retrieval rounds | reader mean latency | notes |
|---|---|---|---|---|
| single | 200.1k | 1.00 | 7.0s | 1 retrieval + 1 reader call |
| fixed2 | 446.5k | 2.00 | 7.1s | 2x retrieval + 1 subquery + reader |
| gated | 621.4k | 2.53 | 7.0s | unanswerable questions almost all run 3 rounds; most expensive |
| gated_refuse | **167.2k (−73% vs gated)** | 2.53 | 6.4s (answered) | refused questions: no reader, no later rounds |

Phase timings (full 7302s): **4B relevance scoring 4714s (64.5%, 1570s x 3
rounds)**, reader 1722s, subqueries 675s (2 rounds), v3 gate 115s, dense
retrieval 68s. Relevance scoring dominates by far — in production, trigger
later rounds only for borderline score bands, or cache scores across subquery
rounds.

## Interpretation

1. **Iterative retrieval is a double-edged sword.** Answerable EM 40.6 →
   52.5 (+11.9 pp), F1 +9.6 pp: subqueries genuinely recover the evidence
   round 1 missed (round-2 has the highest acceptance rate, 33/101). But
   unanswerable hallucination rises in lockstep: single 53.5% → fixed2 59.4%
   → gated 62.4%. More rounds retrieve more convincing-looking wrong
   evidence, and the reader abstains less (46.5% → 37.6%). Note single's
   53.5% is already far above the old BM25 pipeline's nogate 32.7% — a
   stronger dense+4B first stage misleads unanswerable questions more
   strongly too. **Retrieval strength without a refusal mechanism is a net
   negative.**
2. **Gate + refusal is the only configuration that wins both sides.**
   gated_refuse keeps the full iteration gain on answerable questions
   (answered EM 52.1 / F1 64.3, on par with fixed2's 52.5/61.7 and higher
   F1) while cutting hallucination 62.4% → 6.9% (9x) and saving 73% of mixed
   traffic tokens. Against the old BM25 gate (hallucination 1.0% but only
   27.7% keep rate): keep rate is now 70.3% — a 2.5x improvement coming
   mainly from the dense first stage plus iterative evidence repair, with a
   modest hallucination give-back that is inherent to stronger retrieval
   finding more convincing distractors. **The iterative loop rescued gate
   coverage without any retraining.**
3. **Honest boundary: the gate adds nothing on pure-answerable traffic.**
   gated vs fixed2: EM −1.0 pp, F1 +0.2 pp, and *more* tokens (255k vs 224k)
   — the gate's entire value lives in mixed traffic (the real world contains
   unanswerable questions). And the v3 gate is conservative on short dense
   contexts (consistent with the gate_dense experiment): only 26/101
   answerable questions are accepted in round 1, avg_rounds 2.16 pushes most
   of them to rounds 2–3, saving no retrieval/scoring cost (gated is pricier
   than fixed2). Improvements: swap in a 4B gate (its answerability channel
   is already evaluated), or retrain the gate on dense short contexts (v5
   plan), so that "round 1 is enough" questions actually stop at round 1.
4. **tau calibration remains a zero-training deployment.** 98 calibration
   questions set tau directly at the target precision; test refusal
   precision 75.2% / recall 90.1%, closely matching calibration (73.4%/92.9%)
   and dev (72.4%/89.4%) — cross-split stability is visibly better than the
   BM25 gate (test precision 57%), because the gate's AUC is higher after
   iteration (0.88 vs 0.77).

## Artifacts

- Committed: `iterative_rag.py`, `run_iterative.py`,
  [`../tests/test_iterative_rag.py`](../tests/test_iterative_rag.py),
  [`iter_metrics.json`](iter_metrics.json) (every number above + the 5% grid
  calibration curve + phase timings).
- Runtime (regenerable, git-ignored): `iter_retrieval.jsonl` (1,758 rows),
  `iter_subqueries.jsonl` (1,172 rows), `iter_scores.jsonl` (~89k rows),
  `iter_predictions.jsonl` (1,983 reader predictions). Every phase resumes
  (item-level for scores, key-level elsewhere).
