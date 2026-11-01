# Pre-registration

This file binds the CascadeLens ranker from the commit that adds this sentence. The commit hash is
the timestamp. It was committed after the baseline results below, not before every model result as
first planned, and "Status when fixed" says exactly what was and was not already known.

# Pre-registered success criteria

| ID | Criterion | Test | Role |
| --- | --- | --- | --- |
| S1 | Ranker beats frequency prior on group\_run | Paired group-bootstrap CI of the Top-1 difference excludes zero | Primary |
| S2 | Ranker on clean data is non-inferior to the best reproduced baseline: its Top-1 is not worse by 12 points or more | Paired group-bootstrap on `ranker - baseline + 0.12`; the interval must lie above zero, and its p-value joins the Holm family | Primary |
| S3 | Under 30% failure-correlated loss the ranker's margin over baselines is larger than on clean data | Paired bootstrap on the difference of differences; includes the augmentation-matched LightGBM | Primary |
| S4 | Selective Top-1 at 50% coverage is above full-coverage Top-1 and above random abstention | Paired bootstrap; AURC versus random and oracle | Primary |
| S5 | ECE after calibration is lower than before | Bootstrap CI on ECE difference; reliability diagram shown | Secondary |
| S6 | A fresh clone plus one command regenerates all headline tables | CI job on a small sample plus documented full run | Primary |
| S7 | Reproduced baselines match published numbers within a stated tolerance | Tolerance defined per method before running (default: absolute 5 Top-1 points on the same case subset and metric definition) | Primary |

**Outcome policy:** if S2 or S3 fails, report it as a finding. The harness, the leakage analysis and the negative result still stand as contributions. Overlapping-CI comparisons are not used anywhere; all comparisons are paired.

# Status when fixed

**No ranker result exists.** The Stage 1 retrieval, the Stage 2 ranker, calibration and abstention
are not implemented, and no perturbation has been run, so S1, S3, S4 and S5 are blind for the
ranker.

**These results were already known**, and they are in the commit history before this file took
effect:

| Criterion | Already known | Effect |
| --- | --- | --- |
| S1 | Frequency prior Top-1 on group_run: 20.0% | The S1 comparator is known. The ranker is not |
| S2 | LightGBM flat Top-1 on group_run: 94.0%. BARO: 30.1% | The 12-point margin was set before these existed, but committed after them, so history alone cannot show that order |
| S3 | Clean scores of the plain flat baseline. No masked baseline and no perturbed score exists | The clean half of the comparison is partly known |
| S6 | Nothing | Blind |
| S7 | Reproduction is finished: BARO passes, ICA-LiNGAM RW misses on Sock Shop by 1.4 points | **S7 is not blind.** Report it as an observed result, not a pre-registered test |

S7 also departs from a blind protocol in one way. The first BARO run used `data.csv` and failed, and
the passing run reads `simple_data.csv`. The tolerance was set per method before running, on Avg@5,
because no published AC@1 exists for RE1. See `docs/reproduction/PLAN.md`, including its amendments.

Features were built before this file took effect. No threshold or feature was chosen using ranker
results, because none exist. One fix to the simple features came after seeing baseline test scores:
a flat zero baseline now falls back to a scale of 1, as RobustScaler does, because services were
tying at the z-score cap. It raised LightGBM flat on group_run from 89.0% to 94.0%.

**S2 target.** Read literally, "best reproduced baseline" means a published method, which is BARO at
30.1% on these inputs. That reading was fixed knowing LightGBM flat scores 94.0%, so it would be the
easy choice. S2 therefore uses the stricter reading: the comparator is the best baseline of any
kind at evaluation time, supervised or not. On current numbers that is LightGBM flat, so passing
needs a ranker Top-1 of about 82% or more, with the interval clearing the margin. The comparison
with BARO is reported as well, as exploratory.

# Power analysis

From `docs/power_analysis.md`: 75 groups, 375 cases, 2,000 simulations per point and 4,000
group-bootstrap resamples, with target power 80%. All correlations and the 0.60 baseline are
assumed, not measured. The analysis leaves out variation from training on different folds, so real
power is likely to be lower.

| Scenario | Baseline Top-1 | Within-group correlation | Paired correlation | MDE at alpha 0.05 (points) | MDE at alpha 0.01 (points) | Power of the S2 margin at no true difference |
| --- | --- | --- | --- | --- | --- | --- |
| s1_frequency_prior | 0.20 | 0.5 | 0.2 | 12.5 | 15.4 | 0.83 |
| best_baseline | 0.60 | 0.5 | 0.5 | 11.4 | 13.8 | 0.81 |
| best_baseline_low_clustering | 0.60 | 0.2 | 0.5 | 9.5 | 11.5 | 0.94 |
| best_baseline_high_clustering | 0.60 | 0.8 | 0.5 | 13.7 | 16.5 | 0.66 |
| best_baseline_high_pairing | 0.60 | 0.5 | 0.8 | 8.3 | 10.0 | 0.97 |

MDE is the minimum detectable Top-1 difference at 80% power. Alpha 0.01 is the first Holm step over
the five primary criteria.

**Claim language these numbers require:**

- **S1:** well powered for any useful ranker.
- **S2:** see "S2 margin" below.
- **S3:** only large effects are detectable. Scaling, not simulation, gives an MDE of about 11 to 16
  points, or 14 to 20 after Holm, so a null result means no effect of about 15 points or more. It
  does not mean no effect.
- **S4:** not covered by this analysis. A rough separate simulation suggested gains of 6 to 10
  points are detectable and gains of 3 to 5 are not, with optimistic assumptions.
- **All:** every figure assumes the 75 groups are tested across five folds. A one-shot test of 15
  groups would raise each MDE about 2.2 times, so the protocol stays cross-validated.

# S2 margin

The 12-point margin replaces the 3-point margin of PRD section 5.5. It was set from simulated power
on the 75 groups, before any model result existed, and not from what a ranker is hoped to achieve.

- **Why not 3 points:** with 75 groups the 95% interval half-width is about 6 to 10 points, so a
  ranker exactly equal to the baseline would pass a 3-point margin only 9 to 19% of the time.
- **Why 12 points:** it is the smallest margin tried (8, 10, 12, 14, 16) that reaches 80% power in
  the central scenario, which assumes a baseline Top-1 of 0.60, a within-group correlation of 0.5
  and a paired correlation of 0.5. Power there, at no true difference, is 81% at alpha 0.05.
- **Sensitivity:** at alpha 0.05, power is 66% with a within-group correlation of 0.8 and 94% with
  0.2. Under Holm, S2 is tested between alpha 0.01 and 0.05; at 0.01 the central power is 61%.
- **Assumptions:** these correlations and the 0.60 baseline are assumed, not measured. The
  central scenario predates the 94.0% baseline, and a baseline that strong may behave differently.

**Claim language:** passing S2 means only that a Top-1 deficit of 12 points or more is ruled out. It
does not show that the ranker matches the baseline or is within a few points of it. The estimate and
its interval are reported either way. Failing S2 means such a deficit could not be ruled out, not
that the ranker is worse.

# Amendments

These changes came after this file took effect. None used a ranker result, because none exists.

**2026-10-31: inferred call graphs.** "Status when fixed" says features were built before this file
took effect. That was true then. RE1 records no topology, so the graph features were empty, and a
later commit (`18f9016`) infers one graph per case from pre-incident latency. Its settings were
chosen by stability across repetitions, with no label. The criteria are unchanged. The inferred graph
is unvalidated against real call graphs, so the graph group must be ablated.

**2026-10-31: `rule_upstream` tie-break.** After seeing its test scores, `rule_upstream` was changed
to break z-score cap ties by anomaly duration (commit `d1f475a`), which raised its group_run Top-1
from 20.5% to 28.0%. It is not the best baseline, so no criterion's comparator moved.

The amendments below came after exploratory ranker results existed, so the sentence above does not
cover them. No config has been frozen. The criteria, the 12-point margin and the S3 test are
unchanged.

**2026-11-01: Stage 2 runs on one thread.** LightGBM's `bagging_by_query` gave different models
between runs at the same seed when it ran on several threads, so `configs/ranker.yaml` now sets
`num_threads: 1` for Stage 2. This was found because the plain ranker's clean Top-1 differed between
runs (92.1% and 90.7%). The setting was chosen for repeatability, not by test score: a refit of two
folds then matched its saved outcomes on 750 of 750 cases. Part 7 was regenerated, and its group_run
Top-1 moved from 90.7% to 90.6% with no verdict changed. Fold 1 remains fragile, with very early
stopping and large swings across settings.

**2026-11-01: P2 leaks the root cause.** P2 is the 30% failure-correlated loss in S3. It uses ground
truth to place gaps near the fault, and missing share alone gave 16.8 to 24.5% Top-1 against a 4.4%
floor before any ranker ran. Mask features can therefore gain from the gaps revealing the fault
rather than from robustness. On exploratory numbers, `ablate_mask` loses 3.2 [1.2, 5.2] points at P2
30%, and the mask-only probe reaches 31.5% [22.9, 40.7] on clean data. S3 is not changed, but any S3
gain must be reported beside these two numbers.

**2026-11-01: reading of "baselines" in S3.** The criterion names no baseline set and requires the
augmentation-matched LightGBM. The comparators were set in `experiments/robustness/config.yaml`
before any S3 result existed: both LightGBM flats, frequency, `rule`, `rule_upstream` and BARO. They
stay in the Holm family and are reported one by one. The criterion is not narrowed to learned
baselines after seeing that the four non-learned comparators fail: a baseline that ignores the lost
data keeps its score, so its margin shrinks whenever the ranker loses accuracy. S3 is reported as
split, not as a pass.
