# NOT YET COMMITTED

This pre-registration is a draft. It takes effect only once it is committed, together with the
week-1 power analysis, before any model result exists. The commit hash is the timestamp.

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
- **Assumptions:** these correlations and the 0.60 baseline are assumed, not measured.

**Claim language:** passing S2 means only that a Top-1 deficit of 12 points or more is ruled out. It
does not show that the ranker matches the baseline or is within a few points of it. The estimate and
its interval are reported either way. Failing S2 means such a deficit could not be ruled out, not
that the ranker is worse.
