# NOT YET COMMITTED

This pre-registration is a draft. It takes effect only once it is committed, together with the
week-1 power analysis, before any model result exists. The commit hash is the timestamp.

# Pre-registered success criteria

| ID | Criterion | Test | Role |
| --- | --- | --- | --- |
| S1 | Ranker beats frequency prior on group\_run | Paired group-bootstrap CI of the Top-1 difference excludes zero | Primary |
| S2 | Ranker on clean data is within a few Top-1 points of the best reproduced baseline, or better | Paired difference; margin of 3 points stated in advance | Primary |
| S3 | Under 30% failure-correlated loss the ranker's margin over baselines is larger than on clean data | Paired bootstrap on the difference of differences; includes the augmentation-matched LightGBM | Primary |
| S4 | Selective Top-1 at 50% coverage is above full-coverage Top-1 and above random abstention | Paired bootstrap; AURC versus random and oracle | Primary |
| S5 | ECE after calibration is lower than before | Bootstrap CI on ECE difference; reliability diagram shown | Secondary |
| S6 | A fresh clone plus one command regenerates all headline tables | CI job on a small sample plus documented full run | Primary |
| S7 | Reproduced baselines match published numbers within a stated tolerance | Tolerance defined per method before running (default: absolute 5 Top-1 points on the same case subset and metric definition) | Primary |

**Outcome policy:** if S2 or S3 fails, report it as a finding. The harness, the leakage analysis and the negative result still stand as contributions. Overlapping-CI comparisons are not used anywhere; all comparisons are paired.
