# Model card: CascadeLens ranker

**Status: exploratory.** No config is frozen. Numbers come from committed result JSONs, traced in
[provenance](docs/results/provenance.md).

## Intended use

A read-only decision aid that gives engineers a ranking of likely origin services, not proof of causation, from metrics of a single incident
window, with a confidence and an abstention flag.

## Out of scope

- Not an autonomous operator. It must not trigger remediation.
- Not proof of causation, and not causal discovery.
- Not validated on production data. Results come from injected benchmark faults.
- Not a model of changing topology.
- Not an internet-facing service. The API has no authentication.

## Model

A LightGBM retrieval stage followed by a LightGBM LambdaRank reranker over per-service features:
signal, temporal, graph, structural and mask. The CascadeLens ranker (`ranker_augmented`) also
trains on copies with simulated telemetry loss. Service names are not features.

## Training data

RE1 from RCAEval: 375 metrics-only fault-injection cases across Online Boutique, Sock Shop
and Train Ticket. Splits keep each injection run and its repetitions together. RE1 records no
topology, so graphs are inferred from pre-incident latency and have not been checked against real
call graphs.

The `diagnose` command and API serve a model saved for one training fold, the plain two-stage ranker
by default. They do not serve a model trained on all data.

## Metrics

Clean group_run Top-1 for `ranker_augmented`: 88.7 [83.6, 93.1]. Plain ranker: 90.6 [85.8, 94.8]. Flat
LightGBM, the best baseline: 94.0 [90.1, 97.4].

<!-- generated:headline; regenerate with: python -m cascadelens.report build -->

| Criterion | Measure | Result [95% CI] | Holm p | Verdict |
| --- | --- | --- | --- | --- |
| S1 | Ranker minus frequency prior, Top-1 points, group_run | 68.7 [58.8, 77.9] | 0.0016 | pass |
| S2 | Ranker minus lightgbm_flat, plus the 12-point margin | 6.7 [2.8, 10.2] | 0.0016 | pass |
| S3 vs lightgbm_flat | Margin growth, P2 at 30% (simulated) | 16.7 [11.0, 22.5] | 0.0016 | pass |
| S3 vs lightgbm_flat_masking | Margin growth, P2 at 30% (simulated) | 14.8 [9.0, 20.8] | 0.0016 | pass |
| S3 vs frequency | Margin growth, P2 at 30% (simulated) | -14.0 [-19.0, -9.4] | 0.0016 | fail |
| S3 vs rule | Margin growth, P2 at 30% (simulated) | -12.7 [-18.1, -7.7] | 0.0016 | fail |
| S3 vs rule_upstream | Margin growth, P2 at 30% (simulated) | -11.4 [-16.9, -6.1] | 0.0016 | fail |
| S3 vs baro | Margin growth, P2 at 30% (simulated) | -18.0 [-24.1, -12.2] | 0.0016 | fail |
| S4 | Top-1 at 50% coverage minus full coverage and random abstention | 11.0 [7.0, 15.2] | 0.0018 | pass |
| S5 (secondary) | ECE after minus ECE before calibration | -0.187 [-0.240, -0.118] | - | pass |
| S6 | Fresh clone plus one command regenerates all tables on the real data | 2026-11-01 full | - | pass |
| S7 baro (observed) | Reproduced Avg@5 within tolerance on all three suites | see the reproduction table | - | pass |
| S7 lingam_randomwalk (observed) | Reproduced Avg@5 within tolerance on all three suites | see the reproduction table | - | fail |

<!-- /generated -->

<!-- generated:clean_supervised; regenerate with: python -m cascadelens.report build -->

| Method | Top-1 | Top-3 | Avg@5 |
| --- | --- | --- | --- |
| frequency | 20.0 [12.0, 29.3] | 60.0 [48.0, 70.7] | 60.0 [53.6, 66.7] |
| lightgbm_flat | 94.0 [90.1, 97.4] | 98.6 [97.0, 99.7] | 97.7 [96.0, 99.1] |
| two_stage_ranker | 90.6 [85.8, 94.8] | 97.5 [95.7, 99.0] | 96.1 [93.9, 98.0] |
| single_stage_ranker | 91.2 [86.7, 95.0] | 97.0 [94.4, 99.1] | 96.2 [93.7, 98.2] |

<!-- /generated -->

Robustness, simulated: at 30% failure-correlated loss `ranker_augmented` scores 74.7 [67.8, 81.1], and
flat LightGBM scores 63.3 [55.9, 70.5].

Held-out:

<!-- generated:failure_system; regenerate with: python -m cascadelens.report build -->

| System | Random floor | two_stage_ranker Top-1 | lightgbm_flat Top-1 |
| --- | --- | --- | --- |
| online-boutique | 7.5 | 79.4 [68.8, 88.6] | 41.8 [27.7, 56.5] |
| sock-shop | 4.4 | 96.6 [92.2, 99.5] | 89.4 [79.2, 97.4] |
| train-ticket | 1.4 | 73.8 [62.6, 84.3] | 87.7 [79.2, 94.6] |

<!-- /generated -->

## Calibration

Temperature scaling or isotonic regression, chosen per fold on validation only. ECE on clean
group_run falls from 0.219 [0.177, 0.260] to 0.032 [0.014, 0.065], and Brier from 0.158 [0.131, 0.185] to
0.072 [0.046, 0.101]. At 50% coverage Top-1 is 99.7 [98.7, 100.0], against 88.7 [83.6, 93.1] at full coverage.

<!-- generated:coverage_drift; regenerate with: python -m cascadelens.report build -->

| Slice | Label | Test coverage % | Range across folds |
| --- | --- | --- | --- |
| P2:after_boost=3.0,near_boost=3.0,rate=0.3 | simulated | 31.3 | 20.8 to 47.7 |
| P4:offset_seconds=-120 | simulated | 30.7 | 23.7 to 47.2 |
| P4:offset_seconds=-30 | simulated | 45.8 | 38.1 to 60.3 |
| P4:offset_seconds=-60 | simulated | 45.9 | 36.3 to 60.5 |
| P4:offset_seconds=120 | simulated | 46.2 | 39.2 to 61.9 |
| P4:offset_seconds=30 | simulated | 47.6 | 40.8 to 61.9 |
| P4:offset_seconds=60 | simulated | 46.5 | 37.6 to 63.2 |
| clean | clean | 48.4 | 38.9 to 60.8 |
| lofo | held-out | 45.6 | 14.4 to 84.8 |
| loso | held-out | 46.8 | 14.6 to 89.9 |

<!-- /generated -->

## Known failure modes

- Abstention coverage is unstable on unseen fault types and systems, so a 50% target may give far
  more or far less.
- Simulated P2 loss and a 120 s early window shift cut coverage to 31.3% and
  30.7%, and raise ECE to 0.112 [0.069, 0.165] and 0.070 [0.037, 0.116].
- Folds with no validation errors give the calibrator nothing to learn from, so confidence
  saturates. A confidence of 1.00 is not certainty.
- Retrieval can miss the true service, which then cannot be ranked first. Recall at the chosen K is
  99.5% on group_run and 96.0% on LOSO.
- The mask-only probe reaches 31.5 [22.9, 40.7] on clean data, against a floor of 4.4, so
  missingness may reflect collection artifacts.
- Simulated P2 gaps depend on ground truth, and mask features gain 3.2 [1.2, 5.2] points there, which
  may be the gaps revealing the fault rather than robustness.

## Ethical considerations

A wrong ranking can send responders the wrong way during an incident, so a person must check the
evidence. The tool makes no outbound request and changes nothing. It is for defensive use. The
dataset is not redistributed; its terms, MIT on RCAEval and CC BY 4.0 on Zenodo, are in the README.
