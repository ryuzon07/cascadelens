# Pre-registered criteria S1 to S7

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
| S6 | Fresh clone plus one command regenerates all tables on the real data | no completed full run logged | - | not demonstrated |
| S7 baro (observed) | Reproduced Avg@5 within tolerance on all three suites | see the reproduction table | - | pass |
| S7 lingam_randomwalk (observed) | Reproduced Avg@5 within tolerance on all three suites | see the reproduction table | - | fail |
