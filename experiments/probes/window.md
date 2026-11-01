# Window-leak probe (exploratory)

Pass condition (PRD 6.5): accuracy falls smoothly with offset; a cliff at zero means the window encodes `inject_time`. Windows are cropped before features. Simulated.

Window bounds are computed from the recording, without reading `inject_time`, but the recordings were centred on the injection: median position 0.500, and 98.9% of cases within 2% of the middle. So the definition has no direct dependence and a strong indirect one.

| Method | Clean | Cliff | P4:offset_seconds=-120 | P4:offset_seconds=-90 | P4:offset_seconds=-60 | P4:offset_seconds=-45 | P4:offset_seconds=-30 | P4:offset_seconds=-15 | P4:offset_seconds=15 | P4:offset_seconds=30 | P4:offset_seconds=45 | P4:offset_seconds=60 | P4:offset_seconds=90 | P4:offset_seconds=120 | P5:max_jitter_seconds=60 | P5:max_jitter_seconds=120 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ranker | 90.6 | False | 72.5 | 76.9 | 88.7 | 89.5 | 89.5 | 90.8 | 90.2 | 90.3 | 90.0 | 90.0 | 90.6 | 90.6 | 90.4 | 89.1 |
| ranker_augmented | 88.7 | False | 77.9 | 81.3 | 88.3 | 89.3 | 89.1 | 89.5 | 89.3 | 89.7 | 88.2 | 88.6 | 88.6 | 89.3 | 89.2 | 87.9 |
| lightgbm_flat | 94.0 | False | 61.3 | 68.3 | 90.6 | 93.1 | 93.9 | 93.3 | 93.9 | 94.3 | 94.3 | 93.7 | 94.1 | 93.2 | 93.8 | 93.3 |

**Passed.**
