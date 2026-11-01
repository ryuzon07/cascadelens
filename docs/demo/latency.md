# Latency

Target: under 2 s for one case end to end, feature extraction included, on a laptop CPU.

Measured by `python -m cascadelens.demo.latency` at commit `a6977b57c5b7` on Windows-10-10.0.26200-SP0, Python 3.11.18.

| Input | Runs | Median (s) | p95 (s) | Against target |
| --- | --- | --- | --- | --- |
| Synthetic samples (1,800 rows) | 100 | 0.047 | 0.070 | meets |
| Benchmark cases, Parquet, local only (40 cases) | 40 | 0.175 | 0.600 | meets |

Loading the model takes 0.079 s and happens once at API startup, so it is not in the table. A single CLI call pays it, plus Python's own startup.
