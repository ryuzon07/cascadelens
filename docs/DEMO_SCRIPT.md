# Demo script (2 to 3 minutes)

**Status: exploratory.** Numbers are generated. Regenerate after a freeze and before recording.

## Script

**0:00, the problem.** Many alerts fire during an incident. Which service do you look at first?
CascadeLens gives a ranking of likely origin services, not proof of causation.

**0:20, why published numbers mislead.** Show the leakage figure. Repetitions of one injection are
near-duplicates, so a random split scores flat LightGBM 4.0 [0.8, 7.7] points higher than a grouped
split on the same cases.

**0:50, the harness.** One command runs splits, baselines, the ranker, simulated perturbations,
probes and calibration, and every number here is generated from a committed result file.

**1:15, live demo.** Load a sample case and read the ranking. Drop 30% of telemetry with the
failure-correlated slider, and watch the ranking and the abstain banner. Say that these are
synthetic sample cases, so this shows behaviour and not accuracy.

**2:00, the honest summary.** Show the P2 curve, labelled simulated. At 30% failure-correlated loss
the ranker scores 74.7 [67.8, 81.1] against 63.3 [55.9, 70.5] for flat LightGBM. On clean data flat LightGBM
still wins, 94.0 [90.1, 97.4] against 88.7 [83.6, 93.1]. S3 passes against learned baselines and fails
against non-learned ones. S6 passed from a fresh clone on the real data, on exploratory results.

**2:30, close.** Show the repository link [VERIFY: repository URL] and how to reproduce.

## Shot list

```bash
make smoke
```

```bash
cascadelens diagnose --case demo_data/payment-cpu.json
```

```bash
cascadelens diagnose --case demo_data/catalog-latency.json
```

```bash
cascadelens demo-figure --case demo_data/payment-silent.json --out fig.png
```

```bash
python -m cascadelens.demo
```

Figures to show: `docs/results/figures/leakage.png` and
`docs/results/figures/robustness_p2_supervised.png`.
