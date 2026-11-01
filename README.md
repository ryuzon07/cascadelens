# CascadeLens

CascadeLens ranks the likely origin services of a microservice incident from metrics, and is built
to stay honest when telemetry is imperfect. It gives a ranking of likely origin services, not proof
of causation, and it is a decision aid for read-only analysis, not an autonomous operator.

**Status:** the harness, the two-stage ranker, the perturbation engine, missingness-aware training
and the robustness sweep are done. Calibration and abstention are not built yet, so S4 and S5 are
untested. No config is frozen, so every ranker result here is exploratory. Success criteria are
fixed in [PREREGISTRATION.md](PREREGISTRATION.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/), make and Python 3.11.

```bash
make install
make lint typecheck test
make smoke
```

`make smoke` runs the pipeline end to end on a small checked-in fixture, with no downloads.

LightGBM is pinned below 4.7 because 4.7.0 crashes on Windows machines with an older Visual C++
runtime.

## Data

The project uses RE1 from [RCAEval](https://github.com/phamquiluan/RCAEval): 375 metrics-only fault
injection cases across Online Boutique, Sock Shop and Train Ticket. Data is not redistributed here.
Download it yourself, accept its terms, and place the suites at:

```text
data/raw/RE1-OB/<service>_<fault>/<n>/data.csv
data/raw/RE1-SS/...
data/raw/RE1-TT/...
```

## Reproduce

```bash
make all
```

This rebuilds the processed data, splits, features and every clean table. `PUBLISHED=1 make all`
also reruns BARO and ICA-LiNGAM, which takes about 2.5 hours and needs `make install` for the
`published` extra. `ROBUSTNESS=1 make all` also runs the perturbation sweep and the probes, which
takes about three hours and resumes where it stopped. `FROZEN=1` drops `--exploratory` after a
freeze.

Results are committed under `experiments/`. Each result JSON records its config hash, git commit,
data checksum and seed.

## Current results

Scores on the `group_run` split, with 95% group-bootstrap intervals. Supervised and unsupervised
methods should only be compared descriptively.

| Method | Type | Top-1 % |
| --- | --- | --- |
| LightGBM flat | Supervised | 94.0 [90.1, 97.4] |
| Two-stage ranker (exploratory) | Supervised | 90.6 [85.8, 94.8] |
| Frequency prior | Supervised | 20.0 [12.0, 29.3] |
| BARO | Unsupervised | 30.1 [21.9, 38.4] |
| Max anomaly (`rule_upstream`) | Unsupervised | 28.0 [20.5, 35.7] |
| ICA-LiNGAM random walk | Unsupervised | 8.8 [3.7, 14.9] |
| Earliest onset (`rule`) | Unsupervised | 6.1 [3.5, 9.1] |

- A random split inflates LightGBM flat by 4.0 points, 95% CI [0.8, 7.7]:
  [leakage demo](experiments/leakage_demo/note.md).
- BARO reproduces its published numbers, and ICA-LiNGAM misses on one system:
  [reproduction](docs/reproduction/PLAN.md).
- Known leak risk: the injection sits near the middle of every window, so the window itself reveals
  it (see the [data audit](docs/data_audit.md)).
- The ranker with masking augmentation scores 88.7 on clean data. Under simulated 30%
  failure-correlated loss it scores 74.7, against 63.3 for LightGBM flat. S1 and S2 pass, and S3
  is split: it passes against both LightGBM flats and fails against the four non-learned baselines.
  See the [robustness summary](docs/robustness_summary.md), including its leak caveat.

## Freezing the config

Every result is tagged exploratory until a config is frozen. Run this once, after you have stopped
changing `configs/ranker.yaml`, `configs/augment.yaml` and the feature config, and before the final
reported run:

```bash
cascadelens freeze --name final
```

It records each ranker variant's validation choice for every fold under the config hash, in
`configs/frozen/final.yaml`, using training and validation cases only. Then rerun the experiments
without `--exploratory`. A frozen record is never rewritten, and any later config change needs a new
name and a disclosed amendment in [PREREGISTRATION.md](PREREGISTRATION.md).

## Layout

```text
configs/      experiment settings
src/          ingestion, splits, baselines, features, evaluation
experiments/  configs and committed results
docs/         PRD, data audit, power analysis, reproduction reports
scripts/      smoke run and full reproduction
```

## License

MIT. Dataset terms are separate.
