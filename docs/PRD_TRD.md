# CascadeLens: Full Project Documentation

Oct 31, 2026 · @useless\_crew

CascadeLens is a 14-week solo research-engineering project that measures how root-cause analysis (RCA) methods for microservices degrade under realistic data problems, and trains a ranker that degrades less. This document merges your PRD, TRD and master prompt and applies the fixes from the earlier review (power analysis, leakage probes, primary versus exploratory hypotheses, frozen test protocol). Items marked **[VERIFY]** come from your source notes and must be checked against primary sources before they appear in a README or paper.

## 1. Executive summary

- **What it is:** an open evaluation harness plus a missingness-aware two-stage ranker (candidate retrieval, then LightGBM LambdaRank) that ranks which service is the origin of a cascading failure.
- **Why it matters:** published RCA methods are mostly scored on clean, complete telemetry with a known incident time. Production has gaps, noisy detection windows and imperfect topology. There is no standard open way to measure the damage.
- **Main contributions, in priority order:** (1) a leakage-free grouped evaluation protocol with a demonstration of how much random splits inflate scores; (2) a seeded perturbation suite; (3) faithful reproductions of at least two published baselines; (4) a ranker with calibrated confidence and abstention; (5) code, splits and a one-command reproduction.
- **Honest framing:** the harness and the leakage analysis are the durable contribution. The ranker may or may not beat baselines at this data scale, and either result is reportable.
- **Compute and scope:** metrics-only first, laptop (16 GB RAM) or free Colab/Kaggle, about 10-15 hours per week.
- **Not claimed:** causal discovery, remediation, or production readiness. Outputs are rankings of likely origin services, not proof of causation.

## 2. Problem statement

When a microservice system fails, symptoms surface in services far from the true origin, and on-call engineers face many simultaneous alerts. They must choose where to look first. Three gaps make published results hard to trust for that decision:

1. **Optimistic evaluation.** Near-duplicate cases from the same injection run land in both train and test, inflating scores.
2. **Clean-data assumption.** Benchmarks give full telemetry and the exact incident time. Production telemetry is partially missing, and often the telemetry of the failing service is the most likely to be missing.
3. **No degradation measurement.** Papers rarely report how accuracy falls as inputs get worse, so methods cannot be compared on robustness.

## 3. Goals and non-goals

**Goals**

- G1. Build a leakage-free evaluation protocol with group-aware splits and an automated leakage report.
- G2. Reproduce at least two published baselines within a pre-stated tolerance.
- G3. Train a two-stage ranker that uses missingness as evidence and compare it with a single-stage ranker.
- G4. Quantify robustness under a fixed perturbation suite, with group-bootstrap confidence intervals.
- G5. Provide calibrated confidence and an abstention rule, evaluated with risk-coverage curves.
- G6. Release code, split definitions, configs and a one-command reproduction script.

**Non-goals**

- Automatic remediation or any code path that changes infrastructure.
- Claims of true causal discovery.
- Private or production incident data.
- LLM-written remediation advice (an LLM-agent RCA baseline is a stretch comparison only).
- Stage E propagation forecasting (stretch only).

## 4. Research questions and hypotheses

| ID | Question | Status | Pre-stated prediction |
| --- | --- | --- | --- |
| RQ1 | How much do random splits inflate RCA accuracy versus grouped splits? | **Primary** | Random splits inflate Top-1 for every supervised method; the gap is largest for the flat LightGBM baseline. |
| RQ2 | How does each baseline degrade under random versus failure-correlated telemetry loss? | Exploratory | Failure-correlated loss hurts onset-based and anomaly-score baselines more than random loss at equal rate. |
| RQ3 | Does a missingness-aware two-stage ranker degrade less than baselines, including a baseline trained with the same masking? | **Primary** | Margin over baselines grows under 30% failure-correlated loss, but may shrink once the augmentation-matched baseline is included. |
| RQ4 | Does a GNN beat engineered graph features at this data scale? | Exploratory | No; with a few hundred cases engineered features match or beat the GNN. |
| RQ5 | Does calibrated confidence support useful abstention? | **Primary** | Selective accuracy at 50% coverage exceeds both full-coverage accuracy and random abstention. |

Only RQ1, RQ3 and RQ5 are confirmatory. Everything else is labelled exploratory in the report, which keeps multiple-comparison correction manageable.

## 5. Product requirements (PRD)

### 5.1 Users

| Persona | Need | How CascadeLens serves them |
| --- | --- | --- |
| Hiring manager / reviewer | Evidence of rigor, ML and systems skill | Reproducible report, honest negative results, clean repo and CI |
| Researcher | Fair comparison of RCA methods under imperfect telemetry | Perturbation harness, fixed splits, baseline suite |
| SRE / on-call engineer (illustrative) | Shortlist of suspect services with evidence and confidence | Demo with ranked list, dependency graph, abstain flag |
| Author (you) | A defensible portfolio artifact | Pre-registered criteria, write-up, demo video |

### 5.2 Scope

**MVP (must finish)**

- Metrics-based RCA on public benchmarks, starting with the RCAEval metrics subset **[VERIFY suites, case counts, fault types]**.
- Group-aware splits and a leakage demonstration.
- Baselines: frequency prior, rule-based, BARO via RCAEval, one more published method, flat supervised LightGBM, and an augmentation-matched LightGBM.
- Perturbation suite P1-P10.
- Two-stage ranker, single-stage ranker ablation, calibration and abstention.
- CLI and a static graph figure; small API as the *should* item.

**Stretch (only after success criteria are evaluated)**

- Logs and traces modalities.
- GNN ablation.
- Second independent benchmark (the harder fault-propagation benchmark, 1,430 cases per your source) **[VERIFY existence, license, schema]**.
- LLM-agent baseline such as OpenRCA-style, or an explicit written exclusion with a reason.
- Stage E propagation forecasting.

### 5.3 Functional requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-1 | Ingest benchmark cases into a unified Case schema with validation | Must |
| FR-2 | Generate group-aware splits and verify zero group overlap | Must |
| FR-3 | Run any baseline through a common interface returning ranked services | Must |
| FR-4 | Apply a configurable perturbation to any case, deterministic by seed | Must |
| FR-5 | Train the two-stage ranker and save versioned artifacts | Must |
| FR-6 | Compute Top-1, Top-3, Avg@5 with group-bootstrap CIs | Must |
| FR-7 | Compute ECE, Brier, risk-coverage and AURC | Must |
| FR-8 | One command reproduces all headline tables and figures | Must |
| FR-9 | Leakage probes: identity-feature probe and window-leak probe | Must |
| FR-10 | API endpoint returns ranking, scores, evidence features, abstain flag | Should |
| FR-11 | Demo shows graph with suspects highlighted | Should |
| FR-12 | Logs and traces features | Could |
| FR-13 | GNN ablation | Could |

### 5.4 Non-functional requirements

| Area | Requirement |
| --- | --- |
| Reproducibility | Fixed seeds, pinned dependencies, data checksums, Docker image |
| Performance | Single-case end-to-end inference (including feature extraction) under 2 s on a laptop CPU, metrics only |
| Compute | Full MVP on a 16 GB laptop or free Colab/Kaggle |
| Traceability | Every reported number traceable to a config hash and git commit |
| Safety | Read-only analysis; no network calls to production systems; every output carries the disclaimer "ranking of likely origin services, not proof of causation" |
| Code quality | Type hints, ruff, mypy, unit tests for splits, perturbations, metrics |
| Licensing | Respect dataset licenses; do not redistribute data unless allowed **[VERIFY]**; MIT or Apache-2.0 for code |

### 5.5 Pre-registered success criteria (revised)

Commit this section, with the power analysis from week 1, before any model result exists. The commit hash is the timestamp.

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

### 5.6 Deliverables

- GitHub repository with README, license, CI, Dockerfile and Makefile.
- Split definitions, perturbation library and baseline wrappers.
- Trained model artifacts and a model card.
- Technical report of 6-10 pages, optionally posted on arXiv or a blog.
- 2-3 minute demo video and a minimal CLI or web demo.

## 6. Data specification and splitting protocol

### 6.1 Sources

- **RCAEval** (primary): multi-source RCA benchmark with labeled fault-injection cases on applications such as Online Boutique, Sock Shop and Train Ticket **[VERIFY exact suites, case counts, fault types, file formats, download sizes, license]**. Start with the smallest metrics-only suite that gives you at least two systems.
- **Harder fault-propagation benchmark** (stretch): 1,430 cases per your source **[VERIFY existence, license, schema]**.
- **OpenTelemetry Demo**: live demo only, never used for evaluation.

Week-1 audit deliverable: a `data_audit.md` table of systems, fault types, injected services, cases per (system, fault type), runs per case, and which cases share an injection run.

### 6.2 Unified Case schema

| Field | Type | Notes |
| --- | --- | --- |
| case\_id | str | Globally unique |
| system | str | Application name |
| fault\_type | str | cpu, mem, delay, loss, etc. |
| run\_id | str | Injection run identifier; the primary group key |
| root\_cause\_service | str | Ground-truth label |
| inject\_time | int | Ground truth only; never a model input |
| window\_start, window\_end | int | Analysis window given to models; derivation documented |
| metrics | long table | service, metric, timestamp, value |
| topology | edge list | caller, callee; may be missing or inferred |
| logs, traces | optional | Stretch modalities |

**Validation checks (run at ingestion, fail loudly):** label service exists in metrics; no empty windows; timestamps monotonic; one checksum recorded per source file; no duplicate case\_id; topology nodes cover all metric services or the gap is logged.

### 6.3 Splits

| Split | Definition | Purpose |
| --- | --- | --- |
| random | Random by case | Only to demonstrate leakage inflation (RQ1) |
| group\_run | Grouped by run\_id (plus near-duplicate clusters) | Main protocol |
| lofo | Leave one fault type out | Unseen fault types |
| loso | Leave one system out | Unseen applications; report per fold |

**Near-duplicate detection:** cluster cases whose metric windows have very high pairwise correlation or that share system, fault type, injected service and timestamps within the same run. Merge clusters into one group before splitting. Record cluster sizes in the leakage report.

### 6.4 Rules

1. Hyperparameter tuning uses a validation fold carved from the training groups only.
2. `leakage_report.json` lists group overlap between train and test; it must be zero for every grouped split and CI fails otherwise.
3. **Test-fold freeze:** after the final configuration is chosen on validation folds, record its config hash in the repo. The perturbation sweep, ablations and GNN comparison reuse that frozen configuration or are retrained under the same protocol; none of them is used to choose hyperparameters.
4. Where LOSO has only a handful of folds, report each fold individually and do not pool into a single CI that implies more precision than the folds give.
5. The frequency prior is computed per system. Under LOSO it is undefined for unseen service names, so report it as the uniform-random-among-candidates floor there and say so.

### 6.5 Leakage probes (new, must-have)

| Probe | What it checks | Pass condition |
| --- | --- | --- |
| Identity probe | Shuffle graph features (degree, PageRank, betweenness) across services within a case, retrain, compare | Drop is small; a large drop means the model is using node fingerprints |
| Graph-only model | Train on graph-structure features alone | Near the frequency-prior floor on group\_run and LOSO |
| Window-leak probe | Compare accuracy when windows are shifted by P4 offsets and when window bounds are randomised | Accuracy falls smoothly; a cliff at zero offset signals the window encodes inject\_time |
| Mask-feature probe | Train on mask features alone on clean data | Reported; if strongly predictive on clean data, flag it as a possible collection artifact |

## 7. Baselines and reproduction

| Baseline | Type | Notes |
| --- | --- | --- |
| Frequency prior | Trivial floor | Most common root-cause service in training, per system |
| Rule-based | Heuristic | Earliest anomaly onset, or highest anomaly score among nodes with no anomalous upstream |
| BARO | Unsupervised, published | Use the RCAEval implementation **[VERIFY]**; reproduce reported numbers |
| Second published method | Published | Pick from the RCAEval list after reading it; prefer one with an official implementation and metrics-only input |
| LightGBM flat | Supervised | Per-service signal features only; no graph, no mask features |
| LightGBM flat + masking | Supervised | Same as above but trained with the same masking augmentation as the ranker; makes S3 fair |
| LLM-agent RCA | Stretch | Include, or explicitly exclude with a stated reason in the report |

### 7.1 Fairness rules

- Identical input windows, identical splits and identical case subsets for every method.
- Equal tuning budget for tunable methods (for example 30 configurations each, fixed in config).
- Supervised and unsupervised methods go in separate result tables; the comparison across tables is descriptive.
- Unsupervised methods get no access to training labels, including through hyperparameter choice on the test fold.
- Every method receives the same perturbed inputs; a method that cannot handle missing data is given a documented imputation, and that choice is part of its config.

### 7.2 Reproduction protocol (S7)

1. Before running, write down for each method: the published number, the paper's case subset, the metric definition, and the tolerance (default absolute 5 Top-1 points).
2. Run the official code on the paper's subset first, then on your splits.
3. If numbers differ, log the cause in a reproduction note (metric definition, window, seed, preprocessing, version).
4. **Stop rule:** if a baseline is not within tolerance by the end of week 6, report the gap with its documented causes, use the closest faithful configuration, and move on. Do not spend more than 3 weeks total on reproduction.

### 7.3 Common reasons reproduction fails

- Different metric definition (Top-k versus Avg@k, per-case versus per-service averaging).
- Different window around the injection time or different preprocessing such as normalisation and resampling.
- Subset mismatch (suites, systems or fault types included).
- Library or data-version drift.
- Randomness in methods that use sampling.

## 8. Method: the CascadeLens ranker

### 8.1 Features (per service, per case)

| Group | Features |
| --- | --- |
| Signal | Max z-score, change-point score and relative change for error rate, latency, CPU, memory; anomaly duration and peak magnitude; robust baseline statistics (median, MAD) from the pre-incident part of the window |
| Temporal | Anomaly onset relative to the earliest onset in the case (rank and delta); agreement between onset order and topological order |
| Graph | Fraction of upstream and downstream neighbours that are anomalous; personalized PageRank from anomalous nodes on the reversed call graph; in-degree, out-degree, betweenness (structure only) |
| Missingness | Per-metric availability mask; fraction of missing samples; "telemetry stopped before window end" indicator |

**Excluded by design:** service name, service ID, and any feature that directly identifies the service. Because structural graph features can still fingerprint a node in a fixed topology, the identity probe in section 6.5 is mandatory, and structural features are ablated as a group.

**Window hygiene:** z-scores and baselines use only the pre-incident part of the window. Onset detection must not use `inject_time`.

### 8.2 Stage 1: candidate retrieval

- Scorer: a LightGBM binary classifier or a simple heuristic over per-service features.
- Output: top-K candidates, K chosen on validation from {5, 8}, capped by the number of services.
- Report retrieval recall@K. It is an upper bound on Stage 2 Top-1, so a low value means fix Stage 1 before tuning Stage 2.

### 8.3 Stage 2: reranker

- LightGBM with the `lambdarank` objective, one query per case, candidates as documents, optimising NDCG@k.
- Small fixed-budget search over learning rate, leaves, min data in leaf, L1/L2 regularisation and feature fraction.
- Report Top-1, Top-3 and Avg@5 (mean of Top-k for k = 1..5).
- **Single-stage ablation:** rank all services in one LambdaRank model. If it matches the two-stage system within the paired CI, report that the retrieval stage adds no measurable value and prefer the simpler pipeline.

### 8.4 Missingness-aware training

- **Random masking:** drop metrics and whole services at rates sampled per training epoch.
- **Failure-correlated masking:** drop telemetry for services after anomaly onset with a higher probability near the fault.
- Always include mask features. Missing values are left as NaN for LightGBM rather than imputed, so that "missing" stays visible to the model.
- The augmentation-matched flat baseline in section 7 makes sure any gain is attributed to the features and ranking design and not to augmentation alone.

### 8.5 Calibration and abstention

1. Convert reranker scores to a case-level confidence: softmax over candidate scores, then temperature scaling or isotonic regression fitted on a validation fold.
2. Abstain when top-1 confidence falls below a threshold tau chosen on validation to hit a target coverage (for example 50%).
3. Report ECE (10 bins, with bootstrap CI), Brier score, reliability diagram, risk-coverage curve and AURC, each against random and oracle abstention.
4. Pool validation folds across seeds if a single fold is too small for stable calibration.

### 8.6 GNN ablation (stretch)

A message-passing network over the service graph using the same per-node features. Pre-committed expectation: with a few hundred cases it may not beat engineered features. It could win if the data has many cases per topology with rich propagation patterns, and lose if overfitting dominates. Report either way, with the same splits and tuning budget.

## 9. Perturbation suite and evaluation plan

### 9.1 Perturbations

| ID | Perturbation | Parameters | Realism label |
| --- | --- | --- | --- |
| P1 | Random metric loss | rate in {10, 20, 30, 50}% | Simulated, plausible |
| P2 | Failure-correlated loss | rate in {10, 20, 30, 50}%; higher drop probability for services near or after the fault | Simulated; the most production-like |
| P3 | Whole-modality loss | drop one metric type (logs/traces in stretch) | Simulated |
| P4 | Window shift | offset in {+-30, +-60, +-120} s | Simulated; doubles as window-leak probe |
| P5 | Window noise | random jitter on start and end | Simulated |
| P6 | Edge deletion | delete {10, 20, 30}% of edges | Simulated |
| P7 | Edge addition | add {10, 20}% spurious edges | Simulated |
| P8 | Inferred topology | replace graph with one inferred from correlations (or traces in stretch) | Closer to realistic |
| P9 | Unseen fault type | via lofo split | Real held-out condition |
| P10 | Unseen system | via loso split | Real held-out condition |
| P11 (added) | Sampling-rate degradation | downsample series by 2x, 5x | Simulated |
| P12 (added) | Delayed/ragged series | per-service clock offsets and late-arriving points | Simulated |
| P13 (added) | Metric renaming/unit drift | permute metric names within a family or rescale units | Simulated; tests silent schema drift |

**Requirements (all perturbations):** deterministic given (case\_id, perturbation, params, seed); never modify labels or `inject_time`; logged to a manifest with parameters and seed; unit-tested with property tests (labels unchanged, masks equal missing indicators, same seed gives byte-identical output).

**Honest labelling:** every table and figure carries a "simulated" or "held-out" tag. The benchmark has static topology and injected faults, so the report must not claim to model changing topology; it models *imperfect* topology.

### 9.2 Metrics

| Metric | Definition |
| --- | --- |
| Top-k | Fraction of cases whose true root-cause service is in the top k |
| Avg@5 | Mean of Top-k for k = 1..5 |
| Retrieval recall@K | Fraction of cases whose true service is in Stage 1's top K |
| ECE | Weighted mean gap between confidence and accuracy over 10 bins |
| Brier | Mean squared error of the top-1 confidence versus correctness |
| Risk-coverage and AURC | Error rate as low-confidence cases are abstained; AURC is the area under that curve |
| Robustness | Metric versus perturbation intensity; relative drop from clean |
| Latency | End-to-end seconds per case including feature extraction |

### 9.3 Statistics

1. **Power analysis (week 1).** Using the group counts from the data audit, simulate the smallest Top-1 difference detectable at 80% power with a group bootstrap. Write the result into the pre-registration. If the minimum detectable difference exceeds what you expect, adjust the claim language now, not later.
2. **Resampling unit.** Bootstrap by group (run or near-duplicate cluster), 10,000 resamples, for every CI.
3. **Comparisons are paired.** Use paired group-bootstrap on per-case outcomes, with a Wilcoxon signed-rank test as a check. Do not compare by looking at overlapping CIs.
4. **Seeds.** At least 5 seeds for stochastic models; report mean and spread across seeds, with the CI computed over cases on the seed-averaged outcome.
5. **Primary versus exploratory.** Primary: S1, S2, S3, S4, S7. Everything else is exploratory and labelled so. Apply Holm correction across the primary family and across the full perturbation grid per method; for exploratory plots show effect sizes without significance claims.
6. **Difference of differences (S3).** Compute `(ranker - baseline)` at 30% failure-correlated loss minus the same on clean data, per bootstrap resample.
7. **Reporting small samples.** For LOSO and LOFO report each fold on its own; pooled numbers are a secondary line.

### 9.4 Report structure

1. Leakage demonstration (RQ1).
2. Clean-data baseline tables, supervised and unsupervised separate.
3. Robustness curves by perturbation (RQ2, RQ3).
4. Ablations: each feature group, mask features, flat versus graph features, single-stage versus two-stage, GNN.
5. Calibration and risk-coverage (RQ5).
6. Failure analysis by fault type and system.
7. Limitations: benchmark versus production gap, static topology, injected faults, small group counts.

## 10. System design (TRD)

### 10.1 Architecture

Every component reads and writes versioned files, so any experiment reruns from config alone.

```text
Public benchmarks (RCAEval, ...)
   |
[Ingestion] -> unified Case schema (Parquet) + checksums
   |
[Splitter] -> grouped splits + leakage_report.json
   |
[Perturbation engine] -> seeded perturbed Case variants + manifest
   |
   +--> [Baselines] frequency | rule | BARO | method 2 | LightGBM flat (+masking)
   |
   +--> [CascadeLens ranker] Stage 1 retrieval -> Stage 2 LambdaRank
   |         -> calibration -> abstention
   v
[Evaluator] metrics, group bootstrap, paired tests, plots
   |
[Reporting] tables/figures -> API -> CLI / demo
```

### 10.2 Tech stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.11 |
| Data | Polars or pandas, PyArrow (Parquet) |
| Graph | NetworkX; PyTorch Geometric only for the GNN ablation |
| ML | LightGBM (LambdaRank), scikit-learn; PyTorch for the ablation |
| Config | YAML plus pydantic (or Hydra) |
| Tracking | Committed result JSONs; MLflow or W&B optional |
| API | FastAPI |
| Demo | CLI plus static graph figure first; Streamlit if time remains |
| Packaging | uv or Poetry; Docker |
| Quality | pytest, ruff, mypy |
| CI | GitHub Actions: lint, tests, 10-case smoke experiment under 5 minutes |

### 10.3 Repository structure

```text
cascadelens/
  README.md  pyproject.toml  Dockerfile  Makefile
  configs/        data.yaml splits.yaml perturbations.yaml ranker.yaml baselines/*.yaml
  src/cascadelens/
    data/         loaders, schema, validation
    splits/       grouped splitters, near-duplicate clustering, leakage report
    perturb/      P1-P13 implementations
    features/     signal, onset, graph, mask features
    baselines/    frequency, rule, BARO, method 2, LightGBM flat
    models/       retrieval, ranker, calibration, GNN (ablation)
    eval/         metrics, bootstrap, power analysis, significance, plots
    api/          FastAPI app
    cli.py
  experiments/    one folder per experiment: config + results JSON
  notebooks/      exploration only; no result depends on them
  tests/  docs/  scripts/reproduce_all.sh
  PREREGISTRATION.md
```

### 10.4 API (MVP, Should priority)

`POST /diagnose` takes a case (optional `case_id`, `topology` edge list, `metrics` rows, `window` start and end) and returns a `ranking` (service, score, evidence such as onset rank, upstream anomalous ratio, missing fraction), a case-level `confidence`, an `abstain` flag, `notes` (for example "telemetry missing for 2 services") and the fixed disclaimer "ranking of likely origin services, not proof of causation". `GET /health` and `GET /model-info` (version, training data hash, metrics) complete the surface. The API is read-only and makes no network calls to production systems.

### 10.5 Demo

Minimum: a CLI that prints the ranked list with evidence, plus a static graph figure with suspects coloured by score and an abstain banner. If time remains: Streamlit with a perturbation slider (for example drop 30% of telemetry) so the viewer watches the ranking change.

### 10.6 Testing

| Level | Examples |
| --- | --- |
| Unit | Splits have zero group overlap; perturbations are deterministic; metrics match hand-computed toy cases |
| Property | Perturbations never change labels or inject\_time; masked features equal missing indicators |
| Regression | Frequency baseline score on a toy dataset stays fixed |
| Leakage | Identity probe and window probe run on the toy dataset |
| Integration | End-to-end smoke run on 10 cases in CI under 5 minutes |
| Reproduction | Baseline numbers within the pre-stated tolerance (run manually, results committed) |

### 10.7 Reproducibility and MLOps

- `make all` runs download, splits, baselines, ranker, evaluation and figures. The download step documents any manual license acceptance, and CI uses a small checked-in sample if redistribution is not allowed.
- Seeds in config, versions pinned, Docker image published.
- Every result JSON stores config hash, git commit, data checksum and seed.
- A model card covers intended use, training data, metrics, limitations and failure modes.

### 10.8 Security, ethics and licensing

- Respect dataset licenses **[VERIFY]**; do not redistribute data unless allowed.
- Defensive purpose only; no features that help cause outages.
- State clearly that the system is a decision aid with uncertainty, not an autonomous operator.
- Code license MIT or Apache-2.0 after checking dataset terms.

### 10.9 Compute and storage

Metrics-only MVP runs on a laptop CPU with 16 GB RAM. A LightGBM training run takes minutes; the full grid takes hours, parallelised by seed. Storage is tens of GB if logs and traces are included, so download selected suites only **[VERIFY sizes]**.

## 11. Implementation plan

### 11.1 Week-by-week with gates

| Weeks | Work | Deliverable | Gate to proceed |
| --- | --- | --- | --- |
| 1 | Name collision search; data audit; schema; power analysis; commit PREREGISTRATION.md | data\_audit.md, power note, prereg commit | Prereg committed before any model result |
| 2-3 | Ingestion + validation; grouped splits; near-duplicate clustering; frequency and rule baselines; leakage demo | M1, M2, leakage\_report.json | Zero group overlap; random vs group\_run gap shown |
| 4-6 | Reproduce BARO and one more published method | M3, reproduction notes | Within tolerance, or stop rule applied at end of week 6 |
| 7-9 | Feature pipeline; Stage 1 retrieval; LightGBM LambdaRank; flat and augmentation-matched baselines; single-stage ablation | M4, M5 | Ranker beats frequency prior on group\_run (S1); retrieval recall@K measured |
| 10-11 | Perturbation suite P1-P13; missingness-aware training; leakage probes; **freeze final config** | M6, robustness curves | Curves for all primary perturbations; probes pass or are reported |
| 12 | Calibration, abstention, ablations, optional GNN | M7, risk-coverage plots | S4 and S5 evaluated |
| 13-14 | README, report, model card, CLI/API, demo video, clean-clone reproduction | M8 | S6: fresh clone plus one command regenerates headline tables |

**What to cut first if you fall behind:** (1) GNN, (2) Streamlit UI, (3) API, (4) P11-P13, (5) second published baseline reproduction (keep BARO), (6) second benchmark. Never cut the splits, leakage report, pre-registration or reproduction script.

### 11.2 Milestones (definition of done)

| ID | Done when |
| --- | --- |
| M1 | Ingestion and schema validation pass on all selected cases |
| M2 | Splits, leakage report, frequency and rule baselines run |
| M3 | BARO and second method reproduced within tolerance, or gap documented |
| M4 | Feature pipeline and Stage 1 retrieval with measured recall@K |
| M5 | Stage 2 reranker evaluated against the frequency prior on group\_run |
| M6 | Perturbation suite and robustness curves produced |
| M7 | Calibration, abstention and ablations complete |
| M8 | API/CLI, report, README, video done; clean-clone reproduction passes |

## 12. Risks and failure modes

| # | Likely failure | Likelihood | Mitigation | Check you can run |
| --- | --- | --- | --- | --- |
| 1 | Leakage inflates results | High | Grouped splits, near-duplicate clustering | leakage\_report.json is zero; random-vs-group gap reported |
| 2 | Model memorises service identity via structural features | High | Exclude names; identity probe | Shuffled-graph-feature drop is small |
| 3 | Too few groups for any significant result | High | Power analysis; paired tests; honest claims | Minimum detectable effect written into prereg |
| 4 | Window encodes inject\_time | Medium | Window-leak probe | No accuracy cliff at zero shift |
| 5 | Gain comes from augmentation, not design | Medium | Augmentation-matched baseline | S3 computed against it |
| 6 | Baseline reproduction stalls | Medium | Official RCAEval code; 3-week cap; stop rule | Reproduction note per method |
| 7 | Mask features predict via collection artifact | Medium | Mask-only probe on clean data | Probe result reported |
| 8 | Small-data overfitting | High | Gradient boosting first; strong regularisation; GNN only as ablation | Validation vs test gap tracked |
| 9 | Scope creep | High | Gates; stretch only after S1-S7 evaluated | Weekly gate review |
| 10 | Unverified citations or dataset facts | Medium | Claims checklist (section 14) | All [VERIFY] items cleared before publishing |

Claims the data cannot support: causal discovery, real-world changing topology (only simulated imperfection), general production performance, and any LOSO claim with fewer than about five systems.

## 13. Write-up and career framing

### 13.1 Report outline (6-10 pages)

1. Abstract and contributions.
2. Background and related work (metric-, trace-, graph-, causal- and LLM-based RCA; benchmark critiques).
3. Benchmarks and Case schema.
4. Evaluation protocol: grouped splits, leakage probes, statistics.
5. Baselines and reproduction results.
6. CascadeLens method.
7. Results: leakage, clean performance, robustness, ablations, calibration.
8. Failure analysis and limitations.
9. Reproducibility statement and ethics.

### 13.2 README outline

One-paragraph pitch; headline result table (filled with your real numbers); 3-command quickstart; repo map; how to reproduce; baseline list with citations; limitations; license and data notice; citation block.

### 13.3 Demo video script (2-3 minutes)

1. (0:00) The problem: many alerts, which service first?
2. (0:20) Why published numbers mislead: random vs grouped split bar.
3. (0:50) The harness: one command, perturbation list.
4. (1:15) Live CLI/UI: load a case, drop 30% telemetry, watch the ranking and the abstain banner.
5. (2:00) Robustness curve and honest summary, including any negative result.
6. (2:30) Repo link and how to reproduce.

### 13.4 Interview questions this prepares you for

| Question | Strong answer outline |
| --- | --- |
| How did you prevent leakage? | Group by run and near-duplicate cluster; zero-overlap check in CI; random-vs-group gap quantified |
| Why not report overlapping CIs? | Paired group bootstrap on per-case outcomes; power analysis done up front |
| What if the model just learned the graph? | Identity probe and graph-only model |
| How do you discuss a negative result? | State the pre-registered criterion, the outcome, the likely cause, and what the harness still contributes |
| Why a two-stage ranker? | Retrieval caps recall; single-stage ablation shows whether the stage earns its complexity |
| What would you change for production? | Real incident data, live topology, latency budget, drift monitoring, human-in-the-loop |

### 13.5 Resume bullets (fill with your real numbers)

- Built an open evaluation harness for microservice RCA with group-aware splits and a [N]-perturbation robustness suite; showed random splits inflate Top-1 by [X] points.
- Trained a missingness-aware two-stage LightGBM LambdaRank ranker; [result vs. baselines with CI] under [Y]% failure-correlated telemetry loss.
- Reproduced [N] published RCA baselines within [tolerance]; released code, splits and one-command reproduction (CI-verified).
- Added calibrated confidence and abstention; selective Top-1 of [Z] at 50% coverage.

Do not fill any bracket until the number exists in a committed result file.

## 14. Claims verification checklist

Confirm each from its primary source before publishing.

- [ ] Uptime Institute outage-cost percentages and their scope
- [ ] Google Cloud incident of June 12, 2025: details from the official post-incident report
- [ ] Cloudflare incident duration and cause from its official blog
- [ ] FSE 2026 benchmark: 1,430 cases, 21% average and 37% best Top-1, number of models evaluated
- [ ] The September 2026 retrieval-versus-reranking paper: title, authors, findings
- [ ] AegisLab and "CascadeLens" name overlap (GitHub, arXiv, PyPI)
- [ ] RCAEval case counts, fault types, systems, baseline implementations
- [ ] Dataset licenses, download formats and sizes
- [ ] BARO published numbers and the case subset they use

## 15. Open decisions

1. Which second published method to reproduce (decide after reading the RCAEval baseline list in week 1).
2. Whether the harder benchmark is public and compatible; if not, drop it and say so.
3. Include or explicitly exclude the LLM-agent baseline, with a stated reason.
4. Nested cross-validation or a one-shot frozen test protocol (recommended: frozen at end of week 11).
5. Demo scope: CLI plus static figure only, or add Streamlit and API.

## 16. Your profile (fill in)

- Target role: [ ]
- Skills: [ ]
- Experience level: [ ]
- Weeks and hours per week: 14 weeks, 10-15 hours
- Compute: laptop 16 GB or free Colab/Kaggle
