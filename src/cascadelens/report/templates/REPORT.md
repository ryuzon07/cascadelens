# CascadeLens: root-cause service ranking under imperfect telemetry

**Status: {{status}}.** No config is frozen, so every ranker verdict here previews the pre-registered
test. All tables are generated from committed result JSONs, traced in
[provenance](results/provenance.md).

## Abstract

Root-cause ranking methods for microservices are usually scored on clean telemetry and on splits
that can leak near-duplicate runs. We built an evaluation harness with grouped splits, leakage
probes, a perturbation suite and paired group-bootstrap statistics, and used it to test a
missingness-aware two-stage LightGBM ranker on {{n_cases}} RE1 cases from RCAEval. On clean data
the ranker was non-inferior to the best baseline within a 12-point margin, but flat LightGBM stayed
ahead by {{s2_unshifted}} points. Under simulated failure-correlated loss the ranker's margin over
both LightGBM baselines grew, by {{s3_flat}} and {{s3_masking}} points, while its margin over
non-learned baselines shrank. Calibrated abstention at 50% coverage raised Top-1 by {{s4}} points.
Feature ablation found that only signal features mattered measurably on this benchmark. The system
gives a {{disclaimer}}.

**Contributions.** A reusable leakage-aware harness, a pre-registered evaluation including failed
criteria, a quantified random-split inflation, and calibrated abstention tested under shift.

## Background and related work

Root-cause analysis work spans metric-based, trace-based, graph-based, causal and LLM-based methods.
[VERIFY: survey citations for each family are not in this repository]

Two works are confirmed from the repository's reproduction notes. BARO (Pham, Ha, Zhang, FSE 2024,
arXiv 2405.09330) detects change points per metric. The ASE 2024 evaluation by the same authors
(arXiv 2408.13729) studied causal methods for this task, including ICA-LiNGAM with random walk.

Benchmark critiques, and the claims in the PRD's motivation, remain unverified and are not repeated
here. [VERIFY: benchmark-critique citations] See the [claims list](CLAIMS_TODO.md).

## Benchmarks and Case schema

We use RE1 from RCAEval: {{n_cases}} metrics-only cases across Online Boutique, Sock Shop and Train
Ticket, with five fault types and five repetitions per injection, as audited in
[the data audit](data_audit.md). A `Case` holds a long metrics table of service, metric, timestamp
and value, a window, an optional topology, and ground-truth fields that feature code cannot read.
A missing sample is a null row, never imputed. RE1 records no topology, so graphs are inferred per
case from pre-incident latency, chosen by stability across repetitions without labels. They have not
been validated against real call graphs.

## Evaluation protocol

**Splits.** `group_run` holds out whole injection runs. LOFO holds out a fault type, and LOSO holds
out a system. Group overlap between train, validation and test is checked by code.

**Statistics.** Intervals are percentile group bootstraps with 10,000 resamples over seed-averaged
cases. All comparisons are paired, and overlapping intervals are never used. Holm correction covers
the primary family, and exploratory results show effect sizes only.

**Pre-registration.** Criteria S1 to S7 are fixed in `PREREGISTRATION.md`, which discloses what was
known when it took effect. Amendments there cover inferred graphs, the `rule_upstream` tie-break,
the one-thread Stage 2 setting, the P2 label leak, and the reading of "baselines" in S3. That last
amendment keeps every comparator, so S3 is not narrowed after the fact.

**Leakage probes.** Four probes test identity, graph-only skill, window leakage and mask-only skill.

{{table:probes}}

The window definition does not read `inject_time`, but {{window_share}}% of injections sit within 2%
of the window's middle, so it depends on the injection indirectly and strongly. The mask-only flag
means missingness alone reaches {{mask_only}}, against a floor of {{mask_floor}}.

## Baselines and reproduction

Supervised baselines are a frequency prior and LightGBM over flat features. Unsupervised baselines
are `rule`, `rule_upstream`, BARO and ICA-LiNGAM with random walk.

{{table:reproduction}}

S7 is an observed result, because reproduction finished before the pre-registration took effect.
BARO reproduces within tolerance. ICA-LiNGAM misses on Sock Shop by {{lingam_gap}} against a 0.05
tolerance. See [the reproduction plan](reproduction/PLAN.md) for the causes considered.

## CascadeLens method

Stage 1 is a LightGBM scorer over per-service features that retrieves K candidates, with
out-of-fold scores for training Stage 2. Stage 2 is a LightGBM LambdaRank reranker. The tuning budget
equals flat LightGBM's, and K is drawn inside it. Features fall into signal, temporal, graph,
structural and mask groups, and service identity is refused as a feature.

Missingness-aware training adds copies of training cases with random and failure-correlated masks,
computed without ground truth, and features are recomputed on the masked series. Each copy keeps its
group. Confidence is a softmax over candidate scores, calibrated on each fold's validation cases
pooled across seeds. Those cases are not pooled across folds, because validation cases of one fold
are test cases of another. The system abstains below a threshold chosen on validation.

## Results

{{table:headline}}

### Leakage

![Top-1 on a random split and on group_run](results/figures/leakage.png)

{{table:leakage}}

The random split inflates flat LightGBM by {{leak_flat}} points. This is a real effect on real cases,
though its interval is wide.

### Clean performance

{{table:clean_supervised}}

{{table:clean_unsupervised}}

Supervised and unsupervised methods should be compared descriptively only. Flat LightGBM is the best
method on clean data. The plain ranker's difference from it is {{plain_unshifted}} points, and the
augmented ranker's is {{s2_unshifted}}.

### Robustness

P1 to P8 and P11 to P13 are simulated. P9 and P10 are held-out splits.

![Top-1 against simulated failure-correlated loss, supervised methods](results/figures/robustness_p2_supervised.png)

![Top-1 against simulated failure-correlated loss, unsupervised methods](results/figures/robustness_p2_unsupervised.png)

{{table:robustness}}

At 30% failure-correlated loss (simulated), `ranker_augmented` scores {{ranker_p2}}, the plain
ranker {{plain_p2}}, flat LightGBM {{flat_p2}} and the augmentation-matched flat {{masking_p2}}. The
full grid is in [the robustness table](results/tables/robustness_full.md).

### Ablations

{{table:ablation}}

Signal features carry the ranker: removing them costs {{ablate_signal}} points. Removing graph
features changes {{ablate_graph}}, and removing temporal features changes {{ablate_temporal}}, so
neither helps measurably on clean data. The graph here is inferred, which may be part of the reason.
The single-stage version ties on clean data ({{single_stage_clean}}), and the two stages add
{{single_stage_p2}} under simulated P2 loss.

### Calibration

![Error rate by coverage on clean data](results/figures/risk_coverage.png)

Reliability diagrams and per-slice risk-coverage curves are in
`experiments/calibration/figures/`.

{{table:calibration}}

{{table:selective}}

{{table:coverage_drift}}

ECE falls from {{ece_before}} to {{ece_after}}, a paired difference of {{s5}}. S5 is an easy bar,
because ranking scores have no natural scale, so the uncalibrated value is partly arbitrary. AURC is
{{aurc}}, against {{aurc_random}} for random abstention and {{aurc_oracle}} for the oracle.

Coverage held near its target on clean data, at {{coverage_clean}}%, but fell to {{coverage_p2}}%
under simulated P2 loss and {{coverage_shift}}% under a 120 s early shift. This is the safe direction.
On held-out folds it was unstable, and the pooled figure hides that.

### Negative and null results

- **S3 against non-learned baselines failed.** Margins over frequency ({{s3_frequency}}), BARO
  ({{s3_baro}}), `rule` and `rule_upstream` shrank. The likely cause is structural for the frequency
  prior: it reads no telemetry, so its score cannot fall, and any loss in the ranker shrinks the
  margin. The pre-registration does not limit "baselines" to learned ones, so this stands as a
  failure.
- **The ranker did not beat flat LightGBM on clean data.** S2 passing only rules out a 12-point
  deficit.
- **Graph features added nothing measurable.** The graph-aware design is not supported on this data.
- **Two stages did not help on clean data.**
- **Augmentation cost clean accuracy:** the ranker went from {{plain_clean}} to {{ranker_clean}}, and
  flat LightGBM from {{flat_clean}} to {{masking_clean}}.
- {{s6_limitation}}
- **ICA-LiNGAM missed tolerance on one system.**

## Failure analysis

{{table:failure_fault}}

{{table:failure_system}}

Delay and loss are the hard fault types, and Train Ticket and Online Boutique the hard systems. The
ranker generalises far better than flat LightGBM to an unseen Online Boutique, but worse to an unseen
Train Ticket, so neither method dominates on held-out systems. Per-fold intervals are wide, because
each fold holds few groups.

## Limitations

- **Small sample.** Three systems and 75 groups. Power analysis shows S3 can detect only large
  effects, so a null result means no effect of about 15 points or more, not no effect.
- **Injected faults.** Nothing here is validated on production incidents.
- **Static topology.** Graphs are inferred once per case. We do not model changing topology.
- **Simulated perturbations.** P1 to P8 and P11 to P13 are simulated. P2 uses ground truth to place
  gaps, so mask features may gain from the gaps revealing the fault: {{mask_gain}} points.
- **Window dependence.** Windows depend on `inject_time` indirectly, as described above.
- **Validation reuse.** Calibration uses validation cases that also steered early stopping and
  search.
- **Fragile training.** The reranker stops after few boosting rounds, and before the one-thread fix
  one group_run fold swung widely across settings.
- **Not causal discovery.** The output is a ranking.

## Reproducibility statement

Each result JSON records its config hash, git commit, data checksum and seed. `make all` runs every
stage, and a stage is skipped when its inputs are unchanged. `scripts/clean_clone_check.sh` verifies
a fresh clone on the committed fixture. That is not S6: S6 needs a full run from a fresh clone on
the real data, recorded in [the reproduction log](results/reproduction_log.md). The current verdict
is in the headline table. Results were not rerun to improve any number.

## Ethics and intended use

This is a read-only decision aid for defensive use. It is not an autonomous operator, and it is not
ready for production. A wrong ranking can mislead responders, so a person must check the evidence.
The dataset is not redistributed. Its terms, MIT on RCAEval and CC BY 4.0 on Zenodo, are summarised
in the [README](../README.md#license-and-data-notice). See the [model card](../MODEL_CARD.md).

To export a PDF, install pandoc and run `pandoc docs/REPORT.md -o docs/REPORT.pdf`. Pandoc was not
available when this file was generated, so no PDF is committed.
