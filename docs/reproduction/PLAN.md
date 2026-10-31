# Reproduction plan

Written before any reproduction code ran (PRD 7.2). Numbers below were read from the papers and the
repository source, not from memory.

## Sources

| Item | Value |
| --- | --- |
| Library | RCAEval 1.11.0 from PyPI; repository commit `dc9f48786db50cf9469f2c596b79f01907d4ac91` (tag 1.11.0) |
| BARO paper | Pham, Ha, Zhang. *BARO: Robust Root Cause Analysis for Microservices via Multivariate Bayesian Online Change Point Detection*. FSE 2024, arXiv 2405.09330 |
| Benchmark paper | Pham, Ha, Zhang. *Root Cause Analysis for Microservice System based on Causal Inference: How Far Are We?* ASE 2024, arXiv 2408.13729 |
| Data | RE1-OB, RE1-SS, RE1-TT `data.csv`: 375 cases, five fault types, five services per system, five repetitions |

The WWW 2025 RCAEval paper has only an RE2 Train Ticket table (Table 6), and the repository README
quotes only that table, so neither gives RE1 numbers.

## Second method: ICA-LiNGAM with random walk

`lingam_randomwalk`, the row **ICALiNGAM-RW** of ASE 2024 Table 5.

| Requirement | Evidence |
| --- | --- |
| Official implementation | `RCAEval/e2e/pc_randomwalk.py` |
| Metrics only | It takes the metrics frame alone |
| CPU laptop | Table 6 of the ASE paper: 0.53 to 2.77 s per case on OB and SS, 16.39 s on TT, on 8 CPUs |
| Not a BARO variant | Builds a causal graph, then ranks with a random walk. BARO compares two windows per metric |

Methods ruled out:

| Method | Reason |
| --- | --- |
| NSigma | Same family as BARO: it only swaps the scaler |
| CIRCA | 3792.29 s per TT case (Table 6), so about 130 hours for RE1-TT |
| PC-RW, PC-PR | About 130 s per TT case, so about 4.5 hours per run and two runs are needed |
| Granger-RW, fGES-RW, FCI-RW | 150 to 380 s per TT case |
| RCD | Needs a separate Python 3.8 environment; our package requires 3.11 |
| CausalRCA, MicroCause, RUN | Need torch, or take far longer than a laptop run allows |
| ε-Diagnosis | Fast, but a per-metric two-sample test and not a different family |

Expect weak scores. The paper reports it near the Dummy row, and that is the informative contrast
with BARO: it tests whether causal-graph ranking adds anything on this data.

## Published numbers

All values are Avg@5 from **ASE 2024, Table 5, PDF page 7**, on the rows BARO [𝑡Δ = 0] and
ICALiNGAM-RW. The columns "Online Boutique", "Sock Shop 2" and "Train Ticket" are the RE1 systems.

| Method | System | CPU | MEM | DISK | DELAY | LOSS | Mean of the five |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BARO | Online Boutique | 0.97 | 1.00 | 0.91 | 0.98 | 0.67 | 0.906 |
| BARO | Sock Shop 2 | 1.00 | 0.98 | 0.94 | 0.98 | 0.87 | 0.954 |
| BARO | Train Ticket | 0.90 | 0.96 | 0.84 | 0.77 | 0.66 | 0.826 |
| ICALiNGAM-RW | Online Boutique | 0.40 | 0.37 | 0.34 | 0.14 | 0.20 | 0.290 |
| ICALiNGAM-RW | Sock Shop 2 | 0.48 | 0.46 | 0.49 | 0.45 | 0.36 | 0.448 |
| ICALiNGAM-RW | Train Ticket | 0.16 | 0.05 | 0.11 | 0.03 | 0.13 | 0.096 |

The last column is computed here; the table does not print it.

BARO FSE 2024, Table 3, PDF page 16 (coarse-grained, CPU, MEM, DELAY, LOSS, without DISK) reports
averages of 0.86, 0.95 and 0.81. That is a different protocol: fewer fault types, and BOCPD
detection rather than a given time. It is a cross-check only, not the target.

**No published Top-1 or AC@1 for RE1 was found** in these papers: the tables report Avg@5 only.
AC@1 is reported as ours with no published value, and is not part of pass or fail.

## Case subset and metric

- **Subset:** every case of RE1-OB, RE1-SS and RE1-TT, in `data.csv`.
- **Metric:** Avg@5, the mean of AC@1 to AC@5 (ASE section 3.2), per case and then averaged over
  cases of a fault type. It is not averaged per service.
- **Granularity:** service level. Metric rankings are folded to services by
  `RCAEval.benchmark.scoring.service_ranks`: first occurrence kept, the part before the first `_`,
  with `-db` removed. Each RE1 service-fault pair has the same number of cases, so per-case and
  per-service means agree here.
- **Protocol:** the steps of `process()` in `main.py`: drop `time.1`, drop `_latency-50`, rename
  `_latency-90`, replace inf, forward-fill, fill with 0, keep the `length * 60 // 2` rows either
  side of `inject_time`, with `dataset=re1-*`, `dk_select_useful=False`, `n_iter=<column count>`.

## Tolerance

Fixed now, before any result. The prompt's default of 5 Top-1 or AC@1 points cannot apply because
no AC@1 is published, so it is applied to the published metric:

- **Pass:** the mean of the five fault types of a system is within absolute **5 Avg@5 points** of
  the published mean, on all three systems.
- Each fault cell is reported too, and a cell past tolerance is recorded with its cause, but does
  not fail the method alone. Published cells are rounded to two decimals.

## Inputs and mapping to the Case schema

| Method | Needs | Reproduction run | Evaluation run |
| --- | --- | --- | --- |
| BARO | `data` frame with `time`, and `inject_time` that splits normal from anomalous rows | The true `inject_time`, as in `main.py` [𝑡Δ = 0] | See below |
| ICA-LiNGAM RW | `data` frame; `inject_time` is accepted and ignored | Rows chosen around the true `inject_time`, as in `main.py` | The whole window; no timestamp involved |

The Case long table `(service, metric, timestamp, value)` pivots to one wide column named
`<service>_<metric>` per series, plus `time`. The adapter keeps the column-to-service map, so
services map back exactly.

### Anomaly timestamp in the evaluation run

The repository has no change-point detector: `baro()` takes the time as an argument, and the BOCPD
step of the paper is not shipped outside `legacy/`. Our other baselines get no `inject_time`, and
the audit shows that the injection sits within 2% of the middle of the recording in about 99% of
cases, so **the window midpoint would be `inject_time` in disguise**.

The evaluation run therefore uses `window_start + 0.25 * (window_end - window_start)`, the same
`baseline_fraction` as the rule baseline. It reads only the window and never `inject_time`. The
normal part is the first quarter, and the anomalous part is the rest, which still holds pre-incident
data.

This differs from the paper, which gives the true time, so scores will be lower. A midpoint run is
reported as a labelled sensitivity check for Part 9's window-leak probe, and is never a headline.

## Known reasons it might not reproduce

- `data.csv` against other files: `main.py` has a commented-out switch to `simple_data.csv`, and the
  paper does not say which was used.
- Data length: the paper studies lengths up to 10 minutes while `main.py` defaults to `--length 20`.
  The default is primary, and `--length 10` is a sensitivity run.
- Library drift: RCAEval changed since 2024. Its changelog records fixes to PageRank methods and to a
  duplicate `time` column in 50 RE1-OB cases, and disk faults on RE1 are rescored against latency.
- Randomness: the random walk and ICA have no seed parameter, so results vary by run. Global seeds
  are fixed, and several seeds are run for this method.
- Metric definitions (PRD 7.3): the `-db` folding and service-level ranking must match exactly.
- Python 3.11 here against 3.12 to 3.14 recommended upstream, and Windows against Linux.
- Raw-file defects in the audit: seven RE1 cases have untimed tail rows.

## Stop rule

If a method misses tolerance after a reasonable debugging effort, the gap and its causes go in
`docs/reproduction/<method>.md`, the closest faithful configuration is used, and work moves on. The
method's logic is not changed to reach a number.

## Amendments after the first run

The text above is what was planned. These changes came after results existed.

1. **Input files.** The plan named `data.csv`, and listed `simple_data.csv` as a risk. BARO on
   `data.csv` missed badly on Sock Shop and Train Ticket (0.576 and 0.261 against 0.954 and 0.826),
   and matched on Online Boutique, the one system with no `simple_data.csv`. The closest faithful
   configuration reads `simple_data.csv` where a case has one, as the commented-out code in
   `main.py` does. The method was not changed. The `data.csv` result stays in
   `docs/reproduction/baro.md`.
2. **Metric set in the evaluation run.** `simple_data.csv` columns are derived series, not a subset
   of `data.csv`, so our cases cannot supply them. ICA-LiNGAM is infeasible on all metrics: 263 s
   for one Sock Shop case with 438 columns, and Train Ticket has 1,242 columns against 961 rows.
   Before any evaluation score existed, a label-free key metric set was fixed (`KEY_METRICS` in
   `published/adapter.py`). `lingam_randomwalk` runs on it, which departs from the identical inputs
   of PRD 7.1. `baro` stays on all metrics, and `baro_key_metrics` gives the equal-input comparison.
3. **Seeds.** The plan expected several seeds for the second method. Changing the seed left its top
   ten unchanged in all 36 cases checked, so one seed is used.

## Ambiguities

Resolved here by explicit choice, none needing a stop:

1. No published AC@1 exists for RE1, so tolerance applies to Avg@5.
2. The paper's data length is not stated for Table 5, so the `main.py` default is primary.
3. Two BARO tables exist. ASE Table 5 is the target because it covers all five fault types with a
   given time. FSE Table 3 is a cross-check.
4. "Sock Shop 2" is taken to be RE1-SS: RCAEval lists `sock-shop-2` among its RE1 datasets, and its
   five fault types match.
