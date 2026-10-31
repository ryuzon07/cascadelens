"""Simulated power of the paired group bootstrap (PRD section 9.3, item 1).

Outcomes come from a latent Gaussian model. For method m, group g and case i, a case is a hit when
``sqrt(rho) * U[m, g] + sqrt(1 - rho) * E[m, g, i]`` falls below the normal quantile of the
method's Top-1. Here ``rho`` is ``within_group_correlation``, and U and E are standard normal, with
correlation ``paired_correlation`` between the two methods at both levels. The marginal Top-1 of
each method is exact for any correlation.

Every assumption is a parameter. The simulation treats the outcomes of each case as given, so it
leaves out variation from training on different folds, which can only lower real power. Latent
draws are shared across the differences of a scenario, which keeps its power curve smooth.

One matrix of resample counts is shared by all simulations of a scenario. It is drawn without
looking at any outcome, so this saves time without biasing the test.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import polars as pl

from cascadelens.config import config_hash
from cascadelens.data.errors import DataError
from cascadelens.data.io import read_manifest
from cascadelens.eval.bootstrap import resample_counts
from cascadelens.eval.config import EvalConfig, PowerConfig, PowerScenario
from cascadelens.eval.errors import EvalError
from cascadelens.eval.plots import PowerCurve, power_curves
from cascadelens.seed import derive_case_seed

Integers = npt.NDArray[np.int64]
Booleans = npt.NDArray[np.bool_]
CHUNK = 250


def group_sizes_from(path: Path) -> Integers:
    """Read case counts per group, in ``group_id`` order, from a ``case_id, group_id`` table."""
    if not path.is_file():
        raise EvalError(f"No groups in {path}; run `cascadelens splits build` first")
    counts = pl.read_parquet(path).group_by("group_id").len().sort("group_id")
    return counts["len"].to_numpy().astype(np.int64)


def _check_sizes(group_sizes: npt.ArrayLike) -> Integers:
    sizes = np.asarray(group_sizes, dtype=np.int64)
    if sizes.ndim != 1 or sizes.size < 2 or np.any(sizes < 1):
        raise EvalError("Need at least two groups with at least one case each")
    return sizes


def _latent(
    sizes: Integers, correlation: float, paired: float, n: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return latent scores of the two methods, ``n`` simulations by cases."""
    groups, cases = sizes.size, int(sizes.sum())

    def pair(shape: tuple[int, int]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        first = rng.standard_normal(shape)
        return first, paired * first + sqrt(1 - paired**2) * rng.standard_normal(shape)

    group_a, group_b = pair((n, groups))
    case_a, case_b = pair((n, cases))
    shared, own = sqrt(correlation), sqrt(1 - correlation)
    return (
        shared * np.repeat(group_a, sizes, axis=1) + own * case_a,
        shared * np.repeat(group_b, sizes, axis=1) + own * case_b,
    )


def simulate_paired_outcomes(
    group_sizes: npt.ArrayLike,
    scenario: PowerScenario,
    difference: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[Booleans, Booleans]:
    """Return hits of the ranker and of the baseline, ``n_simulations`` by cases.

    Cases are ordered by group. The ranker's true Top-1 is ``baseline_top1 + difference``.
    """
    sizes = _check_sizes(group_sizes)
    target = scenario.baseline_top1 + difference
    if not 0 <= target <= 1:
        raise EvalError(f"Top-1 {target} is outside [0, 1]")
    ranker, baseline = _latent(
        sizes,
        scenario.within_group_correlation,
        scenario.paired_correlation,
        n_simulations,
        rng,
    )
    normal = NormalDist()
    cut = np.inf if target == 1 else -np.inf if target == 0 else normal.inv_cdf(target)
    return ranker < cut, baseline < normal.inv_cdf(scenario.baseline_top1)


@dataclass(frozen=True)
class PowerResult:
    """Power by scenario, alpha and difference, with the minimum detectable differences."""

    grid: pl.DataFrame
    minimum_detectable: pl.DataFrame
    n_groups: int
    n_cases: int


def _bootstrap_bounds(
    ranker: Booleans,
    baseline: Booleans,
    sizes: Integers,
    counts: npt.NDArray[np.float64],
    quantiles: Sequence[float],
) -> npt.NDArray[np.float64]:
    """Return interval bounds, of shape quantiles by simulations, computed in chunks."""
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    case_totals = counts @ sizes.astype(np.float64)
    parts = []
    for begin in range(0, ranker.shape[0], CHUNK):
        difference = ranker[begin : begin + CHUNK].astype(np.float64) - baseline[
            begin : begin + CHUNK
        ].astype(np.float64)
        group_sums = np.add.reduceat(difference, starts, axis=1)
        means = (group_sums @ counts.T) / case_totals
        parts.append(np.quantile(means, quantiles, axis=1))
    return np.concatenate(parts, axis=1)


def estimate_power(
    group_sizes: npt.ArrayLike,
    scenario: PowerScenario,
    differences: Sequence[float],
    alphas: Sequence[float],
    n_simulations: int,
    n_resamples: int,
    seed: int,
    noninferiority_margin: float = 0.03,
) -> pl.DataFrame:
    """Simulate power over a grid of true differences.

    A simulation rejects when the two-sided percentile interval excludes zero, which is the S1
    test, so the row with difference 0 holds the false positive rate. ``noninferiority_power`` is
    the chance the lower bound clears minus the margin, which is the S2 test. Differences that
    would push Top-1 above 1 are skipped.
    """
    sizes = _check_sizes(group_sizes)
    source = derive_case_seed(scenario.name, "power", {}, seed)
    counts = resample_counts(sizes.size, n_resamples, source)
    quantiles = [q for alpha in alphas for q in (alpha / 2, 1 - alpha / 2)]
    rows = []
    for difference in differences:
        if scenario.baseline_top1 + difference > 1:
            continue
        rng = np.random.default_rng(source)
        ranker, baseline = simulate_paired_outcomes(sizes, scenario, difference, n_simulations, rng)
        bounds = _bootstrap_bounds(ranker, baseline, sizes, counts, quantiles)
        for position, alpha in enumerate(alphas):
            lower, upper = bounds[2 * position], bounds[2 * position + 1]
            power = float(np.mean((lower > 0) | (upper < 0)))
            rows.append(
                {
                    "scenario": scenario.name,
                    "alpha": alpha,
                    "difference": difference,
                    "power": power,
                    "monte_carlo_se": sqrt(power * (1 - power) / n_simulations),
                    "half_width": float(np.mean((upper - lower) / 2)),
                    "noninferiority_power": float(np.mean(lower > -noninferiority_margin)),
                }
            )
    return pl.DataFrame(rows)


def minimum_detectable_difference(
    differences: Sequence[float], powers: Sequence[float], target: float
) -> float | None:
    """Interpolate the smallest positive difference reaching ``target``, or None if none does."""
    points = sorted(zip(differences, powers, strict=True))
    for (low, low_power), (high, high_power) in itertools.pairwise(points):
        if high > 0 and high_power >= target:
            if low_power >= target:
                return high
            return low + (high - low) * (target - low_power) / (high_power - low_power)
    return None


def power_analysis(group_sizes: npt.ArrayLike, config: PowerConfig) -> PowerResult:
    """Run every scenario and find its minimum detectable Top-1 difference."""
    sizes = _check_sizes(group_sizes)
    grid = pl.concat(
        [
            estimate_power(
                sizes,
                scenario,
                config.differences,
                config.alphas,
                config.n_simulations,
                config.n_resamples,
                config.seed,
                config.noninferiority_margin,
            )
            for scenario in config.scenarios
        ]
    )
    rows = []
    for scenario in config.scenarios:
        for alpha in config.alphas:
            part = grid.filter(pl.col("scenario") == scenario.name, pl.col("alpha") == alpha)
            detectable = minimum_detectable_difference(
                part["difference"].to_list(), part["power"].to_list(), config.target_power
            )
            rows.append(
                {"scenario": scenario.name, "alpha": alpha, "minimum_detectable": detectable}
            )
    table = pl.DataFrame(rows, schema_overrides={"minimum_detectable": pl.Float64})
    return PowerResult(grid, table, int(sizes.size), int(sizes.sum()))


def _percent(value: float | None) -> str:
    return "not reached" if value is None else f"{100 * value:.1f}"


def render_report(
    result: PowerResult, config: EvalConfig, data_checksum: str, figure: str | None
) -> str:
    """Render the Markdown report. The same inputs give identical text."""
    power = config.power
    first_alpha = power.alphas[0]
    lines = [
        "# Power analysis",
        "",
        "Generated by `cascadelens eval power`. Do not edit by hand.",
        "",
        f"- Groups: {result.n_groups}; cases: {result.n_cases}, from `{power.groups_file}`.",
        f"- Data manifest checksum: `{data_checksum}`.",
        f"- Config hash: `{config_hash(config)}`.",
        f"- {power.n_simulations} simulations per point; {power.n_resamples} group-bootstrap "
        f"resamples; target power {power.target_power:.0%}; seed {power.seed}.",
        "",
        "## Assumptions",
        "",
        "These values are assumed, not measured: no model result exists yet.",
        "",
        "| Scenario | Baseline Top-1 | Within-group correlation | Paired correlation |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {s.name} | {s.baseline_top1:.2f} | {s.within_group_correlation:.2f} "
        f"| {s.paired_correlation:.2f} |"
        for s in power.scenarios
    ]
    headers = " | ".join(f"MDE at alpha {alpha:g} (points)" for alpha in power.alphas)
    lines += [
        "",
        "## Minimum detectable Top-1 difference",
        "",
        f"| Scenario | False positive rate at alpha {first_alpha:g} | {headers} |"
        " Half-width at no difference (points) | "
        f"Power of the {100 * power.noninferiority_margin:g}-point S2 margin at no difference |",
        "| --- | --- | " + " | ".join("---" for _ in power.alphas) + " | --- | --- |",
    ]
    for scenario in power.scenarios:
        null = result.grid.filter(
            pl.col("scenario") == scenario.name,
            pl.col("alpha") == first_alpha,
            pl.col("difference") == 0,
        )
        detectable = result.minimum_detectable.filter(pl.col("scenario") == scenario.name)
        cells = " | ".join(
            _percent(detectable.filter(pl.col("alpha") == alpha)["minimum_detectable"][0])
            for alpha in power.alphas
        )
        rate = f"{null['power'][0]:.3f}" if null.height else "n/a"
        width = f"{100 * null['half_width'][0]:.1f}" if null.height else "n/a"
        margin = f"{null['noninferiority_power'][0]:.2f}" if null.height else "n/a"
        lines.append(f"| {scenario.name} | {rate} | {cells} | {width} | {margin} |")
    lines += [
        "",
        "Holm correction over five primary criteria tests the smallest p-value at alpha / 5, "
        "which is the second alpha column when it is 0.01.",
        "",
        "## How to read this",
        "",
        "- The test is the two-sided percentile interval of the paired group bootstrap (S1).",
        "- Power depends on the number of groups far more than on the number of cases, and "
        "repetitions of one run add little when they tend to succeed or fail together.",
        "- The simulation holds each case's outcomes as given, so it leaves out variation from "
        "training on different folds. Real power can only be lower.",
        "- S3 and S4 are not simulated here. A difference of differences has a standard error "
        "about sqrt(2 * (1 - r)) times that of one difference, where r correlates the clean and "
        "perturbed margins across groups.",
    ]
    if figure is not None:
        lines += ["", f"![Power versus true Top-1 difference]({figure})"]
    return "\n".join(lines) + "\n"


def write_power_report(config: EvalConfig, root: Path = Path()) -> PowerResult:
    """Run the analysis on the real group counts, and write the report and figure."""
    power = config.power
    sizes = group_sizes_from(root / power.groups_file)
    try:
        checksum = read_manifest(root / power.processed_dir).digest()
    except (DataError, FileNotFoundError) as error:
        raise EvalError(f"Cannot read the data manifest: {error}") from error
    result = power_analysis(sizes, power)
    output = root / power.output
    output.parent.mkdir(parents=True, exist_ok=True)
    primary = result.grid.filter(pl.col("alpha") == power.alphas[0])
    curves = [
        PowerCurve(
            scenario.name,
            part["difference"].to_numpy(),
            part["power"].to_numpy(),
        )
        for scenario in power.scenarios
        for part in [primary.filter(pl.col("scenario") == scenario.name)]
    ]
    stem = root / power.figure_stem
    power_curves(curves, power.target_power, stem)
    relative = Path(stem.with_suffix(".png")).resolve().relative_to(output.parent.resolve())
    text = render_report(result, config, checksum, relative.as_posix())
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return result
