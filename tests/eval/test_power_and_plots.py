from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from cascadelens.cli import app
from cascadelens.config import load_config
from cascadelens.data.io import Manifest, write_manifest
from cascadelens.eval.calibration import (
    oracle_abstention,
    random_abstention,
    reliability_table,
    risk_coverage,
)
from cascadelens.eval.config import EvalConfig, PowerConfig, PowerScenario
from cascadelens.eval.errors import EvalError
from cascadelens.eval.plots import (
    CurveSeries,
    PowerCurve,
    bars_with_ci,
    power_curves,
    reliability_diagram,
    risk_coverage_curves,
    robustness_curves,
)
from cascadelens.eval.power import (
    estimate_power,
    group_sizes_from,
    minimum_detectable_difference,
    power_analysis,
    render_report,
    simulate_paired_outcomes,
    write_power_report,
)

RUNNER = CliRunner()
SCENARIO = PowerScenario(
    name="test", baseline_top1=0.5, within_group_correlation=0.3, paired_correlation=0.3
)
SIZES = [5] * 75
CONFIGS = Path(__file__).parents[2] / "configs"


def power_at(difference: float, sizes: list[int], simulations: int = 400) -> float:
    grid = estimate_power(sizes, SCENARIO, [difference], [0.05], simulations, 500, 0)
    return float(grid["power"][0])


def small_config(**overrides: object) -> PowerConfig:
    fields: dict[str, object] = {
        "seed": 0,
        "n_simulations": 100,
        "n_resamples": 200,
        "target_power": 0.8,
        "alphas": [0.05, 0.01],
        "noninferiority_margin": 0.03,
        "differences": [0.0, 0.2, 0.4],
        "scenarios": [SCENARIO.model_dump()],
    }
    return PowerConfig.model_validate({**fields, **overrides})


def test_shipped_config_is_valid() -> None:
    config = load_config(CONFIGS / "eval.yaml", EvalConfig)

    assert config.primary_criteria == ["S1", "S2", "S3", "S4", "S7"]
    assert config.power.alphas[0] == 0.05


def test_simulation_hits_the_marginal_top1() -> None:
    ranker, baseline = simulate_paired_outcomes(
        SIZES, SCENARIO, 0.2, 4000, np.random.default_rng(0)
    )

    assert ranker.mean() == pytest.approx(0.7, abs=0.01)
    assert baseline.mean() == pytest.approx(0.5, abs=0.01)


def test_simulation_produces_the_requested_correlations() -> None:
    scenario = PowerScenario(
        name="corr", baseline_top1=0.5, within_group_correlation=0.6, paired_correlation=0.7
    )
    ranker, baseline = simulate_paired_outcomes(
        [2] * 500, scenario, 0.0, 200, np.random.default_rng(1)
    )
    first, second = ranker[:, ::2].ravel().astype(float), ranker[:, 1::2].ravel().astype(float)

    within = np.corrcoef(first, second)[0, 1]
    paired = np.corrcoef(ranker.ravel().astype(float), baseline.ravel().astype(float))[0, 1]

    # Binary correlation at a median cut is (2 / pi) * asin(latent correlation).
    assert within == pytest.approx(2 / np.pi * np.arcsin(0.6), abs=0.02)
    assert paired == pytest.approx(2 / np.pi * np.arcsin(0.7), abs=0.02)


def test_zero_within_group_correlation_gives_independent_cases() -> None:
    scenario = SCENARIO.model_copy(update={"within_group_correlation": 0.0})

    ranker, _ = simulate_paired_outcomes([2] * 500, scenario, 0.0, 200, np.random.default_rng(2))

    assert np.corrcoef(ranker[:, ::2].ravel().astype(float), ranker[:, 1::2].ravel().astype(float))[
        0, 1
    ] == pytest.approx(0, abs=0.02)


def test_simulation_reproduces_with_a_seed() -> None:
    first = simulate_paired_outcomes(SIZES, SCENARIO, 0.1, 5, np.random.default_rng(3))
    second = simulate_paired_outcomes(SIZES, SCENARIO, 0.1, 5, np.random.default_rng(3))

    assert np.array_equal(first[0], second[0])


def test_power_increases_with_effect_size() -> None:
    assert power_at(0.05, SIZES) < power_at(0.15, SIZES) < power_at(0.3, SIZES)


def test_power_increases_with_the_number_of_groups() -> None:
    assert power_at(0.1, [5] * 20) < power_at(0.1, [5] * 75) < power_at(0.1, [5] * 300)


def test_false_positive_rate_is_near_alpha() -> None:
    assert power_at(0.0, SIZES, simulations=2000) == pytest.approx(0.05, abs=0.025)


def test_more_clustering_costs_power_at_the_same_case_count() -> None:
    assert power_at(0.1, [5] * 75) > power_at(0.1, [25] * 15)


def test_power_is_reproducible() -> None:
    args = (SIZES, SCENARIO, [0.1], [0.05], 50, 100, 7)

    assert estimate_power(*args).equals(estimate_power(*args))


def test_grid_skips_differences_past_a_top1_of_one() -> None:
    grid = estimate_power(SIZES, SCENARIO, [0.1, 0.6], [0.05], 20, 50, 0)

    assert grid["difference"].to_list() == [0.1]


def test_power_at_the_ceiling_works() -> None:
    assert estimate_power(SIZES, SCENARIO, [0.5], [0.05], 20, 50, 0)["power"][0] == 1.0


def test_a_stricter_alpha_has_less_power() -> None:
    grid = estimate_power(SIZES, SCENARIO, [0.1], [0.05, 0.01], 300, 500, 0)

    assert grid["power"][0] > grid["power"][1]


def test_noninferiority_power_rises_with_the_margin() -> None:
    narrow = estimate_power(SIZES, SCENARIO, [0.0], [0.05], 300, 500, 0, 0.03)
    wide = estimate_power(SIZES, SCENARIO, [0.0], [0.05], 300, 500, 0, 0.2)

    assert narrow["noninferiority_power"][0] < wide["noninferiority_power"][0]


def test_interpolated_minimum_detectable_difference() -> None:
    assert minimum_detectable_difference([0, 0.1, 0.2], [0.05, 0.6, 1.0], 0.8) == pytest.approx(
        0.15
    )


def test_no_minimum_detectable_difference_when_never_reached() -> None:
    assert minimum_detectable_difference([0, 0.1], [0.05, 0.5], 0.8) is None


def test_analysis_finds_a_difference_and_reports_the_counts() -> None:
    result = power_analysis(SIZES, small_config())

    assert (result.n_groups, result.n_cases) == (75, 375)
    assert result.minimum_detectable.height == 2
    assert 0 < result.minimum_detectable["minimum_detectable"][0] < 0.4


def test_report_states_the_assumptions_and_is_stable() -> None:
    config = EvalConfig(power=small_config())
    result = power_analysis(SIZES, config.power)

    text = render_report(result, config, "abc", "figures/p.png")

    assert text == render_report(
        power_analysis(SIZES, config.power), config, "abc", "figures/p.png"
    )
    assert "assumed, not measured" in text
    assert "| test | 0.50 | 0.30 | 0.30 |" in text
    assert "`abc`" in text


def test_one_group_is_refused() -> None:
    with pytest.raises(EvalError, match="two groups"):
        power_analysis([5], small_config())


def test_impossible_top1_is_refused() -> None:
    with pytest.raises(EvalError, match="outside"):
        simulate_paired_outcomes(SIZES, SCENARIO, 0.6, 1, np.random.default_rng(0))


def project_with_groups(root: Path) -> None:
    groups = root / "data" / "splits" / "group_run"
    groups.mkdir(parents=True)
    pl.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(40)],
            "group_id": [f"g{i // 5}" for i in range(40)],
        }
    ).write_parquet(groups / "groups.parquet")
    write_manifest(Manifest(source="s", suites=(), files=()), root / "data" / "processed")


def test_group_sizes_come_from_the_groups_file(tmp_path: Path) -> None:
    project_with_groups(tmp_path)

    sizes = group_sizes_from(tmp_path / "data" / "splits" / "group_run" / "groups.parquet")

    assert sizes.tolist() == [5] * 8


def test_missing_groups_file_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="splits build"):
        write_power_report(EvalConfig(power=small_config()), tmp_path)


def test_missing_manifest_fails_cleanly(tmp_path: Path) -> None:
    project_with_groups(tmp_path)
    (tmp_path / "data" / "processed" / "manifest.json").unlink()

    with pytest.raises(EvalError, match="data manifest"):
        write_power_report(EvalConfig(power=small_config()), tmp_path)


def test_report_files_are_byte_stable(tmp_path: Path) -> None:
    project_with_groups(tmp_path)
    config = EvalConfig(power=small_config())
    paths = [
        tmp_path / "docs" / "power_analysis.md",
        tmp_path / "docs" / "figures" / "power_curve.png",
        tmp_path / "docs" / "figures" / "power_curve.svg",
    ]

    write_power_report(config, tmp_path)
    first = [path.read_bytes() for path in paths]
    write_power_report(config, tmp_path)

    assert [path.read_bytes() for path in paths] == first


def test_cli_writes_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    project_with_groups(tmp_path)
    config = tmp_path / "eval.yaml"
    config.write_text(
        "power:\n  seed: 0\n  n_simulations: 50\n  n_resamples: 100\n  target_power: 0.8\n"
        "  alphas: [0.05]\n  noninferiority_margin: 0.03\n  differences: [0.0, 0.3]\n"
        "  scenarios:\n    - {name: x, baseline_top1: 0.5, within_group_correlation: 0.3,"
        " paired_correlation: 0.3}\n",
        encoding="utf-8",
    )

    result = RUNNER.invoke(app, ["eval", "power", "--config", str(config)])

    assert result.exit_code == 0, result.output
    text = (tmp_path / "docs" / "power_analysis.md").read_text(encoding="utf-8")
    assert "![Power versus true Top-1 difference](figures/power_curve.png)" in text


def test_cli_reports_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "eval.yaml"
    config.write_text(
        "power:\n  seed: 0\n  n_simulations: 5\n  n_resamples: 5\n  target_power: 0.8\n"
        "  alphas: [0.05]\n  noninferiority_margin: 0.03\n  differences: [0.0, 0.3]\n"
        "  scenarios:\n    - {name: x, baseline_top1: 0.5, within_group_correlation: 0.3,"
        " paired_correlation: 0.3}\n",
        encoding="utf-8",
    )

    result = RUNNER.invoke(app, ["eval", "power", "--config", str(config)])

    assert result.exit_code == 1
    assert "eval power failed" in result.output


def assert_saved(paths: tuple[Path, Path]) -> None:
    png, svg = paths
    assert png.read_bytes().startswith(b"\x89PNG")
    assert b"<svg" in svg.read_bytes()


def test_robustness_curves_save_both_formats(tmp_path: Path) -> None:
    series = [
        CurveSeries("a", [0, 1], [0.8, 0.6], [0.7, 0.5], [0.9, 0.7]),
        CurveSeries("b", [0, 1], [0.7, 0.4], [0.6, 0.3], [0.8, 0.5]),
    ]

    assert_saved(robustness_curves(series, "Loss rate", "Top-1", "simulated", tmp_path / "r"))


def test_reliability_diagram_saves(tmp_path: Path) -> None:
    table = reliability_table([0.25, 0.25, 0.85], [1, 0, 1])

    assert_saved(reliability_diagram(table, "Reliability", tmp_path / "rel"))


def test_risk_coverage_plot_saves(tmp_path: Path) -> None:
    conf, correct = [0.9, 0.5, 0.1], [1, 0, 1]
    paths = risk_coverage_curves(
        {"ranker": risk_coverage(conf, correct)},
        random_abstention(correct),
        oracle_abstention(correct),
        "Risk",
        tmp_path / "risk",
    )

    assert_saved(paths)


def test_bar_chart_saves(tmp_path: Path) -> None:
    assert_saved(
        bars_with_ci(["a", "b"], [0.5, 0.6], [0.4, 0.5], [0.6, 0.7], "Top-1", "T", tmp_path / "b")
    )


def test_power_curve_saves(tmp_path: Path) -> None:
    curves = [PowerCurve("a", [0.0, 0.1], [0.05, 0.9]), PowerCurve("b", [0.0, 0.1], [0.05, 0.7])]

    assert_saved(power_curves(curves, 0.8, tmp_path / "p"))


def test_too_many_series_are_refused(tmp_path: Path) -> None:
    curves = [PowerCurve(str(i), [0.0], [0.0]) for i in range(9)]

    with pytest.raises(EvalError, match="small multiples"):
        power_curves(curves, 0.8, tmp_path / "p")
