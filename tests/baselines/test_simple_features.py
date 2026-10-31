import polars as pl
import pytest
from baseline_factory import SAMPLES, START, make_metrics

from cascadelens.features.simple import (
    FEATURE_COLUMNS,
    SignalSettings,
    metric_family,
    service_summary,
)

SETTINGS = SignalSettings()
END = START + SAMPLES - 1


@pytest.mark.parametrize(
    ("metric", "family"),
    [
        ("cpu", "cpu"),
        ("mem", "memory"),
        ("latency", "latency"),
        ("error", "error"),
        ("load", "traffic"),
        ("istio-latency-99", "latency"),
        ("istio-error-total", "error"),
        ("container-network-receive-errors-total", "error"),
        ("container-network-receive-bytes-total", "network"),
        ("container-memory-rss", "memory"),
        ("container-fs-reads-total", "disk"),
        ("istio-request-total", "traffic"),
        ("mystery", "other"),
    ],
)
def test_metric_families(metric: str, family: str) -> None:
    assert metric_family(metric) == family


def summary(onsets: dict[str, int | None]) -> pl.DataFrame:
    return service_summary(make_metrics(onsets), START, END, SETTINGS)


def test_onset_is_found_after_the_baseline_and_ordered() -> None:
    table = summary({"a": 100, "b": 120, "c": None})

    onset = dict(zip(table["service"], table["onset"], strict=True))
    assert onset["a"] < onset["b"]
    assert onset["c"] is None


def test_a_cumulative_counter_alone_is_not_an_anomaly() -> None:
    assert summary({"quiet": None})["max_z"][0] < 10


def test_features_separate_the_shifted_service() -> None:
    table = summary({"a": 100, "b": None})

    assert table["cpu_max_z"][0] > 10 * table["cpu_max_z"][1]
    assert table["cpu_relative_change"][0] > table["cpu_relative_change"][1]
    assert table["cpu_anomaly_duration"][0] > 0.5
    assert table["cpu_peak_magnitude"][0] > table["cpu_peak_magnitude"][1]


def test_absent_families_stay_null_and_every_column_exists() -> None:
    table = summary({"a": 100})

    assert set(FEATURE_COLUMNS) <= set(table.columns)
    assert table["latency_max_z"][0] is None


def test_duration_does_not_depend_on_recording_length() -> None:
    frame = make_metrics({"a": 100})
    longer = frame.with_columns(timestamp=START + (pl.col("timestamp") - START) * 4)

    first = service_summary(frame, START, END, SETTINGS)
    second = service_summary(longer, START, START + (END - START) * 4, SETTINGS)

    assert first["cpu_anomaly_duration"][0] == pytest.approx(second["cpu_anomaly_duration"][0])


def flat_zero_baseline(service: str, jump: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "service": [service] * 100,
            "metric": ["error"] * 100,
            "timestamp": list(range(100)),
            "value": [0.0] * 90 + [jump] * 10,
        },
        schema_overrides={"timestamp": pl.Int64},
    )


def test_a_flat_zero_baseline_scores_in_raw_units_and_does_not_hit_the_cap() -> None:
    assert service_summary(flat_zero_baseline("a", 2.0), 0, 99, SETTINGS)["max_z"][0] == 2.0


def test_flat_baselines_do_not_tie_services_at_the_cap() -> None:
    frame = pl.concat([flat_zero_baseline("a", 2.0), flat_zero_baseline("b", 5.0)])

    assert service_summary(frame, 0, 99, SETTINGS)["max_z"].to_list() == [2.0, 5.0]


def test_a_service_without_samples_is_kept() -> None:
    frame = make_metrics({"a": 100}).vstack(
        pl.DataFrame(
            {"service": ["gone"], "metric": ["cpu"], "timestamp": [START], "value": [None]},
            schema=make_metrics({"a": 100}).schema,
        )
    )

    table = service_summary(frame, START, END, SETTINGS)

    assert table["service"].to_list() == ["a", "gone"]
    assert table["max_z"][1] is None
