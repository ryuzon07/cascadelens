import polars as pl
import pytest
from case_factory import make_case
from pydantic import ValidationError

from cascadelens.data.schema import CaseRecord, TopologyEdge, normalize_service_name


def test_valid_case_is_accepted() -> None:
    assert make_case().record().root_cause_service == "db"


def test_topology_is_nullable() -> None:
    assert make_case(topology=None).topology is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"case_id": "Has Spaces"},
        {"case_id": ""},
        {"system": ""},
        {"repetition": -1},
        {"inject_time": "soon"},
        {"unknown_field": 1},
    ],
    ids=str,
)
def test_invalid_fields_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_case(**overrides)


@pytest.mark.parametrize(
    "metrics",
    [
        pl.DataFrame({"service": ["a"]}),
        pl.DataFrame(
            {"service": ["a"], "metric": ["m"], "timestamp": [1.5], "value": [1.0]},
        ),
    ],
    ids=["missing-columns", "float-timestamp"],
)
def test_metrics_schema_is_enforced(metrics: pl.DataFrame) -> None:
    with pytest.raises(ValidationError, match="metrics must have schema"):
        make_case(metrics=metrics)


def test_case_is_immutable() -> None:
    with pytest.raises(ValidationError):
        make_case().inject_time = 5  # type: ignore[misc]


def test_equals_detects_metric_differences() -> None:
    case = make_case()
    changed = case.metrics.with_columns(pl.col("value") + 1)

    assert case.equals(make_case())
    assert not case.equals(make_case(metrics=changed))


def test_inject_time_is_documented_as_ground_truth_only() -> None:
    description = CaseRecord.model_fields["inject_time"].description or ""

    assert "GROUND TRUTH ONLY" in description
    assert "never a model input" in description


def test_edge_requires_both_ends() -> None:
    with pytest.raises(ValidationError):
        TopologyEdge(caller="", callee="db")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("adservice", "adservice"), ("Front_End", "front-end"), (" ts ui ", "ts-ui")],
)
def test_normalize_service_name(raw: str, expected: str) -> None:
    assert normalize_service_name(raw) == expected


def test_normalize_rejects_blank_names() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_service_name("   ")
