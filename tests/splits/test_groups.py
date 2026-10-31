from pathlib import Path

import numpy as np
import polars as pl
import pytest
from split_factory import REPLAY_SUFFIX, identity_groups, make_record, near_duplicate_dataset

from cascadelens.data.io import metrics_path, read_index
from cascadelens.data.schema import METRICS_SCHEMA, CaseRecord
from cascadelens.splits.groups import (
    Signature,
    UnionFind,
    build_groups,
    case_signature,
    stored_signature,
)

BINS = 8
RAMP = np.arange(BINS, dtype=np.float64)


def by_case(signatures: dict[str, Signature]) -> "SignatureLookup":
    return SignatureLookup(signatures)


class SignatureLookup:
    """A signature source that records which cases it was asked about."""

    def __init__(self, signatures: dict[str, Signature]) -> None:
        self.signatures = signatures
        self.requested: list[str] = []

    def __call__(self, record: CaseRecord) -> Signature:
        self.requested.append(record.case_id)
        return self.signatures[record.case_id]


def test_union_find_is_transitive_and_roots_at_the_smallest() -> None:
    forest = UnionFind(["c", "b", "a", "z"])

    forest.union("c", "b")
    forest.union("b", "a")

    assert {forest.find(item) for item in "abc"} == {"a"}
    assert forest.find("z") == "z"


def test_near_duplicate_runs_merge() -> None:
    records = [make_record("a1", "run-a"), make_record("b1", "run-b")]
    source = by_case({"a1": RAMP, "b1": RAMP * 2 + 1})

    grouping = build_groups(records, source, 0.98, BINS)

    assert set(grouping.group_map.values()) == {"run-a"}
    assert grouping.summary.merged[0].run_ids == ("run-a", "run-b")
    assert grouping.summary.merged[0].cases == 2
    assert grouping.summary.max_cross_run_correlation == pytest.approx(1.0)


def test_merging_is_transitive_across_runs() -> None:
    records = [make_record(f"{name}1", f"run-{name}") for name in "abc"]
    wave = np.sin(RAMP)
    source = by_case({"a1": wave, "b1": wave, "c1": wave})

    assert len(set(build_groups(records, source, 0.98, BINS).group_map.values())) == 1


def test_uncorrelated_runs_stay_apart() -> None:
    records = [make_record("a1", "run-a"), make_record("b1", "run-b")]
    source = by_case({"a1": RAMP, "b1": np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)})

    grouping = build_groups(records, source, 0.98, BINS)

    assert grouping.group_map == identity_groups(records)
    assert grouping.summary.merged == ()
    assert any("No run pair reached" in note for note in grouping.summary.notes)


@pytest.mark.parametrize(
    "other",
    [
        {"system": "beta"},
        {"fault_type": "mem"},
        {"root_cause_service": "db"},
    ],
)
def test_runs_differing_in_key_fields_are_never_compared(other: dict[str, str]) -> None:
    records = [make_record("a1", "run-a"), make_record("b1", "run-b", **other)]
    source = by_case({"a1": RAMP, "b1": RAMP})

    grouping = build_groups(records, source, 0.98, BINS)

    assert grouping.summary.merged == ()
    assert source.requested == []
    assert any("no-op on this dataset" in note for note in grouping.summary.notes)


def test_flat_signatures_are_not_duplicates() -> None:
    records = [make_record("a1", "run-a"), make_record("b1", "run-b")]
    flat = np.zeros(BINS)
    source = by_case({"a1": flat, "b1": flat})

    grouping = build_groups(records, source, 0.98, BINS)

    assert grouping.summary.merged == ()
    assert grouping.summary.undefined_signatures == 2
    assert grouping.summary.max_cross_run_correlation is None


def test_cases_of_one_run_stay_together_and_are_reported() -> None:
    records = [make_record("a1", "run-a"), make_record("a2", "run-a")]

    grouping = build_groups(records, by_case({}), 0.98, BINS)

    assert grouping.group_map == {"a1": "run-a", "a2": "run-a"}
    assert grouping.summary.group_sizes == {"2": 1}
    assert grouping.summary.runs_with_several_cases == 1
    assert "does not identify one injection" in grouping.summary.notes[0]


def test_signature_handles_scale_gaps_and_flat_series() -> None:
    rows = [("api", "cpu", step, float(step)) for step in range(8)]
    rows += [("db", "mem", step, 1000.0 * step) for step in range(8)]
    rows += [("db", "flat", step, 5.0) for step in range(8)]
    rows += [("db", "gap", step, None) for step in range(8)]  # type: ignore[misc]
    frame = pl.DataFrame(rows, schema=METRICS_SCHEMA, orient="row")

    signature = case_signature(frame.lazy(), 0, 8, BINS)

    assert signature.shape == (BINS,)
    assert np.all(np.diff(signature) > 0)
    assert np.isfinite(signature).all()


def test_signature_does_not_depend_on_row_order() -> None:
    rows = [("api", "cpu", step, float(step % 3)) for step in range(16)]
    frame = pl.DataFrame(rows, schema=METRICS_SCHEMA, orient="row")

    # Parallel means can differ in the last bit, so exact equality would be flaky.
    np.testing.assert_allclose(
        case_signature(frame.lazy(), 0, 16, BINS),
        case_signature(frame.reverse().lazy(), 0, 16, BINS),
        rtol=1e-12,
    )


def test_clustering_merges_the_planted_replay_only(tmp_path: Path) -> None:
    _, replayed = near_duplicate_dataset(tmp_path)
    records = read_index(tmp_path)

    grouping = build_groups(records, stored_signature(tmp_path, 32), 0.98, 32)

    assert [cluster.run_ids for cluster in grouping.summary.merged] == [
        (replayed, f"{replayed}{REPLAY_SUFFIX}")
    ]
    assert grouping.summary.groups == grouping.summary.runs - 1
    assert grouping.group_map["alpha_api_cpu_replay_0"] == replayed
    assert grouping.group_map["beta_api_cpu_0"] == "beta/api_cpu"


def test_only_candidate_cases_are_read_from_disk(tmp_path: Path) -> None:
    near_duplicate_dataset(tmp_path)
    records = read_index(tmp_path)
    metrics_path(tmp_path, "gamma_db_mem_0").unlink()

    grouping = build_groups(records, stored_signature(tmp_path, 32), 0.98, 32)

    assert grouping.summary.candidate_run_pairs == 1
