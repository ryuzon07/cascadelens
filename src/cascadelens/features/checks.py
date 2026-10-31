"""A leakage self-check on feature tables; input for the identity probe (PRD 6.5).

It reports and judges nothing. A flagged column may be fine, or may be how a model recognises a
service instead of diagnosing it. ``to_model_matrix`` is the hard gate.

Input is a stacked table with ``case_id``, ``service``, features, and optionally ``system``, which
scopes the fingerprint check: service names repeat across cases of a system, and a value only
identifies a service among the services it could be confused with.

- **exact fingerprint:** within a system, each service always has one value across its cases, and
  no two services share a value, for at least ``MINIMUM_SERVICES`` services.
- **near fingerprint:** not exact, but service identity explains at least ``NEAR_SHARE`` of the
  column's variance within the system.
"""

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from cascadelens.features.context import KEYS

ROW_ORDER_CORRELATION = 0.999
NEAR_SHARE = 0.99
MINIMUM_SERVICES = 3
MINIMUM_ROWS = 3
SYSTEM = "system"


@dataclass(frozen=True)
class CheckReport:
    """Everything the self-check found."""

    cases: int
    rows: int
    nan_share: dict[str, float]
    all_nan: list[str]
    constant: list[str]
    row_order: list[str]
    exact_fingerprints: dict[str, list[str]]
    near_fingerprints: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


def _nan_as_null(table: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return table.with_columns(pl.col(columns).fill_nan(None))


def _row_order_columns(table: pl.DataFrame, columns: list[str]) -> list[str]:
    """Return columns that follow row order exactly in every case where that can be measured."""
    numbered = table.with_columns(row=pl.int_range(pl.len()).over("case_id").cast(pl.Float64))
    flagged = []
    for column in columns:
        correlation = (
            numbered.drop_nulls(column)
            .group_by("case_id")
            .agg(r=pl.corr("row", column), n=pl.len())
            .filter(pl.col("n") >= MINIMUM_ROWS, pl.col("r").is_finite())
        )
        if correlation.height and bool((correlation["r"].abs() >= ROW_ORDER_CORRELATION).all()):
            flagged.append(column)
    return flagged


def _fingerprints(
    table: pl.DataFrame, columns: list[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    exact: dict[str, list[str]] = {}
    near: dict[str, list[str]] = {}
    scoped = table if SYSTEM in table.columns else table.with_columns(pl.lit("all").alias(SYSTEM))
    for (system,), part in sorted(scoped.group_by(SYSTEM), key=lambda item: str(item[0])):
        for column in columns:
            rows = part.drop_nulls(column).select("service", column)
            values = rows.group_by("service", maintain_order=True).agg(
                mean=pl.col(column).mean(),
                spread=pl.col(column).n_unique(),
                count=pl.len(),
            )
            if values.height < MINIMUM_SERVICES or rows.height <= values.height:
                continue
            means = values["mean"].to_numpy()
            total = float(np.var(rows[column].to_numpy()))
            if total == 0:
                continue
            is_stable = bool((values["spread"] == 1).all())
            if is_stable and values["mean"].n_unique() == values.height:
                exact.setdefault(str(system), []).append(column)
                continue
            weights = values["count"].to_numpy()
            between = float(np.average((means - rows[column].mean()) ** 2, weights=weights))
            if between / total >= NEAR_SHARE:
                near.setdefault(str(system), []).append(column)
    return exact, near


def check_table(table: pl.DataFrame) -> CheckReport:
    """Run every check on a stacked feature table."""
    columns = [c for c in table.columns if c not in (*KEYS, SYSTEM)]
    nulled = _nan_as_null(table, columns)
    rows = max(table.height, 1)
    missing = {c: nulled[c].null_count() / rows for c in columns}
    filled = [c for c in columns if missing[c] < 1]
    exact, near = _fingerprints(nulled, filled)
    return CheckReport(
        cases=table["case_id"].n_unique(),
        rows=table.height,
        nan_share=missing,
        all_nan=[c for c in columns if missing[c] == 1],
        constant=[c for c in filled if nulled[c].drop_nulls().n_unique() == 1],
        row_order=_row_order_columns(nulled, filled),
        exact_fingerprints=exact,
        near_fingerprints=near,
    )
