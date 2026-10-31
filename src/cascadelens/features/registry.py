"""Feature groups by name, and the one function that builds a case table.

Output has ``case_id``, ``service`` and then numeric features, one row per service sorted by
service. Missing data stays NaN and is never imputed. ``service`` and ``case_id`` are join keys:
``to_model_matrix`` removes them, and refuses anything that could identify a service.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import polars as pl

from cascadelens.features import graph, mask, signal, temporal
from cascadelens.features.config import FeaturesConfig, GroupName
from cascadelens.features.context import KEYS, CaseContext, FeatureError, FeatureInput

GroupFunction = Callable[[CaseContext], pl.DataFrame]


@dataclass(frozen=True)
class FeatureGroup:
    """One switchable group of columns."""

    name: GroupName
    function: GroupFunction
    prefix: str
    columns: tuple[str, ...]


REGISTRY: dict[GroupName, FeatureGroup] = {
    item.name: item
    for item in (
        FeatureGroup("signal", signal.signal_features, signal.PREFIX, signal.COLUMNS),
        FeatureGroup("temporal", temporal.temporal_features, temporal.PREFIX, temporal.COLUMNS),
        FeatureGroup("graph", graph.graph_features, graph.GRAPH_PREFIX, graph.GRAPH_COLUMNS),
        FeatureGroup(
            "structural",
            graph.structural_features,
            graph.STRUCTURAL_PREFIX,
            graph.STRUCTURAL_COLUMNS,
        ),
        FeatureGroup("mask", mask.mask_features, mask.PREFIX, mask.COLUMNS),
    )
}
PREFIXES = tuple(group.prefix for group in REGISTRY.values())
FORBIDDEN_WORDS = frozenset({"service", "services", "name", "id", "idx", "index", "hash", "label"})


def feature_columns(groups: Sequence[GroupName]) -> list[str]:
    """Return the columns of the selected groups, in registry order."""
    return [c for name, group in REGISTRY.items() if name in groups for c in group.columns]


def build_case_features(case: FeatureInput, config: FeaturesConfig) -> pl.DataFrame:
    """Build the feature table of one case, looking at that case alone."""
    context = CaseContext.build(case, config.signal, config.graph, config.mask)
    table = context.frame()
    for name, group in REGISTRY.items():
        if name in config.groups:
            part = group.function(context).select("service", *group.columns)
            table = table.join(part, on="service", how="left", maintain_order="left")
    columns = feature_columns(config.groups)
    return table.with_columns(
        pl.lit(case.case_id).alias("case_id"), pl.col(columns).fill_null(float("nan"))
    ).select(*KEYS, *columns)


def to_model_matrix(table: pl.DataFrame) -> pl.DataFrame:
    """Return only the feature columns, refusing any column that could identify a service."""
    features = [column for column in table.columns if column not in KEYS]
    for column in features:
        if not column.startswith(PREFIXES):
            raise FeatureError(f"Column {column!r} belongs to no feature group")
        words = set(column.lower().split("_"))
        if words & FORBIDDEN_WORDS:
            raise FeatureError(f"Column {column!r} looks like it identifies a service")
        if not table.schema[column].is_float():
            raise FeatureError(f"Column {column!r} is {table.schema[column]}, not numeric float")
    if not features:
        raise FeatureError("The table has no feature columns")
    return table.select(features)
