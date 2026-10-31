"""Typed settings for ``configs/features.yaml``."""

from typing import Literal

from pydantic import Field

from cascadelens.config import StrictModel
from cascadelens.features.context import GraphSettings, MaskSettings, TopologySettings
from cascadelens.features.simple import SignalSettings

GroupName = Literal["signal", "temporal", "graph", "structural", "mask"]


class FeaturesConfig(StrictModel):
    """Which feature groups to build, and how. Its hash names the output directory."""

    groups: list[GroupName] = Field(min_length=1)
    signal: SignalSettings = SignalSettings()
    graph: GraphSettings = GraphSettings()
    mask: MaskSettings = MaskSettings()
    topology: TopologySettings = TopologySettings()
    processed_dir: str = "data/processed"
    output_dir: str = "data/features"
