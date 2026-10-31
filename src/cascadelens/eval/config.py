"""Typed settings for ``configs/eval.yaml``."""

from pydantic import Field

from cascadelens.config import StrictModel
from cascadelens.eval.significance import PRIMARY_CRITERIA


class PowerScenario(StrictModel):
    """One set of assumptions about the outcomes being compared."""

    name: str
    baseline_top1: float = Field(gt=0, lt=1)
    within_group_correlation: float = Field(
        ge=0, lt=1, description="Latent correlation between cases of one group, per method"
    )
    paired_correlation: float = Field(
        ge=0, lt=1, description="Latent correlation between the two methods on the same case"
    )


class PowerConfig(StrictModel):
    """The simulation behind the pre-registered power analysis."""

    seed: int = Field(ge=0)
    processed_dir: str = "data/processed"
    groups_file: str = "data/splits/group_run/groups.parquet"
    output: str = "docs/power_analysis.md"
    figure_stem: str = "docs/figures/power_curve"
    n_simulations: int = Field(ge=1)
    n_resamples: int = Field(ge=1)
    target_power: float = Field(gt=0, lt=1)
    alphas: list[float] = Field(min_length=1, description="The first drives the figure")
    noninferiority_margin: float = Field(gt=0)
    differences: list[float] = Field(min_length=2)
    scenarios: list[PowerScenario] = Field(min_length=1)


class EvalConfig(StrictModel):
    """The evaluation settings."""

    primary_criteria: list[str] = Field(default_factory=lambda: list(PRIMARY_CRITERIA))
    power: PowerConfig
