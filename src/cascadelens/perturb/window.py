"""Window perturbations: P4 and P5.

The window moves and the data stays. Samples outside the new window remain in the case, and instants
the new window needs but the recording lacks are added as missing rows on every series. Feature code
does not crop to the window, so use ``crop_to_window`` where the outside must not be read.
"""

import numpy as np
import polars as pl
from pydantic import Field

from cascadelens.config import StrictModel
from cascadelens.data.schema import METRICS_SCHEMA, Case
from cascadelens.perturb.base import SERIES, Outcome, Perturbation, Stage, with_fields


class ShiftParams(StrictModel):
    """P4. Seconds added to both window bounds."""

    offset_seconds: int


class NoiseParams(StrictModel):
    """P5. Each bound moves by a whole number of seconds, uniform on the range, independently."""

    max_jitter_seconds: int = Field(default=30, ge=1)


def _padding(case: Case, start: int, end: int) -> pl.DataFrame:
    """Return missing rows for instants of the new window beyond the recording."""
    instants = case.metrics["timestamp"].unique().sort()
    steps = instants.diff().drop_nulls()
    step = int(steps.min()) if len(steps) else 1  # type: ignore[arg-type]
    first, last = int(instants[0]), int(instants[-1])
    added = [*range(first - step, start - 1, -step), *range(last + step, end + 1, step)]
    series = case.metrics.select(*SERIES).unique(maintain_order=True)
    if not added:
        return pl.DataFrame(schema=METRICS_SCHEMA)
    times = pl.DataFrame({"timestamp": added}, schema={"timestamp": pl.Int64})
    return series.join(times, how="cross").with_columns(pl.lit(None, pl.Float64).alias("value"))


def move_window(case: Case, start: int, end: int) -> Outcome:
    """Return ``case`` with new bounds, padded where it has no data."""
    padding = _padding(case, start, end)
    metrics = case.metrics
    if not padding.is_empty():
        metrics = pl.concat([metrics, padding.select(metrics.columns)]).sort(*SERIES, "timestamp")
    return Outcome(
        with_fields(case, window_start=start, window_end=end, metrics=metrics),
        details={
            "original_window_start": case.window_start,
            "original_window_end": case.window_end,
            "padded_rows": padding.height,
        },
    )


class WindowShift(Perturbation[ShiftParams]):
    """P4: move both bounds by the same offset."""

    id = "P4"
    name = "window_shift"
    stage = Stage.WINDOW
    schema = ShiftParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Shift the window."""
        offset = self.params.offset_seconds
        return move_window(case, case.window_start + offset, case.window_end + offset)


class WindowNoise(Perturbation[NoiseParams]):
    """P5: jitter each bound independently."""

    id = "P5"
    name = "window_noise"
    stage = Stage.WINDOW
    schema = NoiseParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Jitter the window."""
        limit = self.params.max_jitter_seconds
        first, second = rng.integers(-limit, limit + 1, size=2).tolist()
        return move_window(case, case.window_start + first, case.window_end + second)
