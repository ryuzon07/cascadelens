"""The single-stage ablation: one LambdaRank model over all services, with no retrieval (PRD 8.3).

If it matches the two-stage system within the paired interval, retrieval adds no measurable value
and the simpler pipeline is preferred.
"""

from typing import ClassVar

from cascadelens.models.ranker import TwoStageRanker


class SingleStageRanker(TwoStageRanker):
    """Rank every service of a case in one model."""

    name = "single_stage_ranker"
    retrieves: ClassVar[bool] = False
