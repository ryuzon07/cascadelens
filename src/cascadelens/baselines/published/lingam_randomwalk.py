"""ICA-LiNGAM causal discovery with a random walk, from RCAEval (ASE 2024, row ICALiNGAM-RW).

The method reads no anomaly time and receives the whole window, so nothing about the injection is
passed to it. ICA and the walk draw from global generators, which are seeded per case.

It runs on the key metric set only. On all metrics a Sock Shop case took 263 s, and Train Ticket
has more columns than rows, which ICA-LiNGAM cannot use. This departs from identical inputs.
"""

from cascadelens.baselines.base import CaseHandle
from cascadelens.baselines.published.adapter import TIME, WideFrame, call_library
from cascadelens.baselines.published.base import LIBRARY_DATASET, PublishedBaseline, check_output


class LingamRandomWalk(PublishedBaseline):
    """Rank series by random-walk visits on an ICA-LiNGAM graph."""

    name = "lingam_randomwalk"

    def scores(self, case: CaseHandle, wide: WideFrame) -> dict[str, float]:
        """Turn the ranked columns into scores, as minus the position."""
        columns = [column for column in wide.frame.columns if column != TIME]
        output = call_library(
            "lingam_randomwalk",
            wide.frame,
            inject_time=None,
            dataset=LIBRARY_DATASET,
            dk_select_useful=False,
            n_iter=len(columns),
        )
        check_output(output)
        return {column: -float(index) for index, column in enumerate(output["ranks"])}
