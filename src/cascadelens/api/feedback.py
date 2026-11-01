"""Verdicts from people, and the match rates they support.

A confirmed service here is what a person said after an incident. It is separate from any benchmark
label. Each incident is its own bootstrap group, because nothing links incidents the way runs link
benchmark cases. Rates describe the labelled incidents only, which are not a random sample.
"""

from collections import Counter

from cascadelens.api.failures import ApiFailure
from cascadelens.api.schemas import (
    Decision,
    FeedbackCreate,
    FeedbackLabel,
    HistoryReport,
    MatchRate,
    Outcome,
    WeekOutcomes,
)
from cascadelens.api.store import Feedback, Incident, incident_id, now
from cascadelens.eval.bootstrap import group_bootstrap_ci
from cascadelens.inference import DiagnoseResult

LABELLED_BY = "local user"
FIRST_MONDAY = 345_600
WEEK = 604_800
TOP_K = 3


def outcome_of(result: DiagnoseResult, actual: str | None, correct: bool) -> Outcome:
    """Return where the confirmed service sat in the ranking."""
    if correct:
        return "top_1"
    if actual is None:
        return "unknown"
    top = [item.service for item in result.ranking[:TOP_K]]
    return "top_3" if actual in top else "missed"


def new_feedback(
    body: FeedbackCreate,
    incident: Incident,
    result: DiagnoseResult,
    decision: Decision,
    services: list[str],
) -> Feedback:
    """Return a verdict on the latest diagnosis, refusing one that contradicts itself."""
    assert incident.seq is not None
    predicted = result.ranking[0].service
    actual = body.actual_service
    if actual is not None and actual not in services:
        raise ApiFailure(422, "unknown_service", "That service is not part of this incident")
    if body.correct and actual not in (None, predicted):
        message = "A correct diagnosis cannot name a different service"
        raise ApiFailure(422, "conflicting_feedback", message)
    if not body.correct and actual == predicted:
        message = "An incorrect diagnosis cannot confirm the predicted service"
        raise ApiFailure(422, "conflicting_feedback", message)
    return Feedback(
        incident_seq=incident.seq,
        created_at=now(),
        correct=body.correct,
        actual_service=predicted if body.correct else actual,
        predicted_service=predicted,
        outcome=outcome_of(result, actual, body.correct),
        confidence=result.confidence,
        decision=decision,
        note=body.note,
        labelled_by=LABELLED_BY,
    )


def label(feedback: Feedback, incident: Incident) -> FeedbackLabel:
    """Return a stored verdict for the API."""
    assert incident.seq is not None
    return FeedbackLabel.model_validate(
        {
            **feedback.model_dump(exclude={"id", "incident_seq"}),
            "incident_id": incident_id(incident.seq),
            "title": incident.title,
            "is_demo": incident.is_demo,
        }
    )


def _rate(values: list[float], groups: list[str]) -> MatchRate | None:
    if not values:
        return None
    found = group_bootstrap_ci(values, groups)
    return MatchRate(
        estimate=found.estimate, lower=found.lower, upper=found.upper, labels=len(values)
    )


def _weeks(labels: list[FeedbackLabel]) -> list[WeekOutcomes]:
    counts: dict[int, Counter[str]] = {}
    for item in labels:
        start = FIRST_MONDAY + (item.created_at - FIRST_MONDAY) // WEEK * WEEK
        counts.setdefault(start, Counter())[item.outcome] += 1
    return [
        WeekOutcomes(
            week_start=start,
            top_1=found["top_1"],
            top_3=found["top_3"],
            missed=found["missed"],
            unknown=found["unknown"],
        )
        for start, found in sorted(counts.items())
    ]


def history(rows: list[tuple[Feedback, Incident]]) -> HistoryReport:
    """Return labels, weekly outcomes and match rates. An unknown cause has no top-3 answer."""
    items = [label(feedback, incident) for feedback, incident in rows]
    known = [item for item in items if item.outcome != "unknown"]
    return HistoryReport(
        items=items,
        weeks=_weeks(items),
        top_1=_rate(
            [float(item.outcome == "top_1") for item in items],
            [item.incident_id for item in items],
        ),
        top_3=_rate(
            [float(item.outcome in ("top_1", "top_3")) for item in known],
            [item.incident_id for item in known],
        ),
        includes_demo=any(item.is_demo for item in items),
    )
