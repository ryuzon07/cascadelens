"""Errors and the structured issue record shared by loading and validation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """One problem with one case. Errors fail the build; warnings are only logged."""

    case_id: str
    check: str
    reason: str
    severity: Severity = "error"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.case_id}: {self.check}: {self.reason}"


class DataError(Exception):
    """Base class for ingestion failures reported to the user."""


class RawLayoutError(DataError):
    """The raw directory does not have the layout the loader was written for."""


class CaseLoadError(DataError):
    """One raw case could not be read."""


class DataValidationError(DataError):
    """One or more cases failed validation."""

    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues = tuple(issues)
        details = "\n".join(str(issue) for issue in self.issues)
        super().__init__(f"{len(self.issues)} validation failure(s):\n{details}")
