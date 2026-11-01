"""Load the synthetic samples in demo_data/ as demo incidents. Benchmark data never comes here.

Titles describe the symptom a responder would see, so that they do not give away the simulated
fault. ``simulation`` blocks are dropped before validation, and each case takes a neutral ID.
"""

import json
from pathlib import Path
from typing import Any

from cascadelens.api.incidents import IncidentService
from cascadelens.api.schemas import IncidentCreate, Severity, Status

SAMPLES_DIR = Path("demo_data")
DEMO_INCIDENTS: tuple[tuple[str, str, Severity, Status, str | None], ...] = (
    ("catalog-latency", "Frontend latency above normal", "medium", "needs_feedback", "on-call"),
    ("cart-memory-no-topology", "Slow storefront, no call graph", "medium", "open", None),
    ("checkout-unlabelled", "Frontend requests slowing", "high", "open", "on-call"),
    ("payment-silent", "Checkout latency, telemetry gaps", "critical", "open", "on-call"),
    ("payment-cpu", "Checkout latency above normal", "high", "open", "on-call"),
)


def demo_body(name: str, title: str, severity: Severity, root: Path) -> IncidentCreate:
    """Return one sample as a create request, without its simulated fault."""
    path = root / SAMPLES_DIR / f"{name}.json"
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("simulation", None)
    return IncidentCreate.model_validate(
        {**raw, "case_id": "sample", "title": title, "severity": severity}
    )


def seed_demo(service: IncidentService, root: Path = Path()) -> int:
    """Load every sample once, and return how many incidents were added."""
    if service.store.has_demo():
        return 0
    for name, title, severity, status, owner in DEMO_INCIDENTS:
        body = demo_body(name, title, severity, root)
        service.create(body, status=status, owner=owner, is_demo=True)
    return len(DEMO_INCIDENTS)
