"""Generate docs/data_audit.md from whatever is really in the raw directory.

The raw inventory assumes no layout at all, so it still reports the structure when the loader
refuses it. Every other section comes from loading the cases, one at a time.
"""

import itertools
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cascadelens.config import DataConfig
from cascadelens.data.errors import DataError, Issue
from cascadelens.data.loaders.rcaeval import INJECT_TIME_FILE, METRIC_FILES, iter_cases
from cascadelens.data.schema import Case
from cascadelens.data.validate import CaseValidator

# Case counts stated in the RCAEval README; a mismatch is reported, not fixed.
DOCUMENTED_CASES = {
    "RE1-OB": 125,
    "RE1-SS": 125,
    "RE1-TT": 125,
    "RE2-OB": 90,
    "RE2-SS": 90,
    "RE2-TT": 90,
    "RE3-OB": 30,
    "RE3-SS": 30,
    "RE3-TT": 30,
}
TINY_GROUP = 3
IMBALANCE_RATIO = 3.0
HIGH_MISSING_FRACTION = 0.1
# The injection sits at a near-fixed position when this share of cases is within the tolerance.
TIED_TOLERANCE = 0.02
TIED_SHARE = 0.9
TOPOLOGY_WORDS = ("topolog", "graph", "edge", "dependenc")


@dataclass(frozen=True)
class RawInventory:
    """What is on disk, gathered without assuming any layout."""

    exists: bool
    top_level: tuple[str, ...] = ()
    extensions: tuple[tuple[str, int], ...] = ()
    file_names: tuple[tuple[str, int], ...] = ()
    depths: tuple[tuple[int, int], ...] = ()
    example_files: tuple[str, ...] = ()
    example_header: str | None = None
    example_inject_time: str | None = None
    topology_candidates: tuple[str, ...] = ()


def inventory_raw(raw_dir: Path) -> RawInventory:
    """Summarize the file tree under ``raw_dir``."""
    if not raw_dir.is_dir():
        return RawInventory(exists=False)
    files = sorted((p for p in raw_dir.rglob("*") if p.is_file()), key=Path.as_posix)
    relative = [path.relative_to(raw_dir) for path in files]
    header = next((path for path in files if path.name in METRIC_FILES), None)
    inject = next((path for path in files if path.name == INJECT_TIME_FILE), None)
    return RawInventory(
        exists=True,
        top_level=tuple(sorted(path.name for path in raw_dir.iterdir())),
        extensions=tuple(sorted(Counter(path.suffix or "(none)" for path in files).items())),
        file_names=tuple(Counter(path.name for path in files).most_common(15)),
        depths=tuple(sorted(Counter(len(path.parts) for path in relative).items())),
        example_files=tuple(path.as_posix() for path in relative[:5]),
        example_header=_first_line(header),
        example_inject_time=_first_line(inject),
        topology_candidates=tuple(
            path.as_posix()
            for path in relative
            if any(word in path.name.lower() for word in TOPOLOGY_WORDS)
        ),
    )


def _first_line(path: Path | None) -> str | None:
    if path is None:
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.readline().strip()


@dataclass(frozen=True)
class CaseStats:
    """The per-case numbers behind every table; holds no samples."""

    case_id: str
    suite: str
    system: str
    fault_type: str
    run_id: str
    root_cause_service: str
    services: int
    metrics_per_service: tuple[int, ...]
    missing_fraction: float
    window_seconds: int
    sample_step: int
    inject_position: float
    window_start: int
    window_end: int
    has_topology: bool


def case_stats(case: Case) -> CaseStats:
    """Reduce a case to its audit numbers."""
    series = case.metrics.select("service", "metric").unique()
    per_service = series.group_by("service").len().sort("service")["len"].to_list()
    timestamps = case.metrics["timestamp"].unique().sort()
    steps = timestamps.diff().drop_nulls().to_list()
    expected = series.height * timestamps.len()
    length = case.window_end - case.window_start
    return CaseStats(
        case_id=case.case_id,
        suite=case.suite,
        system=case.system,
        fault_type=case.fault_type,
        run_id=case.run_id,
        root_cause_service=case.root_cause_service,
        services=series["service"].n_unique(),
        metrics_per_service=tuple(per_service),
        missing_fraction=1 - case.metrics["value"].drop_nulls().len() / expected if expected else 1,
        window_seconds=length,
        sample_step=int(statistics.median(steps)) if steps else 0,
        inject_position=(case.inject_time - case.window_start) / length if length else 0.0,
        window_start=case.window_start,
        window_end=case.window_end,
        has_topology=case.topology is not None,
    )


@dataclass
class AuditData:
    """Everything the report is rendered from."""

    inventory: RawInventory
    suites: tuple[str, ...]
    stats: list[CaseStats] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    layout_error: str | None = None


def collect(config: DataConfig, root: Path = Path()) -> AuditData:
    """Load and validate every selected case, keeping only statistics."""
    raw_dir = root / config.raw_dir
    data = AuditData(inventory=inventory_raw(raw_dir), suites=tuple(config.suites))
    validator = CaseValidator()
    try:
        for item in iter_cases(raw_dir, config.suites):
            if isinstance(item, Issue):
                data.issues.append(item)
            else:
                data.stats.append(case_stats(item))
                data.issues.extend(validator.check(item))
    except DataError as error:
        data.layout_error = str(error)
    return data


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _spread(values: Sequence[float]) -> tuple[str, str, str]:
    """Return (min, median, max) as display strings."""
    return (
        f"{min(values):.4g}",
        f"{statistics.median(values):.4g}",
        f"{max(values):.4g}",
    )


def _by(stats: Iterable[CaseStats], key: Callable[[CaseStats], str]) -> dict[str, list[CaseStats]]:
    groups: dict[str, list[CaseStats]] = defaultdict(list)
    for stat in stats:
        groups[key(stat)].append(stat)
    return dict(sorted(groups.items()))


def _suite_spread(
    stats: list[CaseStats], value: Callable[[CaseStats], Sequence[float]]
) -> list[tuple[str, str, str, str]]:
    return [
        (suite, *_spread([number for stat in group for number in value(stat)]))
        for suite, group in _by(stats, lambda stat: stat.suite).items()
    ]


def overlapping_cases(stats: Sequence[CaseStats]) -> int:
    """Count neighbouring cases of a suite whose recordings overlap in time."""
    total = 0
    for group in _by(stats, lambda stat: stat.suite).values():
        ordered = sorted(group, key=lambda stat: stat.window_start)
        total += sum(
            later.window_start <= earlier.window_end
            for earlier, later in itertools.pairwise(ordered)
        )
    return total


def find_concerns(data: AuditData) -> list[str]:
    """List everything suspicious, in the order it should be read."""
    concerns: list[str] = []
    stats = data.stats
    if not data.inventory.exists:
        concerns.append(
            "**No raw data found.** Nothing in this report has been checked on real data."
        )
    if data.layout_error:
        concerns.append(f"The loader refused the raw layout: {data.layout_error}")
    errors = [issue for issue in data.issues if issue.severity == "error"]
    if errors:
        concerns.append(f"{len(errors)} validation or load error(s); see the section above.")
    for suite, group in _by(stats, lambda stat: stat.suite).items():
        documented = DOCUMENTED_CASES.get(suite)
        if documented is not None and documented != len(group):
            concerns.append(f"{suite} has {len(group)} cases; the README states {documented}.")
    pairs = Counter((stat.system, stat.fault_type) for stat in stats)
    for (system, fault), count in sorted(pairs.items()):
        if count < TINY_GROUP:
            concerns.append(f"Tiny group: {system}/{fault} has {count} case(s).")
    for system, group in _by(stats, lambda stat: stat.system).items():
        labels = Counter(stat.root_cause_service for stat in group)
        if max(labels.values()) > IMBALANCE_RATIO * min(labels.values()):
            concerns.append(f"Label imbalance in {system}: {dict(sorted(labels.items()))}.")
    for suite, group in _by(stats, lambda stat: stat.suite).items():
        positions = [stat.inject_position for stat in group]
        middle = statistics.median(positions)
        near = sum(abs(position - middle) <= TIED_TOLERANCE for position in positions)
        if len(group) > 1 and near / len(group) >= TIED_SHARE:
            concerns.append(
                f"**Window tied to inject_time in {suite}:** in {near} of {len(group)} cases the "
                f"injection falls within {TIED_TOLERANCE:.0%} of {middle:.1%} of the recording, so "
                "the window bounds reveal inject_time even though they are computed without it. "
                "The window-leak probe (PRD 6.5) must cover this."
            )
        lengths = sorted({stat.window_seconds for stat in group})
        if len(lengths) > 1:
            by_fault = {
                fault: statistics.median(stat.window_seconds for stat in members)
                for fault, members in _by(group, lambda stat: stat.fault_type).items()
            }
            if len(set(by_fault.values())) > 1:
                concerns.append(
                    f"**Window length tracks fault type in {suite}** (median seconds: "
                    f"{', '.join(f'{fault}={value:g}' for fault, value in by_fault.items())}). "
                    "Recording length alone separates fault types, which is a leak for any model "
                    "that sees window bounds, and for LOFO. Crop all cases to a common length."
                )
            else:
                concerns.append(
                    f"Window length varies within {suite}: {len(lengths)} distinct lengths from "
                    f"{lengths[0]} to {lengths[-1]} s, not by fault type."
                )
    repaired = {i.case_id for i in data.issues if i.check == "load_repair"}
    if repaired:
        concerns.append(
            f"{len(repaired)} case(s) needed a raw-file repair (see the warnings table): "
            "untimed tail rows or a second table on another clock were dropped."
        )
    empty_series = {i.case_id for i in data.issues if i.check == "nan_only_series"}
    if empty_series:
        concerns.append(
            f"{len(empty_series)} case(s) hold series with no values on non-label services. They "
            "are kept as missing telemetry; per-metric availability will differ across cases, "
            "and a model could read that as a collection artifact (mask-feature probe)."
        )
    unused = [
        f"{name} ({count})"
        for name, count in data.inventory.file_names
        if name not in (*METRIC_FILES, INJECT_TIME_FILE)
    ]
    if unused:
        concerns.append(f"Raw files the loader never reads: {', '.join(unused)}.")
    largest_run = max(Counter(stat.run_id for stat in stats).values(), default=0)
    if largest_run > 1:
        concerns.append(
            f"Runs hold up to {largest_run} repetitions, which are near-duplicate candidates. "
            "Split by run_id, never by case."
        )
    overlaps = overlapping_cases(stats)
    if overlaps:
        concerns.append(f"{overlaps} neighbouring case pair(s) overlap in time within a suite.")
    missing = [stat for stat in stats if stat.missing_fraction > HIGH_MISSING_FRACTION]
    if missing:
        concerns.append(
            f"{len(missing)} case(s) miss over {HIGH_MISSING_FRACTION:.0%} of their samples."
        )
    if stats and not any(stat.has_topology for stat in stats):
        concerns.append(
            "No topology for any case. Graph features and P6-P8 need an inferred or hand-built "
            "call graph, which would be a separate decision."
        )
    if data.inventory.topology_candidates:
        sample = list(data.inventory.topology_candidates[:5])
        concerns.append(f"Possible topology files that the loader ignores: {sample}.")
    return concerns or ["None found."]


def _render_inventory(inventory: RawInventory, raw_dir: str) -> list[str]:
    if not inventory.exists:
        return [f"`{raw_dir}` does not exist, so no structure could be observed."]
    return [
        f"Top level of `{raw_dir}`: {', '.join(f'`{name}`' for name in inventory.top_level)}",
        _table(["Extension", "Files"], inventory.extensions),
        _table(["File name", "Files"], inventory.file_names),
        _table(["Path depth", "Files"], inventory.depths),
        "Example files:\n\n" + "\n".join(f"- `{path}`" for path in inventory.example_files),
        f"Example metrics header: `{(inventory.example_header or 'none')[:300]}`",
        f"Example `{INJECT_TIME_FILE}` first line: `{inventory.example_inject_time}`",
    ]


def _render_tables(stats: list[CaseStats]) -> list[str]:
    systems = _by(stats, lambda stat: stat.system)
    runs = Counter(stat.run_id for stat in stats)
    position = _suite_spread(stats, lambda stat: [stat.inject_position])
    return [
        "## Systems",
        _table(
            ["System", "Suites", "Cases"],
            [
                (name, ", ".join(sorted({stat.suite for stat in group})), len(group))
                for name, group in systems.items()
            ],
        ),
        "## Fault types",
        _table(["Fault type", "Cases"], sorted(Counter(s.fault_type for s in stats).items())),
        "## Injected services per system",
        _table(
            ["System", "Service", "Cases"],
            [
                (system, service, count)
                for (system, service), count in sorted(
                    Counter((s.system, s.root_cause_service) for s in stats).items()
                )
            ],
        ),
        "## Cases per (system, fault type)",
        _table(
            ["System", "Fault type", "Cases"],
            [
                (system, fault, count)
                for (system, fault), count in sorted(
                    Counter((s.system, s.fault_type) for s in stats).items()
                )
            ],
        ),
        "## Runs per system",
        _table(
            ["System", "Runs"],
            [(name, len({stat.run_id for stat in group})) for name, group in systems.items()],
        ),
        "## Cases per run",
        "Runs with more than one case hold near-duplicate candidates.",
        _table(["Cases in run", "Runs"], sorted(Counter(runs.values()).items())),
        "## Services per case",
        _table(["Suite", "Min", "Median", "Max"], _suite_spread(stats, lambda s: [s.services])),
        "## Metrics per service",
        _table(
            ["Suite", "Min", "Median", "Max"],
            _suite_spread(stats, lambda s: s.metrics_per_service),
        ),
        "## Missing-sample fraction",
        _table(
            ["Suite", "Min", "Median", "Max"],
            _suite_spread(stats, lambda s: [s.missing_fraction]),
        ),
        "## Window length (seconds)",
        _table(
            ["Suite", "Min", "Median", "Max"],
            _suite_spread(stats, lambda s: [s.window_seconds]),
        ),
        "## Sampling step (seconds)",
        _table(["Suite", "Min", "Median", "Max"], _suite_spread(stats, lambda s: [s.sample_step])),
        "## Injection position inside the window",
        "0 is the start of the window and 1 the end. A narrow range means that the window "
        "bounds reveal `inject_time` even though they are not computed from it.",
        _table(["Suite", "Min", "Median", "Max"], position),
        "## Topology availability per system",
        _table(
            ["System", "Cases with topology", "Cases"],
            [
                (name, sum(stat.has_topology for stat in group), len(group))
                for name, group in systems.items()
            ],
        ),
    ]


def render(data: AuditData, raw_dir: str) -> str:
    """Render the report. The same inputs give identical text."""
    blocks = [
        "# RCAEval data audit",
        "Generated by `cascadelens data audit`. Do not edit by hand.",
        f"Selected suites: {', '.join(data.suites) or 'none'}. Cases loaded: {len(data.stats)}.",
        "## Raw structure found",
        *_render_inventory(data.inventory, raw_dir),
        "## How fields are derived",
        "- **Labels:** the `<service>_<fault>` directory name, split on the last underscore.\n"
        "- **run_id:** `<suite>/<service>_<fault>`, shared by all repetitions of that fault.\n"
        "- **Window:** first and last timestamp of the recording, computed from the `time` "
        "column before `inject_time.txt` is read. No direct dependence on `inject_time`.",
    ]
    if data.stats:
        blocks += _render_tables(data.stats)
    errors = [issue for issue in data.issues if issue.severity == "error"]
    warnings = [issue for issue in data.issues if issue.severity == "warning"]
    blocks.append("## Validation failures")
    blocks.append(
        _table(["Case", "Check", "Reason"], [(i.case_id, i.check, i.reason) for i in errors])
        if errors
        else "None."
        if data.stats
        else "Not run: no cases were loaded."
    )
    if warnings:
        checks = sorted({issue.check for issue in warnings})
        blocks.append("## Logged warnings")
        blocks.append(
            _table(
                ["Check", "Cases", "Example case", "Example reason"],
                [
                    (
                        check,
                        len({i.case_id for i in warnings if i.check == check}),
                        next(i for i in warnings if i.check == check).case_id,
                        next(i for i in warnings if i.check == check).reason,
                    )
                    for check in checks
                ],
            )
        )
    repairs = [issue for issue in warnings if issue.check == "load_repair"]
    if repairs:
        blocks.append("## Raw-file repairs")
        blocks.append(_table(["Case", "Repair"], [(i.case_id, i.reason) for i in repairs]))
    blocks.append("## Concerns")
    blocks.append("\n".join(f"- {concern}" for concern in find_concerns(data)))
    return "\n\n".join(blocks) + "\n"


def run_audit(config: DataConfig, output: Path, root: Path = Path()) -> AuditData:
    """Write the report to ``output`` and return what it was built from."""
    data = collect(config, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render(data, config.raw_dir))
    return data
