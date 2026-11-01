"""Every table of the report, built only from result JSONs."""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascadelens.report.sources import (
    REPRODUCTION_LOG,
    ReportError,
    Store,
    ci,
    fraction,
    p_value,
    pct,
)

CLEAN = "clean"
S3_POINT = "P2:after_boost=3.0,near_boost=3.0,rate=0.3"
SHIFT_POINT = "P4:offset_seconds=-120"
RANKER = "ranker_augmented"
PLAIN = "ranker"
S3_NAMES = ("lightgbm_flat", "lightgbm_flat_masking", "frequency", "rule", "rule_upstream", "baro")
SUPERVISED = ("frequency", "lightgbm_flat")
UNSUPERVISED = ("rule", "rule_upstream")
CURVE_METHODS = (RANKER, PLAIN, "lightgbm_flat_masking", "lightgbm_flat", "frequency")
KEY_POINTS = (
    CLEAN,
    "P1:level=series,rate=0.3",
    "P2:after_boost=3.0,near_boost=3.0,rate=0.2",
    S3_POINT,
    "P2:after_boost=3.0,near_boost=3.0,rate=0.5",
    "P3:family=latency",
    "P3:family=memory",
    SHIFT_POINT,
    "P6:rate=0.3",
    "P8:method=correlation",
    "P11:factor=5",
    "P13:mode=rename,service_fraction=0.3",
)
SUITES = ("RE1-OB", "RE1-SS", "RE1-TT")
MARKER = re.compile(r"\[VERIFY[^\]]*\]|\bTODO\b")
OWN_FILES = ("docs/CLAIMS_TODO.md",)


@dataclass(frozen=True)
class Table:
    """A generated Markdown table and where it came from."""

    name: str
    title: str
    columns: tuple[str, ...]
    rows: Sequence[tuple[str, ...]]
    sources: tuple[str, ...]

    def markdown(self) -> str:
        """Return the table as Markdown."""
        head = "| " + " | ".join(self.columns) + " |"
        rule = "| " + " | ".join("---" for _ in self.columns) + " |"
        return "\n".join([head, rule, *("| " + " | ".join(row) + " |" for row in self.rows)])


def full_run(root: Path) -> str | None:
    """Return the log row of a completed full run from a fresh clone that passed, if any."""
    path = root / REPRODUCTION_LOG
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip("| ").split("|")]
        if len(cells) >= 6 and cells[1] == "full" and cells[-1] == "pass":
            return " ".join(cells[:2])
    return None


def _verdict(passed: bool) -> str:
    return "pass" if passed else "fail"


def _s3_rows(store: Store) -> list[tuple[str, ...]]:
    hypotheses = store("robustness").metrics["hypotheses"]
    holm = hypotheses["holm"]
    rows: list[tuple[str, ...]] = []
    for name in S3_NAMES:
        entry = hypotheses["S3"][name]
        rows.append(
            (
                f"S3 vs {name}",
                "Margin growth, P2 at 30% (simulated)",
                ci(entry),
                p_value(holm[f"S3 vs {name}"]),
                _verdict(entry["lower"] > 0),
            )
        )
    return rows


def s7_passed(store: Store) -> dict[str, bool]:
    """Return whether each reproduced method passed on every suite."""
    return {
        name: all(store(key).metrics["comparison"][suite]["passed"] for suite in SUITES)
        for key, name in (("baro", "baro"), ("lingam", "lingam_randomwalk"))
    }


def headline(store: Store) -> Table:
    """Return S1 to S7 with values, intervals and verdicts."""
    hypotheses = store("robustness").metrics["hypotheses"]
    ranker = hypotheses["ranker"]
    holm = hypotheses["holm"]
    calibration = store("calibration").metrics
    clean = calibration["results"][RANKER][CLEAN]
    run = full_run(store.root)
    passed = s7_passed(store)
    rows: list[tuple[str, ...]] = [
        (
            "S1",
            "Ranker minus frequency prior, Top-1 points, group_run",
            ci(ranker["S1"]),
            p_value(holm["S1"]),
            _verdict(ranker["S1"]["lower"] > 0),
        ),
        (
            "S2",
            f"Ranker minus {hypotheses['s2_comparator']['method']}, plus the 12-point margin",
            ci(ranker["S2"]),
            p_value(holm["S2"]),
            _verdict(ranker["S2"]["lower"] > 0),
        ),
        *_s3_rows(store),
        (
            "S4",
            "Top-1 at 50% coverage minus full coverage and random abstention",
            ci(clean["selective_gain"]),
            p_value(calibration["holm"]["S4"]),
            _verdict(calibration["verdicts"]["S4"]),
        ),
        (
            "S5 (secondary)",
            "ECE after minus ECE before calibration",
            fraction(clean["ece_difference"]),
            "-",
            _verdict(calibration["verdicts"]["S5"]),
        ),
        (
            "S6",
            "Fresh clone plus one command regenerates all tables on the real data",
            run if run else "no completed full run logged",
            "-",
            "pass" if run else "not demonstrated",
        ),
    ]
    for name, ok in passed.items():
        rows.append(
            (
                f"S7 {name} (observed)",
                "Reproduced Avg@5 within tolerance on all three suites",
                "see the reproduction table",
                "-",
                _verdict(ok),
            )
        )
    return Table(
        "headline",
        "Pre-registered criteria S1 to S7",
        ("Criterion", "Measure", "Result [95% CI]", "Holm p", "Verdict"),
        rows,
        ("robustness", "calibration", "baro", "lingam"),
    )


def leakage(store: Store) -> Table:
    """Return the random split against group_run."""
    metrics = store("leakage").metrics
    rows = []
    for method in sorted(metrics["scores"]):
        scores = metrics["scores"][method]
        rows.append(
            (
                method,
                ci(scores["group_run"]["Top-1"]),
                ci(scores["random"]["Top-1"]),
                ci(metrics["paired_differences"][method]["Top-1"]),
            )
        )
    columns = ("Method", "Top-1 group_run", "Top-1 random", "Random minus group_run")
    return Table("leakage", "Leakage demonstration (real cases)", columns, rows, ("leakage",))


def _score_row(row: dict[str, Any]) -> tuple[str, ...]:
    return (row["method"], ci(row["Top-1"]), ci(row["Top-3"]), ci(row["Avg@5"]))


def _clean_rows(store: Store, names: Iterable[str]) -> list[tuple[str, ...]]:
    rows = {r["method"]: r for r in store("baselines").metrics["group_run"]}
    rows |= {r["method"]: r for r in store("reproduction").metrics["group_run"]}
    return [_score_row(rows[name]) for name in names]


SCORE_COLUMNS = ("Method", "Top-1", "Top-3", "Avg@5")


def clean_supervised(store: Store) -> Table:
    """Return supervised methods on clean group_run."""
    rows = _clean_rows(store, SUPERVISED)
    rows += [_score_row(row) for row in store("ranker").metrics["group_run"]]
    sources = ("baselines", "ranker")
    title = "Clean supervised methods on group_run (real cases)"
    return Table("clean_supervised", title, SCORE_COLUMNS, rows, sources)


def clean_unsupervised(store: Store) -> Table:
    """Return unsupervised methods on clean group_run."""
    names = (*UNSUPERVISED, "baro", "lingam_randomwalk")
    title = "Clean unsupervised methods on group_run (real cases)"
    sources = ("baselines", "reproduction")
    return Table("clean_unsupervised", title, SCORE_COLUMNS, _clean_rows(store, names), sources)


def _grid(store: Store) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["method"], r["point"]): r for r in store("robustness").metrics["grid"]}


def robustness(store: Store, name: str = "robustness", points: Sequence[str] = KEY_POINTS) -> Table:
    """Return Top-1 by perturbation, with each row's realism label."""
    grid = _grid(store)
    rows = []
    for point in points:
        first = grid[(RANKER, point)]
        cells = [
            ci(grid[(method, point)]["Top-1"]) if (method, point) in grid else "-"
            for method in CURVE_METHODS
        ]
        rows.append((point, first["realism"], *cells))
    title = "Top-1 by perturbation (simulated unless labelled clean)"
    return Table(name, title, ("Point", "Label", *CURVE_METHODS), rows, ("robustness",))


def robustness_full(store: Store) -> Table:
    """Return every grid point."""
    points = [r["point"] for r in store("robustness").metrics["grid"] if r["method"] == RANKER]
    return robustness(store, "robustness_full", points)


def ablation(store: Store) -> Table:
    """Return the full ranker minus each ablated variant."""
    rows = [
        (row["variant"], ci(row["clean"]), ci(row["p2_30"]))
        for row in store("robustness").metrics["ablation"]
    ]
    columns = ("Variant", "Clean", "P2 at 30% (simulated)")
    return Table(
        "ablation", "Ablation: full minus variant, Top-1 points", columns, rows, ("robustness",)
    )


def calibration(store: Store) -> Table:
    """Return calibration and abstention on clean, simulated and held-out cases."""
    results = store("calibration").metrics["results"]
    rows = []
    for variant, entry in results.items():
        slices = (
            (CLEAN, entry[CLEAN]),
            ("P2 30% (simulated)", entry[S3_POINT]),
            ("P4 -120 s (simulated)", entry[SHIFT_POINT]),
            ("lofo (held-out)", entry["lofo"]["pooled"]),
            ("loso (held-out)", entry["loso"]["pooled"]),
        )
        for label, item in slices:
            rows.append(
                (
                    variant,
                    label,
                    fraction(item["ece_before"]),
                    fraction(item["ece_after"]),
                    fraction(item["ece_difference"]),
                    ci(item["selective"]["0.5"]),
                    f"{item['aurc']:.3f} / {item['aurc_random']:.3f} / {item['aurc_oracle']:.3f}",
                    pct(item["realised_coverage"]),
                )
            )
    columns = (
        "Method",
        "Slice",
        "ECE before",
        "ECE after",
        "ECE difference",
        "Top-1 at 50%",
        "AURC / random / oracle",
        "Test coverage %",
    )
    return Table("calibration", "Calibration and abstention", columns, rows, ("calibration",))


def selective(store: Store) -> Table:
    """Return selective Top-1 by coverage."""
    entry = store("calibration").metrics["results"]
    variants = list(entry)
    coverages = list(entry[variants[0]][CLEAN]["selective"])
    rows = [
        (pct(float(coverage)), *(ci(entry[v][CLEAN]["selective"][coverage]) for v in variants))
        for coverage in coverages
    ]
    title = "Selective Top-1 by coverage, clean group_run"
    return Table("selective", title, ("Coverage %", *variants), rows, ("calibration",))


def coverage_drift(store: Store) -> Table:
    """Return realised test coverage of the CascadeLens ranker, with the range across folds."""
    entry = store("calibration").metrics["results"][RANKER]
    rows = []
    for key, item in entry.items():
        if key in ("lofo", "loso"):
            folds = [fold["realised_coverage"] for fold in item["folds"].values()]
            pooled = item["pooled"]["realised_coverage"]
            label = "held-out"
        else:
            folds, pooled = item["fold_coverage"], item["realised_coverage"]
            label = item["tag"]
        rows.append((key, label, pct(pooled), f"{pct(min(folds))} to {pct(max(folds))}"))
    columns = ("Slice", "Label", "Test coverage %", "Range across folds")
    return Table("coverage_drift", "Coverage against a 50% target", columns, rows, ("calibration",))


def _failure(store: Store, split: str, name: str, title: str, column: str) -> Table:
    ranker = [r for r in store("ranker").metrics["folds"] if r["split"] == split]
    flat = {
        r["fold"]: r
        for r in store("baselines").metrics["folds"]
        if r["split"] == split and r["method"] == "lightgbm_flat"
    }
    rows = [
        (row["fold"], pct(row["floor"]), ci(row["Top-1"]), ci(flat[row["fold"]]["Top-1"]))
        for row in ranker
        if row["method"] == "two_stage_ranker"
    ]
    columns = (column, "Random floor", "two_stage_ranker Top-1", "lightgbm_flat Top-1")
    return Table(name, title, columns, rows, ("ranker", "baselines"))


def failure_fault(store: Store) -> Table:
    """Return held-out results by fault type."""
    return _failure(store, "lofo", "failure_fault", "By unseen fault type (held-out)", "Fault")


def failure_system(store: Store) -> Table:
    """Return held-out results by system."""
    return _failure(store, "loso", "failure_system", "By unseen system (held-out)", "System")


def reproduction(store: Store) -> Table:
    """Return S7: reproduced Avg@5 against the published means."""
    rows = []
    for key, name in (("baro", "baro"), ("lingam", "lingam_randomwalk")):
        comparison = store(key).metrics["comparison"]
        for suite in SUITES:
            entry = comparison[suite]
            rows.append(
                (
                    name,
                    suite,
                    f"{entry['published_mean']:.3f}",
                    f"{entry['our_mean']:.3f}",
                    f"{entry['difference']:+.3f}",
                    _verdict(entry["passed"]),
                )
            )
    columns = ("Method", "Suite", "Published Avg@5", "Reproduced", "Difference", "Within tolerance")
    return Table(
        "reproduction", "Reproduction of published results", columns, rows, ("baro", "lingam")
    )


def probes(store: Store) -> Table:
    """Return the four leakage probes."""
    identity = store("probe_identity").metrics
    graph = store("probe_graph").metrics
    mask = store("probe_mask").metrics
    window = store("probe_window").metrics
    position = window["injection_position"]
    rows = [
        (
            "Identity",
            f"drop, graph and structural shuffled: {ci(identity['graph_and_structural']['drop'])}",
            _verdict(identity["passed"]),
        ),
        (
            "Graph-only",
            f"Top-1 group_run {ci(graph['group_run']['top1'])}, loso {ci(graph['loso']['top1'])}",
            _verdict(graph["passed"]),
        ),
        (
            "Window leak",
            f"no cliff at zero; {pct(position['share_within_2pct_of_middle'])}% of injections "
            "sit within 2% of the window's middle",
            _verdict(window["passed"]),
        ),
        (
            "Mask-only",
            f"Top-1 {ci(mask['top1'])} on clean data, uniform floor {pct(mask['uniform_floor'])}",
            "flagged" if mask["flagged"] else "not flagged",
        ),
    ]
    sources = ("probe_identity", "probe_graph", "probe_mask", "probe_window")
    return Table("probes", "Leakage probes", ("Probe", "Result", "Outcome"), rows, sources)


def markers(root: Path, generated: Mapping[str, str]) -> Table:
    """Return every open marker under docs/ and in the README, using generated text where given."""
    texts = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in (root / "docs").rglob("*.md")
    }
    texts |= {name: text for name, text in generated.items() if name.endswith(".md")}
    counts: dict[tuple[str, str], int] = {}
    for name, text in texts.items():
        if name in OWN_FILES:
            continue
        for found in MARKER.findall(text):
            counts[(name, found)] = counts.get((name, found), 0) + 1
    rows = [(f"`{name}`", f"`{mark}`", str(n)) for (name, mark), n in sorted(counts.items())]
    return Table("markers", "Open markers", ("File", "Marker", "Count"), rows, ())


BUILDERS: tuple[Callable[[Store], Table], ...] = (
    headline,
    leakage,
    clean_supervised,
    clean_unsupervised,
    robustness,
    robustness_full,
    ablation,
    calibration,
    selective,
    coverage_drift,
    failure_fault,
    failure_system,
    reproduction,
    probes,
)


def all_tables(store: Store) -> dict[str, Table]:
    """Return every table by name."""
    tables = {}
    for build in BUILDERS:
        table = build(store)
        if table.name in tables:
            raise ReportError(f"Two tables are named {table.name}")
        tables[table.name] = table
    return tables
