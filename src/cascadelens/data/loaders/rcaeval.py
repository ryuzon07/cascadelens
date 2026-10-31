"""Read RCAEval metrics into Case objects.

Layout. RCAEval's sources disagree, so one is chosen and anything else is refused:

- ``main.py`` globs ``<suite>/<service>_<fault>/<repetition>/data.csv`` (falling back to
  ``simple_metrics.csv``) with ``inject_time.txt`` beside it. This is the layout read here,
  because it is the only one backed by parsing code.
- The README describes flat ``{benchmark}_{service}_{fault}_{instance}/metrics.json`` directories.
- The utility module also mentions a Parquet conversion.

The chosen layout was confirmed against the downloaded RE1 suites (375 cases, one ``data.csv``
and one ``inject_time.txt`` each). ``cascadelens data audit`` reports what is really on disk.

Labels. The service and fault come from the ``<service>_<fault>`` directory name, split on the
last underscore. There is no separate label file.

run_id. ``<suite>/<service>_<fault>``: every repetition of the same fault on the same service
in the same suite shares one run_id, and the repetition directory is kept in ``repetition``.
The dataset has no finer session identifier, and a per-repetition id would make each group a
single case, which would give grouped splits no protection against near-duplicate repetitions.

Windows. ``window_start`` and ``window_end`` are the first and last timestamps of the recording.
They are computed from the ``time`` column alone, before ``inject_time.txt`` is opened, so they
have no direct dependence on ``inject_time``. They can still depend on it indirectly if the
recordings were cut around the injection at a fixed offset; the audit reports that offset.
RCAEval's own evaluation window (rows before and after ``inject_time``) is deliberately not
used here because it does read ``inject_time``.

Timestamps. The ``time`` column and ``inject_time.txt`` are taken to be whole Unix seconds. A
fractional value or a value too large for seconds fails the case instead of being rounded.

Raw defects found in the real RE1 files, each handled without guessing and reported as a
``load_repair`` warning on the case:

- Trailing rows with metric values but no timestamp: dropped (RCAEval's own filters drop them).
- A repeated ``time`` column in another number format: compared numerically, then dropped.
- A repeated ``time`` column that disagrees with the first: a second table on another clock.
  Its columns are dropped rather than mis-dated.
- ``simple_data.csv`` beside ``data.csv`` in the SS and TT suites is never read. ``main.py``
  looks for ``simple_metrics.csv``, which does not exist in these files, so the fallback in
  ``METRIC_FILES`` is inert.
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from polars.exceptions import InvalidOperationError, PolarsError

from cascadelens.data.errors import CaseLoadError, Issue, RawLayoutError
from cascadelens.data.schema import Case, normalize_service_name

SYSTEMS = {"OB": "online-boutique", "SS": "sock-shop", "TT": "train-ticket"}
METRIC_FILES = ("data.csv", "simple_metrics.csv")
INJECT_TIME_FILE = "inject_time.txt"
TIME_COLUMN = "time"
TIMESTAMP_COLUMN = "timestamp"
LOAD_CHECK = "load"
REPAIR_CHECK = "load_repair"
# 2286-11-20 in Unix seconds; anything larger is milliseconds or nanoseconds.
MAX_UNIX_SECONDS = 10_000_000_000

_SUITE_RE = re.compile(r"^(RE\d+)-([A-Z]+)$")
_CASE_DIR_RE = re.compile(r"^(?P<service>.+)_(?P<fault>[^_]+)$")
_REPEATED_TIME_RE = re.compile(rf"^{TIME_COLUMN}(_duplicated_\d+|\.\d+)$")
_UNNAMED_COLUMN_RE = re.compile(r"^(|column_\d+|Unnamed: \d+)$")


@dataclass(frozen=True)
class CaseSource:
    """Where one case lives on disk, as parsed from its path."""

    suite: str
    service: str
    fault: str
    repetition: int
    path: Path

    @property
    def label(self) -> str:
        """Return a name for error messages, usable before a Case exists."""
        return f"{self.suite}/{self.service}_{self.fault}/{self.repetition}"


def system_of(suite: str) -> str:
    """Return the application name for a suite such as ``RE1-OB``."""
    match = _SUITE_RE.match(suite)
    if match is None or match.group(2) not in SYSTEMS:
        raise RawLayoutError(f"Unknown suite {suite!r}; expected RE<n>-<{'|'.join(SYSTEMS)}>")
    return SYSTEMS[match.group(2)]


def find_suite_dir(raw_dir: Path, suite: str) -> Path:
    """Find ``<raw_dir>/<suite>`` or ``<raw_dir>/<parent>/<suite>``, refusing any ambiguity."""
    if not raw_dir.is_dir():
        raise RawLayoutError(f"Raw data directory {raw_dir} does not exist")
    matches = [path for path in sorted(raw_dir.glob(suite)) if path.is_dir()]
    matches += [path for path in sorted(raw_dir.glob(f"*/{suite}")) if path.is_dir()]
    if len(matches) != 1:
        children = sorted(path.name for path in raw_dir.iterdir())
        raise RawLayoutError(
            f"Expected one directory for suite {suite} under {raw_dir}, "
            f"found {[str(path) for path in matches]}; {raw_dir} contains {children}"
        )
    return matches[0]


def discover_cases(suite_dir: Path, suite: str) -> tuple[list[CaseSource], list[Issue]]:
    """List the cases of a suite in a stable order, with an issue for each unusable directory."""
    sources: list[CaseSource] = []
    issues: list[Issue] = []
    children = sorted(path for path in suite_dir.iterdir() if path.is_dir())
    for case_dir in children:
        label = f"{suite}/{case_dir.name}"
        match = _CASE_DIR_RE.match(case_dir.name)
        if match is None:
            issues.append(Issue(label, LOAD_CHECK, "directory is not <service>_<fault>"))
            continue
        repetitions = sorted(path for path in case_dir.iterdir() if path.is_dir())
        if not repetitions:
            issues.append(Issue(label, LOAD_CHECK, "no repetition directories"))
        for path in repetitions:
            if not path.name.isdecimal():
                reason = f"repetition directory {path.name!r} is not a number"
                issues.append(Issue(label, LOAD_CHECK, reason))
                continue
            sources.append(
                CaseSource(suite, match["service"], match["fault"], int(path.name), path)
            )
    if not sources:
        sample = sorted(path.name for path in suite_dir.iterdir())[:10]
        raise RawLayoutError(
            f"No <service>_<fault>/<repetition> cases under {suite_dir}; it contains {sample}. "
            "Run `cascadelens data audit` to inspect the layout before changing the loader."
        )
    sources.sort(key=lambda source: (source.service, source.fault, source.repetition))
    return sources, issues


def _metric_file(path: Path) -> Path:
    """Pick the metrics file with the same preference order as RCAEval's ``main.py``."""
    for name in METRIC_FILES:
        if (path / name).is_file():
            return path / name
    raise CaseLoadError(f"none of {METRIC_FILES} in {path}")


def _as_seconds(column: pl.Series, path: Path) -> pl.Series:
    try:
        return column.cast(pl.Float64, strict=True)
    except InvalidOperationError as error:
        raise CaseLoadError(f"non-numeric {column.name!r} in {path.name}: {error}") from error


def _drop_repeated_time_columns(raw: pl.DataFrame, path: Path, notes: list[str]) -> pl.DataFrame:
    """Drop repeated ``time`` columns, and any block that sits on a different clock.

    RCAEval cases are sometimes several tables glued side by side. A repeated ``time`` column
    equal to the first (compared as numbers, since ``1.0`` and ``1`` are the same instant) is
    redundant. One that disagrees starts a block on another clock, so its columns cannot be
    placed on the first timeline; they are dropped and reported instead of being mis-dated.
    """
    first = _as_seconds(raw[TIME_COLUMN], path)
    keep: list[str] = []
    dropped: list[str] = []
    foreign = False
    for column in raw.columns:
        if column == TIME_COLUMN:
            keep.append(column)
        elif _REPEATED_TIME_RE.match(column):
            other = _as_seconds(raw[column], path)
            both = first.is_not_null() & other.is_not_null()
            foreign = bool(both.any()) and not bool(((first == other) | ~both).all())
            if foreign:
                notes.append(f"{column!r} disagrees with {TIME_COLUMN!r}; later columns dropped")
        elif foreign:
            dropped.append(column)
        else:
            keep.append(column)
    if dropped:
        notes.append(f"dropped {len(dropped)} column(s) on another clock: {dropped[:5]}")
    return raw.select(keep)


def _drop_untimed_tail(raw: pl.DataFrame, notes: list[str]) -> pl.DataFrame:
    """Drop trailing rows without a timestamp, as RCAEval's own window selection does.

    Its ``time < inject`` and ``time >= inject`` filters both reject a missing time, so those
    rows never reach a method there either. Timestamps are not guessed. A missing time anywhere
    but at the end is not a recording artifact and fails the case.
    """
    untimed = raw[TIME_COLUMN].is_null().to_list()
    count = sum(untimed)
    if not count:
        return raw
    if untimed != [False] * (len(untimed) - count) + [True] * count:
        raise CaseLoadError(f"a {TIME_COLUMN!r} value is missing before the last timestamped row")
    kept = raw.head(len(untimed) - count)
    tail = raw.tail(count).drop(TIME_COLUMN)
    with_values = int(tail.select(pl.any_horizontal(pl.all().is_not_null())).to_series().sum())
    notes.append(
        f"dropped {count} trailing row(s) with no timestamp, {with_values} of them holding values"
    )
    return kept


def _read_wide(path: Path, notes: list[str]) -> pl.DataFrame:
    """Read a metrics CSV, strictly typed, with a ``timestamp`` column and float metrics.

    Repairs are appended to ``notes`` so the caller can report them.
    """
    try:
        # Everything is read as text first so that a bad cell fails instead of inferring a type.
        raw = pl.read_csv(path, infer_schema=False)
    except PolarsError as error:
        raise CaseLoadError(f"cannot parse {path.name}: {error}") from error
    if TIME_COLUMN not in raw.columns:
        raise CaseLoadError(f"{path.name} has no {TIME_COLUMN!r} column")
    raw = _drop_untimed_tail(_drop_repeated_time_columns(raw, path, notes), notes)
    if raw.width == 1:
        raise CaseLoadError(f"{path.name} has no metric columns")
    try:
        times = raw[TIME_COLUMN].cast(pl.Float64, strict=True)
        metrics = raw.drop(TIME_COLUMN).select(
            pl.all().cast(pl.Float64, strict=True).fill_nan(None)
        )
    except InvalidOperationError as error:
        raise CaseLoadError(f"non-numeric value in {path.name}: {error}") from error
    if times.is_null().any() or times.is_nan().any():
        raise CaseLoadError(f"{path.name} has a missing {TIME_COLUMN} value")
    if (times % 1 != 0).any():
        raise CaseLoadError(f"{path.name} has fractional {TIME_COLUMN} values")
    if (times.abs() >= MAX_UNIX_SECONDS).any():
        raise CaseLoadError(f"{path.name}: {TIME_COLUMN} is not in Unix seconds")
    return metrics.with_columns(times.cast(pl.Int64).alias(TIMESTAMP_COLUMN))


def _split_columns(columns: Sequence[str]) -> tuple[pl.DataFrame, dict[str, str]]:
    """Map each ``<service>_<metric>`` column to its parts, keeping the original service names."""
    rows: list[dict[str, str]] = []
    originals: dict[str, str] = {}
    for column in columns:
        if _UNNAMED_COLUMN_RE.match(column):
            raise CaseLoadError(f"unnamed column {column!r}")
        if "_duplicated_" in column:
            raise CaseLoadError(f"duplicate metric column {column!r}")
        service, separator, metric = column.partition("_")
        if not separator or not service or not metric:
            raise CaseLoadError(f"column {column!r} is not <service>_<metric>")
        normalized = normalize_service_name(service)
        if originals.setdefault(normalized, service) != service:
            raise CaseLoadError(
                f"services {originals[normalized]!r} and {service!r} both normalize to "
                f"{normalized!r}"
            )
        rows.append({"column": column, "service": normalized, "metric": metric})
    return pl.DataFrame(rows, schema={name: pl.String for name in rows[0]}), originals


def _to_long(wide: pl.DataFrame, columns: pl.DataFrame) -> pl.DataFrame:
    """Unpivot to the long table. Row order within a series stays as read from the file."""
    return (
        wide.unpivot(index=TIMESTAMP_COLUMN, variable_name="column", value_name="value")
        .join(columns, on="column", maintain_order="left")
        .select("service", "metric", TIMESTAMP_COLUMN, "value")
        .sort("service", "metric", maintain_order=True)
    )


def _read_inject_time(path: Path) -> int:
    """Read the first line of ``inject_time.txt`` as an integer, as RCAEval's ``main.py`` does."""
    target = path / INJECT_TIME_FILE
    if not target.is_file():
        raise CaseLoadError(f"{INJECT_TIME_FILE} is missing")
    lines = target.read_text(encoding="utf-8").splitlines()
    try:
        return int(lines[0].strip())
    except (IndexError, ValueError) as error:
        raise CaseLoadError(f"{INJECT_TIME_FILE} does not start with an integer") from error


def load_case(source: CaseSource) -> Case:
    """Read one case, raising CaseLoadError when its files are unusable."""
    return load_case_with_notes(source)[0]


def load_case_with_notes(source: CaseSource) -> tuple[Case, list[Issue]]:
    """Read one case, with a warning for every repair made to its raw file."""
    notes: list[str] = []
    wide = _read_wide(_metric_file(source.path), notes)
    if wide.is_empty():
        raise CaseLoadError("metrics file has no rows")
    columns, originals = _split_columns(
        [column for column in wide.columns if column != TIMESTAMP_COLUMN]
    )
    # Derived before inject_time is read; see the module docstring.
    timestamp = pl.col(TIMESTAMP_COLUMN)
    window_start, window_end = wide.select(timestamp.min().alias("start"), timestamp.max()).row(0)
    inject_time = _read_inject_time(source.path)

    service = normalize_service_name(source.service)
    originals.setdefault(service, source.service)
    case = Case(
        case_id=f"{source.suite.lower()}_{service}_{source.fault.lower()}_{source.repetition}",
        suite=source.suite,
        system=system_of(source.suite),
        fault_type=source.fault.lower(),
        run_id=f"{source.suite}/{service}_{source.fault.lower()}",
        repetition=source.repetition,
        root_cause_service=service,
        inject_time=inject_time,
        window_start=window_start,
        window_end=window_end,
        service_names=originals,
        metrics=_to_long(wide, columns),
    )
    return case, [Issue(case.case_id, REPAIR_CHECK, note, "warning") for note in notes]


def iter_cases(raw_dir: Path, suites: Sequence[str]) -> Iterator[Case | Issue]:
    """Yield each case one at a time, or an Issue where a case could not be read."""
    for suite in suites:
        system_of(suite)
        sources, issues = discover_cases(find_suite_dir(raw_dir, suite), suite)
        yield from issues
        for source in sources:
            try:
                case, repairs = load_case_with_notes(source)
            except (CaseLoadError, ValueError) as error:
                # ValueError covers schema rejections, such as a name unusable in a case_id.
                yield Issue(source.label, LOAD_CHECK, str(error))
            else:
                yield from repairs
                yield case
