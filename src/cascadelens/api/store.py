"""SQLite storage for incidents, their diagnoses and the decision settings.

An incident keeps the case it was created from, so it can be run again. It has no field for a
benchmark label or an injection time. Each run adds a diagnosis, and the incident row copies the
latest one so that the list can filter on it. Without a path the store lives in memory.
"""

import time
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, col, create_engine, select

ID_PREFIX = "INC-"
ID_WIDTH = 4


class Incident(SQLModel, table=True):
    """One incident and a copy of its latest diagnosis."""

    seq: int | None = Field(default=None, primary_key=True)
    title: str
    severity: str = Field(index=True)
    status: str = Field(index=True)
    owner: str | None = None
    is_demo: bool = False
    window_start: int = Field(index=True)
    window_end: int
    created_at: int
    case_json: str
    top_service: str = Field(index=True)
    top_score: float
    confidence: float
    abstain: bool
    calibrated: bool


class DiagnosisRecord(SQLModel, table=True):
    """One run of the model on an incident's case. Nothing here changes after it is written."""

    id: int | None = Field(default=None, primary_key=True)
    incident_seq: int = Field(foreign_key="incident.seq", index=True)
    created_at: int
    result_json: str
    settings_json: str
    topology_source: str
    abstain_threshold: float | None = None


class StoredSettings(SQLModel, table=True):
    """The decision settings, in a single row. A null falls back to the model or the case."""

    id: int = Field(default=1, primary_key=True)
    abstain_threshold: float | None = None
    window_seconds: int | None = None
    use_provided_topology: bool = True


def incident_id(seq: int) -> str:
    """Return the public ID of a row."""
    return f"{ID_PREFIX}{seq:0{ID_WIDTH}d}"


def parse_incident_id(text: str) -> int | None:
    """Return the row of a public ID, or None if it is not one."""
    digits = text.removeprefix(ID_PREFIX)
    if digits == text or not digits.isascii() or not digits.isdigit():
        return None
    return int(digits)


def open_engine(path: Path | None) -> Engine:
    """Open the database, creating its tables on first use."""
    if path is None:
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


class Store:
    """The reads and writes the API makes."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def add(self, incident: Incident, record: DiagnosisRecord) -> Incident:
        """Save a new incident and its first diagnosis in one transaction."""
        with Session(self.engine) as session:
            session.add(incident)
            session.flush()
            assert incident.seq is not None
            record.incident_seq = incident.seq
            session.add(record)
            session.commit()
            session.refresh(incident)
            return incident

    def add_diagnosis(self, incident: Incident, record: DiagnosisRecord) -> Incident:
        """Save a further diagnosis and copy its summary to the incident."""
        with Session(self.engine) as session:
            session.add(record)
            merged = session.merge(incident)
            session.commit()
            session.refresh(merged)
            return merged

    def incident(self, seq: int) -> Incident | None:
        """Return one incident."""
        with Session(self.engine) as session:
            return session.get(Incident, seq)

    def latest_diagnosis(self, seq: int) -> DiagnosisRecord:
        """Return the newest diagnosis of an incident."""
        query = (
            select(DiagnosisRecord)
            .where(DiagnosisRecord.incident_seq == seq)
            .order_by(col(DiagnosisRecord.id).desc())
        )
        with Session(self.engine) as session:
            return session.exec(query).first()  # type: ignore[return-value]

    def incidents(
        self,
        status: str | None = None,
        severity: str | None = None,
        service: str | None = None,
        since: int | None = None,
        before: int | None = None,
        limit: int = 50,
    ) -> list[Incident]:
        """Return incidents, newest first, optionally only those older than row ``before``."""
        query = select(Incident)
        if status is not None:
            query = query.where(Incident.status == status)
        if severity is not None:
            query = query.where(Incident.severity == severity)
        if service is not None:
            query = query.where(Incident.top_service == service)
        if since is not None:
            query = query.where(Incident.window_start >= since)
        if before is not None:
            query = query.where(col(Incident.seq) < before)
        with Session(self.engine) as session:
            return list(session.exec(query.order_by(col(Incident.seq).desc()).limit(limit)))

    def has_demo(self) -> bool:
        """Report whether demo incidents have been loaded."""
        with Session(self.engine) as session:
            found = session.exec(select(Incident).where(col(Incident.is_demo))).first()
            return found is not None

    def settings(self) -> StoredSettings:
        """Return the decision settings."""
        with Session(self.engine) as session:
            return session.get(StoredSettings, 1) or StoredSettings()

    def save_settings(self, settings: StoredSettings) -> StoredSettings:
        """Replace the decision settings."""
        with Session(self.engine) as session:
            merged = session.merge(settings)
            session.commit()
            session.refresh(merged)
            return merged


def now() -> int:
    """Return the current Unix time in seconds."""
    return int(time.time())
