"""The local HTTP API. Read-only analysis, with no authentication: never expose it publicly.

Every response carries the disclaimer, errors included. Handlers call ``inference.diagnose`` and
make no outbound request.
"""

import json
from pathlib import Path
from typing import Any, NoReturn

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from cascadelens import __version__
from cascadelens.api.incidents import IncidentService
from cascadelens.api.model_report import frozen_config_hash, read_metrics
from cascadelens.api.store import Store, open_engine
from cascadelens.api.v1 import ApiFailure, build_router
from cascadelens.baselines.base import BaselineError
from cascadelens.eval.errors import EvalError
from cascadelens.features.context import FeatureError
from cascadelens.inference import (
    DEFAULT_TOLERANCE_SECONDS,
    CasePayload,
    LoadedModel,
    case_from_payload,
    diagnose,
    load_model,
    model_details,
)
from cascadelens.models.diagnose import DISCLAIMER

DEFAULT_MAX_BODY_BYTES = 25_000_000
API_PREFIX = "/api/v1"
SCHEMA_REFERENCE = "#/components/schemas/{model}"
STATUS_CODES = {
    404: "not_found",
    405: "method_not_allowed",
    413: "body_too_large",
    422: "validation_error",
}
CALIBRATION_RESULTS = Path("experiments/calibration/results.json")
RANKER_RESULTS = Path("experiments/ranker_clean/results.json")
ROBUSTNESS_RESULTS = Path("experiments/robustness/results.json")
CASE_FAILURES = (BaselineError, EvalError, FeatureError)


class ApiSettings(BaseModel):
    """Where the model lives and what the API accepts."""

    model_dir: Path
    root: Path = Path()
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    window_tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS
    # Without a path, incidents last as long as the process.
    database: Path | None = None
    cors_origins: tuple[str, ...] = ()


def _top1(rows: list[dict[str, Any]], method: str) -> dict[str, float] | None:
    for row in rows:
        if row.get("method") == method:
            found: dict[str, float] = row["Top-1"]
            return found
    return None


def headline_metrics(root: Path) -> dict[str, Any]:
    """Return headline numbers from the committed result JSONs, or null where one is absent."""
    ranker = read_metrics(root / RANKER_RESULTS)
    robustness = read_metrics(root / ROBUSTNESS_RESULTS)
    calibration = read_metrics(root / CALIBRATION_RESULTS)
    return {
        "status": ranker["status"] if ranker else None,
        "two_stage_top1_group_run": _top1(ranker["group_run"], "two_stage_ranker")
        if ranker
        else None,
        "lightgbm_flat_top1_group_run": _top1(robustness["grid"], "lightgbm_flat")
        if robustness
        else None,
        "calibration_verdicts": calibration["verdicts"] if calibration else None,
    }


def _error(request: Request, status: int, detail: str, code: str | None = None) -> JSONResponse:
    """Return an error, in the ``/api/v1`` shape under that prefix."""
    if not request.url.path.startswith(API_PREFIX):
        return JSONResponse({"detail": detail, "disclaimer": DISCLAIMER}, status_code=status)
    name = code or STATUS_CODES.get(status, "internal_error")
    return JSONResponse(
        {"error": {"code": name, "message": detail}, "disclaimer": DISCLAIMER}, status_code=status
    )


def _field_errors(error: ValidationError | RequestValidationError) -> str:
    """Return a short message. Input is not echoed, and neither are unknown field names."""
    messages = []
    for item in error.errors()[:5]:
        place = item["loc"]
        if item["type"] == "extra_forbidden":
            where = ".".join(str(part) for part in place[:-1])
            messages.append(f"{where or 'body'}: unknown field")
        else:
            where = ".".join(str(part) for part in place)
            messages.append(f"{where or 'body'}: {item['msg']}")
    return "; ".join(messages)


async def _body(request: Request, limit: int) -> bytes:
    """Read the body, stopping as soon as it passes the limit, with or without a length header."""
    message = f"Body exceeds the limit of {limit} bytes"
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=message)
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=message)
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not a finite number")


def _payload(body: bytes) -> CasePayload:
    try:
        raw = json.loads(body, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        raise HTTPException(status_code=422, detail=f"Body is not valid JSON: {error}") from error
    try:
        return CasePayload.model_validate(raw)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=_field_errors(error)) from error


def _install_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_failure(request: Request, error: StarletteHTTPException) -> JSONResponse:
        return _error(request, error.status_code, str(error.detail))

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, error: RequestValidationError) -> JSONResponse:
        return _error(request, 422, _field_errors(error))

    @app.exception_handler(ApiFailure)
    async def refused(request: Request, error: ApiFailure) -> JSONResponse:
        return _error(request, error.status, error.message, error.code)

    for failure in CASE_FAILURES:

        @app.exception_handler(failure)
        async def invalid_case(request: Request, error: Exception) -> JSONResponse:
            return _error(request, 422, str(error), "invalid_case")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, __: Exception) -> JSONResponse:
        return _error(request, 500, "The case could not be analysed")


def create_app(settings: ApiSettings, model: LoadedModel | None = None) -> FastAPI:
    """Build the app. The model loads here, once, and a missing model is a startup error."""
    loaded = model if model is not None else load_model(settings.model_dir)
    details = model_details(loaded)
    # The built-in docs pages load scripts from a public CDN, so they stay off.
    app = FastAPI(
        title="CascadeLens",
        version=__version__,
        description=DISCLAIMER,
        docs_url=None,
        redoc_url=None,
    )
    _install_handlers(app)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "PUT"],
            allow_headers=["content-type"],
        )
    service = IncidentService(
        Store(open_engine(settings.database)), loaded, settings.window_tolerance_seconds
    )

    async def read_body(request: Request) -> bytes:
        return await _body(request, settings.max_body_bytes)

    app.include_router(build_router(service, settings.root, read_body))

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Report whether the model is loaded."""
        return {"status": "ok", "model_loaded": True, "disclaimer": DISCLAIMER}

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        """Report the model's version, provenance and headline metrics."""
        return {
            **details,
            "frozen_config_hash": frozen_config_hash(details["config_hash"], settings.root),
            "headline_metrics": headline_metrics(settings.root),
            "disclaimer": DISCLAIMER,
        }

    @app.post(
        "/diagnose",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": CasePayload.model_json_schema(ref_template=SCHEMA_REFERENCE)
                    }
                },
            }
        },
    )
    async def post_diagnose(request: Request) -> dict[str, Any]:
        """Rank likely origin services for one case."""
        payload = _payload(await _body(request, settings.max_body_bytes))
        try:
            case = case_from_payload(payload, settings.window_tolerance_seconds)
            result = await run_in_threadpool(diagnose, case, loaded)
        except CASE_FAILURES as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result.model_dump(mode="json")

    return app
