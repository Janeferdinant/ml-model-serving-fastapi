"""FastAPI application factory.

Run with ``uvicorn mlserving.app:create_app --factory`` (configuration from ``MLS_*``).
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import IO, Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from threadpoolctl import threadpool_limits

from mlserving import __version__
from mlserving.config import Settings
from mlserving.logging_config import configure_logging
from mlserving.metrics import CONTENT_TYPE, Metrics
from mlserving.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    TokenBucketLimiter,
)
from mlserving.predictor import ModelBundle, load_bundle
from mlserving.reqlog import RequestLogWriter
from mlserving.schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    PredictionRequest,
    PredictionResponse,
)

log = logging.getLogger("mlserving")

DESCRIPTION = """
Serves a small gradient-boosted regressor trained on California Housing. The model is
deliberately simple; the point of the service is the engineering around it: checksum-verified
versioned artifacts, validated inputs, rate limiting, Prometheus metrics, request logging for
drift monitoring.

The target is the median house value of a census block group, in units of USD 100,000.
"""

COMMON_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key."},
    413: {"model": ErrorResponse, "description": "Request body too large."},
    422: {"model": ErrorResponse, "description": "Validation failed or value outside range."},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded (see Retry-After)."},
    503: {"model": ErrorResponse, "description": "Model not loaded or failed verification."},
}


def _rid(request: Request) -> str | None:
    value = request.scope.get("state", {}).get("request_id")
    return value if isinstance(value, str) else None


def get_bundle(request: Request) -> ModelBundle:
    bundle: ModelBundle | None = request.app.state.bundle
    if bundle is None:
        raise HTTPException(
            status_code=503, detail=f"Model not ready: {request.app.state.load_error}"
        )
    return bundle


BundleDep = Annotated[ModelBundle, Depends(get_bundle)]


def create_app(settings: Settings | None = None, *, log_stream: IO[str] | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level, log_stream)
    metrics = Metrics()
    request_log = RequestLogWriter(settings.request_log_path) if settings.request_log_path else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.bundle = None
        app.state.load_error = None
        limits = threadpool_limits(limits=settings.predict_threads)
        try:
            bundle = load_bundle(settings.model_dir, settings.model_version, settings.range_margin)
        except Exception as exc:
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            metrics.set_model(None, None)
            log.error("model failed to load", extra={"fields": {"error": app.state.load_error}})
            if settings.strict_startup:
                limits.restore_original_limits()
                raise
        else:
            app.state.bundle = bundle
            sha = bundle.metadata["artifacts"]["model"]["sha256"]
            metrics.set_model(bundle.version, sha)
            log.info(
                "model loaded",
                extra={
                    "fields": {
                        "model_version": bundle.version,
                        "stage": bundle.stage,
                        "sha256": sha,
                        "auth_enabled": settings.api_key is not None,
                        "rate_limit_per_minute": settings.rate_limit_per_minute,
                    }
                },
            )
        try:
            yield
        finally:
            limits.restore_original_limits()
            if request_log:
                request_log.close()

    app = FastAPI(
        title="California Housing model service",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.bundle = None
    app.state.load_error = None

    # ------------------------------------------------------------------ dependencies

    async def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if settings.api_key is None:
            return
        supplied = (x_api_key or "").encode()
        if not secrets.compare_digest(supplied, settings.api_key.encode()):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API key (send it in the X-API-Key header)",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    auth = [Depends(require_api_key)]

    # ------------------------------------------------------------------ error handlers

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        if any(e.get("type") == "json_invalid" for e in errors):
            return JSONResponse(
                {"detail": "Malformed JSON body", "request_id": _rid(request)}, status_code=400
            )
        detail = [
            {"loc": list(e["loc"]), "msg": str(e["msg"]), "type": str(e["type"])} for e in errors
        ]
        return JSONResponse({"detail": detail, "request_id": _rid(request)}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail, "request_id": _rid(request)},
            status_code=exc.status_code,
            headers=dict(exc.headers) if exc.headers else None,
        )

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        rid = _rid(request)
        return JSONResponse(
            {"detail": "Internal server error", "request_id": rid},
            status_code=500,
            headers={"x-request-id": rid} if rid else None,
        )

    # ------------------------------------------------------------------ helpers

    def validate_rows(
        bundle: ModelBundle, rows: list[dict[str, float]], loc_prefix: tuple[str | int, ...]
    ) -> None:
        errors: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            for v in bundle.range_checker.violations(row):
                loc = (
                    (*loc_prefix, i, v.feature) if len(loc_prefix) > 1 else (*loc_prefix, v.feature)
                )
                errors.append(
                    {
                        "type": "value_error.out_of_range",
                        "loc": loc,
                        "msg": v.message,
                        "input": v.value,
                    }
                )
        if errors:
            raise RequestValidationError(errors[:50])

    def score(bundle: ModelBundle, rows: list[dict[str, float]], request: Request) -> list[float]:
        preds = bundle.predict(bundle.to_matrix(rows)).tolist()
        metrics.observe_predictions(bundle.version, preds)
        if request_log:
            request_log.log(_rid(request) or "", bundle.version, rows, preds)
        return [float(p) for p in preds]

    # ------------------------------------------------------------------ routes

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        dependencies=auth,
        responses=COMMON_ERRORS,
        tags=["inference"],
        summary="Predict the median house value for one block group",
    )
    def predict(
        payload: PredictionRequest, request: Request, bundle: BundleDep
    ) -> PredictionResponse:
        row = payload.model_dump()
        validate_rows(bundle, [row], ("body",))
        (value,) = score(bundle, [row], request)
        return PredictionResponse(
            prediction=value, model_version=bundle.version, request_id=_rid(request) or ""
        )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        dependencies=auth,
        responses=COMMON_ERRORS,
        tags=["inference"],
        summary="Predict for several block groups in one call",
    )
    def predict_batch(
        payload: BatchPredictionRequest, request: Request, bundle: BundleDep
    ) -> BatchPredictionResponse:
        n = len(payload.instances)
        if n > settings.max_batch_size:
            raise RequestValidationError(
                [
                    {
                        "type": "value_error.batch_too_large",
                        "loc": ("body", "instances"),
                        "msg": f"Batch has {n} instances; at most {settings.max_batch_size} "
                        "are accepted.",
                        "input": n,
                    }
                ]
            )
        rows = [inst.model_dump() for inst in payload.instances]
        validate_rows(bundle, rows, ("body", "instances"))
        preds = score(bundle, rows, request)
        return BatchPredictionResponse(
            predictions=preds,
            count=len(preds),
            model_version=bundle.version,
            request_id=_rid(request) or "",
        )

    @app.get("/health", tags=["operations"], summary="Liveness probe")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/ready",
        tags=["operations"],
        summary="Readiness probe: model loaded and checksum verified",
        responses={503: {"description": "Model not loaded or failed verification."}},
    )
    def ready(request: Request) -> JSONResponse:
        bundle: ModelBundle | None = request.app.state.bundle
        if bundle is None:
            return JSONResponse(
                {"status": "not_ready", "reason": request.app.state.load_error}, status_code=503
            )
        return JSONResponse(
            {"status": "ready", "model_version": bundle.version, "checksum_verified": True}
        )

    @app.get(
        "/model-info",
        dependencies=auth,
        responses=COMMON_ERRORS,
        tags=["operations"],
        summary="Loaded model version, schema, metrics and provenance",
    )
    def model_info(bundle: BundleDep) -> dict[str, Any]:
        meta = bundle.metadata
        schema = []
        for spec in meta["feature_schema"]:
            low, high = bundle.range_checker.bounds(spec["name"])
            schema.append({**spec, "accepted_min": low, "accepted_max": high})
        return {
            "version": bundle.version,
            "stage": bundle.stage,
            "model_type": meta["model_type"],
            "target": meta["target"],
            "feature_names": meta["feature_names"],
            "feature_schema": schema,
            "range_margin": settings.range_margin,
            "hyperparameters": meta["hyperparameters"],
            "seed": meta["seed"],
            "metrics": meta["metrics"],
            "dataset": meta["dataset"],
            "git": meta["git"],
            "trained_at": meta["trained_at"],
            "library_versions": meta["library_versions"],
            "artifacts": meta["artifacts"],
            "checksum_verified": True,
        }

    @app.get(
        "/metrics",
        tags=["operations"],
        summary="Prometheus metrics (text exposition format)",
        response_class=Response,
        responses={200: {"content": {"text/plain": {}}, "description": "Prometheus text format."}},
    )
    def metrics_endpoint() -> Response:
        return Response(content=metrics.render(), media_type=CONTENT_TYPE)

    # ------------------------------------------------------------------ middleware
    # Added innermost first; the request-context middleware ends up outermost so that
    # 413 and 429 responses are logged and counted too.
    known_paths = {route.path for route in app.routes if hasattr(route, "path")}
    app.add_middleware(
        RateLimitMiddleware, limiter=TokenBucketLimiter(settings.rate_limit_per_minute)
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RequestContextMiddleware, metrics=metrics, known_paths=known_paths)
    return app
