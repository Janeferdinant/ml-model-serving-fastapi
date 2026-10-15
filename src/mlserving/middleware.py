"""Pure-ASGI middleware: request IDs and access logs, body-size limit, rate limiting."""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Collection
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mlserving.metrics import Metrics

log = logging.getLogger("mlserving")

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
EXEMPT_FROM_RATE_LIMIT = frozenset({"/health", "/ready", "/metrics"})


def request_id_of(scope: Scope) -> str | None:
    state = scope.get("state")
    return state.get("request_id") if isinstance(state, dict) else None


class TokenBucketLimiter:
    """Per-key token bucket: ``per_minute`` tokens of burst, refilled continuously."""

    def __init__(
        self,
        per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self.per_minute = per_minute
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    @property
    def size(self) -> int:
        return len(self._buckets)

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, seconds until a token is available)."""
        if self.per_minute <= 0:
            return True, 0.0
        now = self._clock()
        rate = self.per_minute / 60.0
        tokens, last = self._buckets.get(key, (float(self.per_minute), now))
        tokens = min(float(self.per_minute), tokens + (now - last) * rate)
        allowed = tokens >= 1.0
        if allowed:
            tokens -= 1.0
        self._buckets[key] = (tokens, now)
        self._buckets.move_to_end(key)
        while len(self._buckets) > self._max_keys:
            self._buckets.popitem(last=False)
        return allowed, 0.0 if allowed else (1.0 - tokens) / rate


class RequestContextMiddleware:
    """Assigns a request ID, records metrics and writes one JSON access-log line."""

    def __init__(self, app: ASGIApp, metrics: Metrics, known_paths: Collection[str]) -> None:
        self.app = app
        self.metrics = metrics
        self.known_paths = frozenset(known_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get("x-request-id")
        rid = incoming if incoming and REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        status = 500
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)["x-request-id"] = rid
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            log.exception("unhandled error", extra={"fields": {"request_id": rid}})
            raise
        finally:
            seconds = time.perf_counter() - start
            path = scope["path"]
            label = path if path in self.known_paths else "unmatched"
            self.metrics.observe_request(scope["method"], label, status, seconds)
            client = scope.get("client")
            log.info(
                "request",
                extra={
                    "fields": {
                        "request_id": rid,
                        "method": scope["method"],
                        "path": path,
                        "status": status,
                        "duration_ms": round(seconds * 1000, 3),
                        "client": client[0] if client else None,
                    }
                },
            )


class _BodyTooLarge(BaseException):
    pass


class BodySizeLimitMiddleware:
    """Rejects bodies over ``max_bytes`` using Content-Length or, for chunked bodies, a count."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    def _too_large(self, scope: Scope) -> JSONResponse:
        return JSONResponse(
            {
                "detail": f"Request body too large (limit {self.max_bytes} bytes)",
                "request_id": request_id_of(scope),
            },
            status_code=413,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                size = int(declared)
            except ValueError:
                response = JSONResponse(
                    {"detail": "Invalid Content-Length header", "request_id": request_id_of(scope)},
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if size > self.max_bytes:
                await self._too_large(scope)(scope, receive, send)
                return
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await self._too_large(scope)(scope, receive, send)


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp, limiter: TokenBucketLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or self.limiter.per_minute <= 0
            or scope["path"] in EXEMPT_FROM_RATE_LIMIT
        ):
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        allowed, retry_after = self.limiter.check(client[0] if client else "unknown")
        if allowed:
            await self.app(scope, receive, send)
            return
        body: dict[str, Any] = {
            "detail": "Rate limit exceeded; retry later",
            "request_id": request_id_of(scope),
        }
        response = JSONResponse(
            body,
            status_code=429,
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
        await response(scope, receive, send)


__all__ = [
    "BodySizeLimitMiddleware",
    "RateLimitMiddleware",
    "RequestContextMiddleware",
    "TokenBucketLimiter",
]
