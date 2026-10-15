"""Prometheus metrics (text exposition format), one registry per app instance."""

from __future__ import annotations

from collections.abc import Sequence

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest as _generate_latest

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
# Target is in units of USD 100,000 and is capped near 5.0 in the training data.
PREDICTION_BUCKETS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)

CONTENT_TYPE = CONTENT_TYPE_LATEST


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests by method, route template and status code.",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds.",
            ["method", "path"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "model_predictions_total",
            "Predictions served (a batch of n rows counts n).",
            ["model_version"],
            registry=self.registry,
        )
        self.prediction_value = Histogram(
            "model_prediction_value",
            "Distribution of predicted values (units of USD 100,000).",
            ["model_version"],
            buckets=PREDICTION_BUCKETS,
            registry=self.registry,
        )
        self.ready = Gauge(
            "model_ready", "1 if a model is loaded and checksum-verified.", registry=self.registry
        )
        self.info = Gauge(
            "model_info",
            "Loaded model identity (value is always 1).",
            ["version", "sha256"],
            registry=self.registry,
        )

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        self.requests.labels(method, path, str(status)).inc()
        self.latency.labels(method, path).observe(seconds)

    def observe_predictions(self, version: str, values: Sequence[float]) -> None:
        self.predictions.labels(version).inc(len(values))
        hist = self.prediction_value.labels(version)
        for v in values:
            hist.observe(float(v))

    def set_model(self, version: str | None, sha256: str | None) -> None:
        if version is None:
            self.ready.set(0)
            return
        self.ready.set(1)
        self.info.labels(version, (sha256 or "")[:12]).set(1)

    def render(self) -> bytes:
        return _generate_latest(self.registry)
