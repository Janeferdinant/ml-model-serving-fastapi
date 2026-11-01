"""Tests for Prometheus metrics emitted by LLM endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from mlserving.app import create_app
from mlserving.config import Settings
from mlserving.llm import MockLLMClient


def test_llm_metrics_on_successful_generation(model_dir: Path) -> None:
    settings = Settings(model_dir=model_dir, rate_limit_per_minute=0)
    app = create_app(settings)
    app.state.llm_client = MockLLMClient(tokens=["Hello", " world", "!"])

    with TestClient(app) as client:
        payload = {"prompt": "Say hello to everyone", "stream": True}
        with client.stream("POST", "/v1/generate", json=payload) as resp:
            assert resp.status_code == 200
            _ = list(resp.iter_lines())

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        text = metrics_resp.text

        assert 'llm_requests_total{status="success"} 1.0' in text
        assert 'llm_tokens_total{direction="prompt"}' in text
        assert 'llm_tokens_total{direction="completion"} 3.0' in text
        assert "llm_ttft_seconds_bucket" in text
        assert "llm_request_duration_seconds_bucket" in text


def test_llm_metrics_on_non_streaming_generation(model_dir: Path) -> None:
    settings = Settings(model_dir=model_dir, rate_limit_per_minute=0)
    app = create_app(settings)
    app.state.llm_client = MockLLMClient(tokens=["One", " two"])

    with TestClient(app) as client:
        payload = {"prompt": "Count numbers", "stream": False}
        res = client.post("/v1/generate", json=payload)
        assert res.status_code == 200

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        text = metrics_resp.text

        assert 'llm_requests_total{status="success"} 1.0' in text
        assert 'llm_tokens_total{direction="completion"} 2.0' in text


def test_llm_metrics_on_rate_limit(model_dir: Path) -> None:
    settings = Settings(model_dir=model_dir, rate_limit_per_minute=1)
    app = create_app(settings)
    app.state.llm_client = MockLLMClient(tokens=["Testing rate limit"])

    with TestClient(app) as client:
        payload = {"prompt": "Prompt 1", "stream": False}
        res1 = client.post("/v1/generate", json=payload)
        assert res1.status_code == 200

        res2 = client.post("/v1/generate", json=payload)
        assert res2.status_code == 429

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        text = metrics_resp.text

        assert 'llm_requests_total{status="rate_limited"} 1.0' in text
