"""Rate limiting tests for the LLM streaming endpoint /v1/generate."""

from pathlib import Path

from fastapi.testclient import TestClient

from mlserving.app import create_app
from mlserving.config import Settings
from mlserving.llm import MockLLMClient


def test_llm_generate_rate_limit_returns_429(model_dir: Path) -> None:
    settings = Settings(model_dir=model_dir, rate_limit_per_minute=3)
    app = create_app(settings)
    app.state.llm_client = MockLLMClient(tokens=["Test response"])

    with TestClient(app) as client:
        payload = {"prompt": "Hello world", "stream": False}
        codes = [client.post("/v1/generate", json=payload).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert codes[3:] == [429, 429]

        res = client.post("/v1/generate", json=payload)
        assert res.status_code == 429
        assert int(res.headers["retry-after"]) >= 1
        assert "rate limit" in res.json()["detail"].lower()
        assert "request_id" in res.json()


def test_llm_streaming_rate_limit_returns_429(model_dir: Path) -> None:
    settings = Settings(model_dir=model_dir, rate_limit_per_minute=2)
    app = create_app(settings)
    app.state.llm_client = MockLLMClient(tokens=["Chunk 1", "Chunk 2"])

    with TestClient(app) as client:
        payload = {"prompt": "Streaming test", "stream": True}
        with client.stream("POST", "/v1/generate", json=payload) as resp1:
            assert resp1.status_code == 200
        with client.stream("POST", "/v1/generate", json=payload) as resp2:
            assert resp2.status_code == 200
        with client.stream("POST", "/v1/generate", json=payload) as resp3:
            assert resp3.status_code == 429
            assert int(resp3.headers["retry-after"]) >= 1
