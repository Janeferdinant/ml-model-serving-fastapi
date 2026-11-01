"""Integration tests for POST /v1/generate streaming SSE endpoint."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from mlserving.app import create_app
from mlserving.config import Settings
from mlserving.llm import MockLLMClient


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(strict_startup=False)
    app = create_app(settings)
    app.state.llm_client = MockLLMClient(tokens=["FastAPI", " streaming", " with", " SSE!"])
    with TestClient(app) as tc:
        yield tc


def test_generate_streams_sse(client: TestClient) -> None:
    payload = {
        "prompt": "Hello world",
        "model": "qwen3:8b",
        "stream": True,
        "temperature": 0.0,
    }
    with client.stream("POST", "/v1/generate", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [line.strip() for line in response.iter_lines() if line.strip()]
        assert len(lines) >= 5  # 4 tokens + 1 terminal + [DONE]

        events = []
        for line in lines:
            assert line.startswith("data: ")
            data_str = line[len("data: ") :]
            if data_str == "[DONE]":
                break
            events.append(json.loads(data_str))

        full_text = "".join(e.get("text", "") for e in events)
        assert full_text == "FastAPI streaming with SSE!"

        # Check last event has completion stats
        last_event = events[-1]
        assert last_event.get("done") is True
        assert last_event.get("tokens_out") == 4
        assert last_event.get("ttft_ms") is not None


def test_generate_non_streaming(client: TestClient) -> None:
    payload = {
        "prompt": "Hello world non-streaming",
        "model": "qwen3:8b",
        "stream": False,
    }
    response = client.post("/v1/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "FastAPI streaming with SSE!"
    assert data["tokens_out"] == 4
    assert data["model"] == "qwen3:8b"
    assert "duration_ms" in data


def test_generate_validation_error(client: TestClient) -> None:
    # Empty prompt fails validation
    response = client.post("/v1/generate", json={"prompt": "", "stream": True})
    assert response.status_code == 422
    assert "request_id" in response.json()

    # Extra disallowed fields fail validation
    response = client.post("/v1/generate", json={"prompt": "valid", "extra_field": "disallowed"})
    assert response.status_code == 422


def test_generate_auth_required() -> None:
    settings = Settings(strict_startup=False, api_key="secret-key-123")
    app = create_app(settings)
    app.state.llm_client = MockLLMClient()
    with TestClient(app) as tc:
        # Missing header
        res_no_key = tc.post("/v1/generate", json={"prompt": "test"})
        assert res_no_key.status_code == 401

        # Correct header
        res_with_key = tc.post(
            "/v1/generate",
            json={"prompt": "test", "stream": False},
            headers={
                "X-API-Key": "secret-key-123",
            },
        )
        assert res_with_key.status_code == 200
