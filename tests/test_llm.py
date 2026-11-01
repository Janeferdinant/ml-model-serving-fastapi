"""Unit tests for LLM streaming client protocol and Ollama adapter."""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from mlserving.llm import (
    LLMGenerateRequest,
    LLMGenerateResponse,
    MockLLMClient,
    OllamaStreamingClient,
    TokenChunk,
)


@pytest.mark.anyio
async def test_mock_llm_client_streaming():
    client = MockLLMClient(tokens=["Hello", " from", " local", " LLM!"])
    req = LLMGenerateRequest(prompt="Say hello", model="test-model", stream=True)

    chunks: list[TokenChunk] = []
    async for chunk in client.stream_generate(req):
        chunks.append(chunk)

    # 4 token chunks + 1 terminal chunk
    assert len(chunks) == 5
    full_text = "".join(c.text for c in chunks)
    assert full_text == "Hello from local LLM!"
    assert chunks[-1].done is True
    assert chunks[-1].tokens_in is not None
    assert chunks[-1].tokens_in > 0
    assert chunks[-1].tokens_out == 4
    assert chunks[-1].ttft_ms is not None
    assert chunks[-1].ttft_ms >= 0


@pytest.mark.anyio
async def test_mock_llm_client_non_streaming():
    client = MockLLMClient(tokens=["Structured", " response"])
    req = LLMGenerateRequest(prompt="Test prompt", model="test-model", stream=False)

    resp: LLMGenerateResponse = await client.generate(req)
    assert resp.text == "Structured response"
    assert resp.tokens_out == 2
    assert resp.model == "test-model"


def test_generate_request_validation():
    with pytest.raises(ValidationError):
        LLMGenerateRequest(prompt="", model="test-model")  # prompt cannot be empty


@pytest.mark.anyio
async def test_ollama_streaming_client_parsing(monkeypatch: pytest.MonkeyPatch):
    class MockAsyncResponse:
        def __init__(self, lines: list[str]) -> None:
            self._lines = lines

        def raise_for_status(self) -> None:
            pass

        async def aiter_lines(self):
            for line in self._lines:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def stream(self, method: str, url: str, json=None):
            lines = [
                '{"response": "Hello", "done": false}',
                '{"response": " Ollama", "done": false}',
                '{"response": "", "done": true, "prompt_eval_count": 5, "eval_count": 2}',
            ]
            return MockAsyncResponse(lines)

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    client = OllamaStreamingClient(base_url="http://mock-ollama:11434")
    req = LLMGenerateRequest(prompt="Test Ollama", model="qwen3:8b", max_tokens=10)

    chunks: list[TokenChunk] = []
    async for c in client.stream_generate(req):
        chunks.append(c)

    assert len(chunks) == 3
    assert chunks[0].text == "Hello"
    assert chunks[1].text == " Ollama"
    assert chunks[2].done is True
    assert chunks[2].tokens_in == 5
    assert chunks[2].tokens_out == 2

    # Also test non-streaming generate()
    resp = await client.generate(req)
    assert resp.text == "Hello Ollama"
    assert resp.tokens_out == 2
