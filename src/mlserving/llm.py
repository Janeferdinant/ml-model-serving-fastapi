"""LLM client protocol, streaming models, MockLLMClient and OllamaStreamingClient."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field


class LLMGenerateRequest(BaseModel):
    """Payload for POST /v1/generate."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=10000, description="Input user prompt.")
    model: str = Field(default="qwen3:8b", description="Model identifier.")
    stream: bool = Field(default=True, description="Whether to stream response as SSE.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: int | None = Field(default=None, ge=1, le=8192, description="Max generated tokens.")


class TokenChunk(BaseModel):
    """Incremental chunk produced by streaming LLM generator."""

    text: str = Field(default="", description="Token delta.")
    done: bool = Field(default=False, description="Whether stream has finished.")
    tokens_in: int | None = Field(default=None, description="Input prompt token count.")
    tokens_out: int | None = Field(default=None, description="Completion token count.")
    ttft_ms: float | None = Field(default=None, description="Time to first token in milliseconds.")
    duration_ms: float | None = Field(default=None, description="Total latency in milliseconds.")


class LLMGenerateResponse(BaseModel):
    """Full generation result for non-streaming or completed requests."""

    text: str = Field(description="Aggregated generated text.")
    model: str = Field(description="Model used for generation.")
    tokens_in: int = Field(description="Input prompt tokens.")
    tokens_out: int = Field(description="Output completion tokens.")
    ttft_ms: float = Field(description="Time to first token in milliseconds.")
    duration_ms: float = Field(description="Total latency in milliseconds.")
    request_id: str | None = Field(default=None, description="Associated request ID.")


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM streaming and batch generation."""

    def stream_generate(self, req: LLMGenerateRequest) -> AsyncIterator[TokenChunk]:
        """Stream chunks as an async iterator."""
        ...

    async def generate(self, req: LLMGenerateRequest) -> LLMGenerateResponse:
        """Collect full generation non-streamingly."""
        ...


class MockLLMClient:
    """Mock client for offline testing in CI without Ollama running."""

    def __init__(
        self,
        tokens: list[str] | None = None,
        latency_per_token_s: float = 0.0,
    ) -> None:
        self.tokens = list(tokens or ["This", " is", " a", " test", " response."])
        self.latency_per_token_s = latency_per_token_s

    async def stream_generate(self, req: LLMGenerateRequest) -> AsyncIterator[TokenChunk]:
        t0 = time.perf_counter()
        ttft: float | None = None

        tokens_in = max(1, len(req.prompt.split()))
        tokens_out = 0

        for idx, tok in enumerate(self.tokens):
            if self.latency_per_token_s > 0:
                import anyio

                await anyio.sleep(self.latency_per_token_s)
            if idx == 0:
                ttft = (time.perf_counter() - t0) * 1000.0
            tokens_out += 1
            yield TokenChunk(text=tok, done=False)

        total_ms = (time.perf_counter() - t0) * 1000.0
        if ttft is None:
            ttft = total_ms

        yield TokenChunk(
            text="",
            done=True,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            ttft_ms=round(ttft, 2),
            duration_ms=round(total_ms, 2),
        )

    async def generate(self, req: LLMGenerateRequest) -> LLMGenerateResponse:
        t0 = time.perf_counter()
        text_parts: list[str] = []
        ttft_ms: float = 0.0
        tokens_in = max(1, len(req.prompt.split()))
        tokens_out = 0

        async for chunk in self.stream_generate(req):
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.done:
                ttft_ms = chunk.ttft_ms or 0.0
                tokens_out = chunk.tokens_out or len(text_parts)
                if chunk.tokens_in:
                    tokens_in = chunk.tokens_in

        dur_ms = (time.perf_counter() - t0) * 1000.0
        return LLMGenerateResponse(
            text="".join(text_parts),
            model=req.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            ttft_ms=round(ttft_ms, 2),
            duration_ms=round(dur_ms, 2),
        )


class OllamaStreamingClient:
    """Client for local Ollama server streaming generation."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def stream_generate(self, req: LLMGenerateRequest) -> AsyncIterator[TokenChunk]:
        payload: dict[str, Any] = {
            "model": req.model,
            "prompt": req.prompt,
            "stream": True,
            "options": {
                "temperature": req.temperature,
                "seed": 0,
            },
        }
        if req.max_tokens:
            payload["options"]["num_predict"] = req.max_tokens

        t0 = time.perf_counter()
        ttft_ms: float | None = None
        tokens_out = 0
        prompt_tokens = max(1, len(req.prompt.split()))

        async with (
            httpx.AsyncClient(timeout=self.timeout_s) as client,
            client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                tok = data.get("response", "")
                done = data.get("done", False)

                if tok:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    tokens_out += 1
                    yield TokenChunk(text=tok, done=False)

                if done:
                    total_ms = (time.perf_counter() - t0) * 1000.0
                    if ttft_ms is None:
                        ttft_ms = total_ms
                    p_eval = data.get("prompt_eval_count")
                    if p_eval is not None:
                        prompt_tokens = int(p_eval)
                    eval_c = data.get("eval_count")
                    if eval_c is not None:
                        tokens_out = int(eval_c)

                    yield TokenChunk(
                        text="",
                        done=True,
                        tokens_in=prompt_tokens,
                        tokens_out=tokens_out,
                        ttft_ms=round(ttft_ms, 2),
                        duration_ms=round(total_ms, 2),
                    )

    async def generate(self, req: LLMGenerateRequest) -> LLMGenerateResponse:
        t0 = time.perf_counter()
        text_parts: list[str] = []
        ttft_ms: float = 0.0
        tokens_in = 0
        tokens_out = 0

        async for chunk in self.stream_generate(req):
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.done:
                ttft_ms = chunk.ttft_ms or 0.0
                tokens_in = chunk.tokens_in or 0
                tokens_out = chunk.tokens_out or len(text_parts)

        dur_ms = (time.perf_counter() - t0) * 1000.0
        return LLMGenerateResponse(
            text="".join(text_parts),
            model=req.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            ttft_ms=round(ttft_ms, 2),
            duration_ms=round(dur_ms, 2),
        )
