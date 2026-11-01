"""Load test for LLM streaming SSE endpoint /v1/generate.

Measures Time To First Token (TTFT), total latency, streaming token throughput,
and p50/p95 percentiles across configurable concurrency levels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from mlserving.app import create_app
from mlserving.config import Settings
from mlserving.llm import MockLLMClient

ROOT = Path(__file__).resolve().parents[1]


async def run_single_stream(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
) -> dict[str, Any]:
    """Execute a single streaming request and collect latency & token metrics."""
    t0 = time.perf_counter()
    ttft: float | None = None
    token_count = 0
    error: str | None = None

    try:
        async with client.stream(
            "POST",
            url,
            json={"prompt": prompt, "stream": True, "model": "qwen3:8b"},
            timeout=30.0,
        ) as response:
            if response.status_code != 200:
                return {
                    "status": response.status_code,
                    "ttft_ms": None,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "tokens": 0,
                    "tokens_per_sec": 0.0,
                    "error": f"HTTP {response.status_code}",
                }
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000.0
                    try:
                        chunk = json.loads(data_str)
                        if chunk.get("text"):
                            token_count += 1
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        error = str(exc)

    duration_ms = (time.perf_counter() - t0) * 1000.0
    duration_s = duration_ms / 1000.0
    tokens_per_sec = (token_count / duration_s) if duration_s > 0 else 0.0

    return {
        "status": 200 if error is None else 500,
        "ttft_ms": round(ttft, 2) if ttft is not None else round(duration_ms, 2),
        "duration_ms": round(duration_ms, 2),
        "tokens": token_count,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "error": error,
    }


async def run_concurrency_batch(
    client: httpx.AsyncClient,
    url: str,
    concurrency: int,
    total_requests: int,
    prompt: str,
) -> dict[str, Any]:
    """Run concurrent streaming requests and compute statistical percentiles."""
    sem = asyncio.Semaphore(concurrency)

    async def worker() -> dict[str, Any]:
        async with sem:
            return await run_single_stream(client, url, prompt)

    tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
    t_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time_s = time.perf_counter() - t_start

    successful = [r for r in results if r["status"] == 200 and r["error"] is None]
    errors = len(results) - len(successful)

    if not successful:
        return {
            "concurrency": concurrency,
            "requests": total_requests,
            "errors": errors,
            "ttft_p50_ms": 0.0,
            "ttft_p95_ms": 0.0,
            "duration_p50_ms": 0.0,
            "duration_p95_ms": 0.0,
            "total_tokens": 0,
            "aggregate_tokens_per_sec": 0.0,
            "mean_tokens_per_sec": 0.0,
        }

    ttfts = [float(r["ttft_ms"]) for r in successful if r["ttft_ms"] is not None]
    durations = [float(r["duration_ms"]) for r in successful]
    tokens = [int(r["tokens"]) for r in successful]
    tps = [float(r["tokens_per_sec"]) for r in successful]

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    total_tokens = sum(tokens)
    aggregate_tps = (total_tokens / total_time_s) if total_time_s > 0 else 0.0

    return {
        "concurrency": concurrency,
        "requests": total_requests,
        "errors": errors,
        "ttft_p50_ms": round(statistics.median(ttfts), 2) if ttfts else 0.0,
        "ttft_p95_ms": round(percentile(ttfts, 95.0), 2) if ttfts else 0.0,
        "duration_p50_ms": round(statistics.median(durations), 2),
        "duration_p95_ms": round(percentile(durations, 95.0), 2),
        "total_tokens": total_tokens,
        "aggregate_tokens_per_sec": round(aggregate_tps, 2),
        "mean_tokens_per_sec": round(statistics.mean(tps), 2) if tps else 0.0,
    }


def format_summary(conc: int, requests: int, s: dict[str, Any]) -> str:
    return (
        f"concurrency={conc:<3} req={requests} | "
        f"TTFT p50={s['ttft_p50_ms']:>6.2f}ms p95={s['ttft_p95_ms']:>6.2f}ms | "
        f"dur p50={s['duration_p50_ms']:>6.2f}ms p95={s['duration_p95_ms']:>6.2f}ms | "
        f"tok/s={s['aggregate_tokens_per_sec']:>7.2f} | err={s['errors']}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM endpoint load test")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--base-url", type=str, default=None)
    ap.add_argument("--hardware", type=str, default="AMD Ryzen 7 7730U, 16 logical CPUs")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "runs")
    return ap.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    started = datetime.now(UTC)
    runs: list[dict[str, Any]] = []

    prompt = "Explain California housing price prediction using gradient boosted decision trees."

    if args.base_url:
        async with httpx.AsyncClient(base_url=args.base_url) as client:
            for conc in args.concurrency:
                summary = await run_concurrency_batch(
                    client, "/v1/generate", conc, args.requests, prompt
                )
                runs.append(summary)
                print(format_summary(conc, args.requests, summary))
    else:
        # Standalone ASGI in-process runner with mock tokens
        settings = Settings(model_dir=ROOT / "models", rate_limit_per_minute=0)
        app = create_app(settings)
        sample_tokens = [
            "California",
            " housing",
            " prices",
            " depend",
            " heavily",
            " on",
            " location,",
            " median",
            " income,",
            " and",
            " house",
            " age.",
            " Gradient",
            " boosting",
            " ensembles",
            " shallow",
            " decision",
            " trees",
            " sequentially,",
            " minimizing",
            " squared",
            " error",
            " loss.",
        ]
        app.state.llm_client = MockLLMClient(tokens=sample_tokens, latency_per_token_s=0.005)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for conc in args.concurrency:
                summary = await run_concurrency_batch(
                    client, "/v1/generate", conc, args.requests, prompt
                )
                runs.append(summary)
                print(format_summary(conc, args.requests, summary))

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"{started:%Y-%m-%d}-llm-endpoint-loadtest.json"
    result = {
        "date": f"{started:%Y-%m-%d}",
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": args.hardware,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runs": runs,
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark report to {json_path}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
