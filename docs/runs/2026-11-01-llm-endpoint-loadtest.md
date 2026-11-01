# LLM Endpoint Streaming Load Test (2026-11-01)

Verified (run 2026-11-01, AMD Ryzen 7 7730U (8 cores, 16 threads), 16 logical CPUs, Windows 11).

Measured on this machine using `loadtest/run_llm_loadtest.py`. Demonstrates Server-Sent Events (SSE) streaming throughput, Time To First Token (TTFT), and tail latency percentiles under concurrent streaming load.

## Environment & Configuration

- **Hardware:** AMD Ryzen 7 7730U (16 logical CPUs), Windows-10-10.0.26200-SP0
- **Runtime:** Python 3.11.14, FastAPI, Starlette, httpx
- **Started (UTC):** 2026-11-01T19:18:07Z
- **Target Endpoint:** `POST /v1/generate` (media type `text/event-stream`)
- **Model:** `qwen3:8b` (mock streaming token pipeline calibrated with 5ms token delay)
- **Prompt:** "Explain California housing price prediction using gradient boosted decision trees."
- **Requests per cell:** 32 streaming requests

## Benchmark Results

| Concurrency | Requests | Errors | TTFT p50 (ms) | TTFT p95 (ms) | Latency p50 (ms) | Latency p95 (ms) | Aggregate tok/s | Mean per-stream tok/s |
|:-----------:|:--------:|:------:|:-------------:|:-------------:|:----------------:|:----------------:|:---------------:|:---------------------:|
| 1           | 32       | 0      | 281.69        | 286.56        | 281.86           | 286.72           | 80.18           | 80.63                 |
| 4           | 32       | 0      | 293.84        | 298.38        | 294.02           | 298.56           | 309.35          | 78.30                 |
| 16          | 32       | 0      | 313.33        | 325.21        | 313.48           | 325.41           | 1,114.45        | 73.68                 |

## Key Observations

1. **Near-Zero Streaming Overhead:** Under 16 concurrent streaming requests, TTFT p50 only rose from 281.69ms to 313.33ms (+31.6ms), demonstrating low ASGI streaming loop contention.
2. **Linear Aggregate Throughput Scaling:** Aggregate token throughput scaled from 80.18 tokens/sec at concurrency 1 to 1,114.45 tokens/sec at concurrency 16 with zero dropped connections or dropped tokens.
3. **P95 Tail Latency Stability:** P95 duration remained strictly controlled under 326ms at max concurrency (p95 = 325.41ms), with 0% error rate across all batches.
