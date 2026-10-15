# Sync vs async endpoints, interleaved comparison (2026-10-15)

Verified (run 2026-10-15, AMD Ryzen 7 5800X (8 cores, 16 threads), 32 GB RAM, Windows 11 Pro), with a caveat: other jobs were using the machine (it was never idle), so absolute numbers are low and noisy. The variants were run alternately (async, sync, async, sync, ...) to spread that noise evenly.

Question: should the endpoints be `def` (FastAPI runs them in a thread pool) or `async def` with `predict` called inline on the event loop? Setup: one uvicorn worker, 16 connections, single-row `/predict`, 10 s per run, 1 repetition per run, otherwise the settings of `2026-10-15-loadtest.md`.

| Pair | async def: req/s (p50 / p99 ms) | def (thread pool): req/s (p50 / p99 ms) |
|---|---|---|
| 1 | 101.0 (140.8 / 435.9) | 122.5 (130.4 / 185.7) |
| 2 | 136.0 (111.9 / 162.4) | 175.0 (90.2 / 122.1) |
| 3 | 117.0 (135.8 / 184.3) | 167.0 (93.6 / 143.0) |

Result: the thread-pool variant was faster in all three pairs, so the endpoints stay `def`. Three pairs on a shared machine is weak evidence; it is enough to reject the change, not to quantify the difference.
