"""Closed-loop HTTP load test against a locally started uvicorn (aiohttp client).

For each worker count it starts ``uvicorn mlserving.app:create_app --factory`` on localhost,
warms it up, then runs every (scenario, concurrency) combination several times and records
p50/p95/p99 latency and throughput. The client runs on the same machine as the server, in
several processes so the client is not the bottleneck on a single core.

    python loadtest/run_loadtest.py --workers 1 4 --hardware "<description>"

The results are a measurement of this machine and this configuration only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import psutil

from mlserving import registry

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def make_payloads(seed: int, n: int, batch: int) -> tuple[list[bytes], list[bytes]]:
    """Pre-serialised request bodies drawn from the training distribution."""
    version = registry.resolve_version(ROOT / "models", "production")
    meta = json.loads((ROOT / "models" / version / "metadata.json").read_text(encoding="utf-8"))
    schema = {f["name"]: f for f in meta["feature_schema"]}
    rng = np.random.default_rng(seed)

    def row() -> dict[str, float]:
        out = {}
        for name in FEATURES:
            s = schema[name]
            out[name] = float(np.clip(rng.normal(s["mean"], s["std"]), s["min"], s["max"]))
        return out

    single = [json.dumps(row()).encode() for _ in range(n)]
    batches = [json.dumps({"instances": [row() for _ in range(batch)]}).encode() for _ in range(n)]
    return single, batches


async def _client(
    url: str, bodies: list[bytes], deadline: float, lat: list[float], errors: dict[str, int]
) -> None:
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=0)
    headers = {"Content-Type": "application/json"}
    i = 0
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while time.perf_counter() < deadline:
            body = bodies[i % len(bodies)]
            i += 1
            start = time.perf_counter()
            kind = None
            try:
                async with session.post(url, data=body, headers=headers) as resp:
                    await resp.read()
                    if resp.status != 200:
                        kind = f"http_{resp.status}"
            except (aiohttp.ClientError, TimeoutError) as exc:
                kind = type(exc).__name__
            elapsed = time.perf_counter() - start
            if kind is None:
                lat.append(elapsed)
            else:
                errors[kind] = errors.get(kind, 0) + 1


async def _run_async(url: str, bodies: list[bytes], tasks: int, seconds: float) -> dict[str, Any]:
    lat: list[float] = []
    errors: dict[str, int] = {}
    t0 = time.perf_counter()
    deadline = t0 + seconds
    await asyncio.gather(*[_client(url, bodies, deadline, lat, errors) for _ in range(tasks)])
    return {"latencies": lat, "errors": errors, "wall": time.perf_counter() - t0}


def _worker(args: tuple[str, list[bytes], int, float]) -> dict[str, Any]:
    url, bodies, tasks, seconds = args
    return asyncio.run(_run_async(url, bodies, tasks, seconds))


def run_load(
    pool: Any, url: str, bodies: list[bytes], concurrency: int, procs: int, seconds: float
) -> dict[str, Any]:
    per_proc = [concurrency // procs + (1 if i < concurrency % procs else 0) for i in range(procs)]
    jobs = [(url, bodies, t, seconds) for t in per_proc if t > 0]
    psutil.cpu_percent(interval=None)  # reset the system-wide CPU counter
    results = pool.map(_worker, jobs)
    system_cpu = psutil.cpu_percent(interval=None)
    lat = np.concatenate([np.asarray(r["latencies"], dtype=float) for r in results])
    wall = max(r["wall"] for r in results)
    kinds: dict[str, int] = {}
    for r in results:
        for k, v in r["errors"].items():
            kinds[k] = kinds.get(k, 0) + v
    errors = sum(kinds.values())
    ms = lat * 1000.0
    return {
        "requests_ok": int(lat.size),
        "errors": int(errors),
        "error_kinds": kinds,
        "wall_seconds": round(wall, 3),
        "system_cpu_percent": system_cpu,
        "throughput_rps": round(lat.size / wall, 1),
        "latency_ms": {
            "p50": round(float(np.percentile(ms, 50)), 3),
            "p95": round(float(np.percentile(ms, 95)), 3),
            "p99": round(float(np.percentile(ms, 99)), 3),
            "max": round(float(ms.max()), 3),
            "mean": round(float(ms.mean()), 3),
        },
    }


def wait_quiet(threshold: float, deadline: float) -> float:
    """Wait until system-wide CPU use stays below ``threshold`` percent (other jobs may share
    this machine) or the shared ``deadline`` (monotonic seconds) passes. Returns the last
    measured background utilisation."""
    calm = 0
    last = 0.0
    while True:
        last = psutil.cpu_percent(interval=2.0)
        calm = calm + 1 if last < threshold else 0
        if calm >= 3 or time.monotonic() > deadline:
            return float(last)
        print(f"  waiting for a quiet machine (CPU {last:.0f}%)", flush=True)


def start_server(workers: int, port: int) -> subprocess.Popen[bytes]:
    env = {**os.environ, "MLS_RATE_LIMIT_PER_MINUTE": "0", "MLS_MODEL_DIR": str(ROOT / "models")}
    env.pop("MLS_API_KEY", None)
    env.pop("MLS_REQUEST_LOG_PATH", None)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "mlserving.app:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--no-access-log",
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    """Stop only the server this script started, by PID, together with its worker processes."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def wait_ready(port: int, workers: int, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    ok = 0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2) as r:
                ok = ok + 1 if r.status == 200 else 0
        except OSError:
            ok = 0
        if ok >= 3 * workers:  # several consecutive successes: all workers have started
            return
        time.sleep(0.3)
    raise SystemExit("server did not become ready")


def merge_kinds(reps: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in reps:
        for k, v in r["error_kinds"].items():
            out[k] = out.get(k, 0) + v
    return out


def summarise(reps: list[dict[str, Any]]) -> dict[str, Any]:
    def med(path: tuple[str, ...]) -> float:
        vals = []
        for r in reps:
            v: Any = r
            for k in path:
                v = v[k]
            vals.append(v)
        return round(float(statistics.median(vals)), 3)

    return {
        "reps": len(reps),
        "throughput_rps_median": med(("throughput_rps",)),
        "throughput_rps_min": min(r["throughput_rps"] for r in reps),
        "throughput_rps_max": max(r["throughput_rps"] for r in reps),
        "p50_ms_median": med(("latency_ms", "p50")),
        "p95_ms_median": med(("latency_ms", "p95")),
        "p99_ms_median": med(("latency_ms", "p99")),
        "errors_total": sum(r["errors"] for r in reps),
        "error_kinds": merge_kinds(reps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[16, 64])
    ap.add_argument(
        "--scenarios", nargs="+", default=["single", "batch"], choices=["single", "batch"]
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--duration", type=float, default=15.0, help="seconds per repetition")
    ap.add_argument("--warmup", type=float, default=4.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--client-procs", type=int, default=4)
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--wait-quiet",
        type=float,
        default=0.0,
        help="before each cell wait until background CPU is below this percent",
    )
    ap.add_argument(
        "--wait-budget",
        type=float,
        default=1800.0,
        help="total seconds the run may spend waiting for a quiet machine",
    )
    ap.add_argument("--hardware", default="unspecified")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "runs")
    ap.add_argument("--name", default=None, help="output file stem (default: <date>-loadtest)")
    args = ap.parse_args()

    started = datetime.now(UTC)
    stem = args.name or f"{started:%Y-%m-%d}-loadtest"
    single, batches = make_payloads(args.seed, 500, args.batch_size)
    bodies = {"single": single, "batch": batches}
    paths = {"single": "/predict", "batch": "/predict/batch"}

    import sklearn

    version = registry.resolve_version(ROOT / "models", "production")
    result: dict[str, Any] = {
        "date": f"{started:%Y-%m-%d}",
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": args.hardware,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "settings": {
            "server": "uvicorn --factory mlserving.app:create_app --no-access-log, 127.0.0.1",
            "app_config": "MLS_RATE_LIMIT_PER_MINUTE=0 (limiter off), auth off, request log off, "
            "JSON access log on (stdout discarded), MLS_PREDICT_THREADS=1 (default)",
            "model": f"{version} (production)",
            "client": f"aiohttp, {args.client_procs} client processes on the same machine, "
            "keep-alive connections, closed loop (no think time)",
            "duration_seconds_per_rep": args.duration,
            "warmup_seconds": args.warmup,
            "repetitions": args.reps,
            "batch_size": args.batch_size,
            "payloads": "500 pre-serialised bodies drawn from N(mean, std) clipped to the "
            "training range, seed " + str(args.seed),
        },
        "runs": [],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{stem}.json"
    runs: list[dict[str, Any]] = result["runs"]

    def save() -> None:
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    quiet_deadline = time.monotonic() + args.wait_budget
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.client_procs) as pool:
        for workers in args.workers:
            server = start_server(workers, args.port)
            try:
                wait_ready(args.port, workers)
                base = f"http://127.0.0.1:{args.port}"
                run_load(pool, base + "/predict", single, 8, args.client_procs, args.warmup)
                for scenario in args.scenarios:
                    for conc in args.concurrency:
                        background = (
                            wait_quiet(args.wait_quiet, quiet_deadline) if args.wait_quiet else None
                        )
                        reps = [
                            run_load(
                                pool,
                                base + paths[scenario],
                                bodies[scenario],
                                conc,
                                args.client_procs,
                                args.duration,
                            )
                            for _ in range(args.reps)
                        ]
                        entry: dict[str, Any] = {
                            "workers": workers,
                            "scenario": scenario,
                            "rows_per_request": 1 if scenario == "single" else args.batch_size,
                            "concurrency": conc,
                            "background_cpu_percent_before": background,
                            "repetitions": reps,
                            "summary": summarise(reps),
                        }
                        runs.append(entry)
                        save()
                        s = entry["summary"]
                        print(
                            f"workers={workers} {scenario:<6} conc={conc:<4} "
                            f"{s['throughput_rps_median']:>8.1f} req/s  "
                            f"p50 {s['p50_ms_median']:.2f}  "
                            f"p95 {s['p95_ms_median']:.2f}  p99 {s['p99_ms_median']:.2f} ms  "
                            f"errors {s['errors_total']}",
                            flush=True,
                        )
            finally:
                stop_server(server)
                time.sleep(2)

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
