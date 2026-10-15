"""Smoke test for a running service (standard library only).

Used by ``scripts/docker_smoke.sh`` against a container and runnable against a local
uvicorn: ``python scripts/smoke_test.py --base-url http://127.0.0.1:8000``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

PAYLOAD = {
    "MedInc": 3.5,
    "HouseAge": 28.0,
    "AveRooms": 5.4,
    "AveBedrms": 1.1,
    "Population": 1400.0,
    "AveOccup": 3.0,
    "Latitude": 34.2,
    "Longitude": -118.4,
}


def call(
    base: str, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def wait_ready(base: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, _, body = call(base, "GET", "/ready")
            if status == 200:
                return
            last = f"{status} {body[:200]!r}"
        except OSError as exc:
            last = repr(exc)
        time.sleep(1)
    raise SystemExit(f"service not ready after {seconds:.0f}s: {last}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None, help="send as X-API-Key; also checks 401")
    parser.add_argument("--wait", type=float, default=60.0, help="seconds to wait for /ready")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    auth = {"X-API-Key": args.api_key} if args.api_key else {}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail and not ok else ''}")
        if not ok:
            failures.append(name)

    wait_ready(base, args.wait)

    status, _, body = call(base, "GET", "/health")
    check("GET /health is 200", status == 200 and json.loads(body) == {"status": "ok"})

    status, _, body = call(base, "GET", "/ready")
    ready = json.loads(body)
    check("GET /ready reports a verified model", status == 200 and ready["checksum_verified"])

    if args.api_key:
        status, _, _ = call(base, "POST", "/predict", PAYLOAD)
        check("POST /predict without key is 401", status == 401, str(status))

    status, headers, body = call(base, "POST", "/predict", PAYLOAD, auth)
    pred = json.loads(body) if status == 200 else {}
    check(
        "POST /predict returns a plausible value",
        status == 200 and 0.0 < pred.get("prediction", -1) < 6.0,
        f"{status} {body[:200]!r}",
    )
    check(
        "response carries X-Request-ID",
        bool(headers.get("x-request-id") or headers.get("X-Request-ID")),
    )

    status, _, body = call(base, "POST", "/predict/batch", {"instances": [PAYLOAD] * 3}, auth)
    batch = json.loads(body) if status == 200 else {}
    check("POST /predict/batch returns 3 predictions", batch.get("count") == 3, f"{status}")
    check(
        "batch prediction equals single prediction",
        bool(batch) and batch["predictions"][0] == pred.get("prediction"),
    )

    status, _, body = call(base, "POST", "/predict", {**PAYLOAD, "MedInc": "high"}, auth)
    check("invalid input is a 422 with a location", status == 422 and "MedInc" in body.decode())

    status, _, body = call(base, "POST", "/predict", {**PAYLOAD, "MedInc": 1e6}, auth)
    check("out-of-range input is a 422", status == 422 and b"training range" in body)

    status, _, body = call(base, "GET", "/model-info", None, auth)
    info = json.loads(body) if status == 200 else {}
    check("GET /model-info names the model version", info.get("version", "").startswith("v"))

    status, headers, body = call(base, "GET", "/metrics")
    text = body.decode()
    check(
        "GET /metrics is Prometheus text with request counts and latency buckets",
        status == 200
        and "http_requests_total" in text
        and "http_request_duration_seconds_bucket" in text
        and "model_prediction_value_bucket" in text,
    )

    if failures:
        print(f"SMOKE FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
