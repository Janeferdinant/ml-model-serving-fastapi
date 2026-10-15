"""Concurrency smoke test: many simultaneous requests, consistent results and counters."""

import re
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

THREADS = 16
PER_THREAD = 20


def test_parallel_requests_are_consistent(
    client: TestClient, valid_payload: dict[str, float]
) -> None:
    baseline = client.post("/predict", json=valid_payload).json()["prediction"]

    def worker(i: int) -> list[tuple[int, float | None]]:
        out: list[tuple[int, float | None]] = []
        for j in range(PER_THREAD):
            if (i + j) % 4 == 0:
                r = client.post("/predict/batch", json={"instances": [valid_payload] * 3})
                out.append((r.status_code, r.json()["predictions"][0] if r.is_success else None))
            else:
                r = client.post("/predict", json=valid_payload)
                out.append((r.status_code, r.json()["prediction"] if r.is_success else None))
        return out

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = [item for chunk in pool.map(worker, range(THREADS)) for item in chunk]

    assert len(results) == THREADS * PER_THREAD
    assert all(code == 200 for code, _ in results)
    assert all(value == baseline for _, value in results)

    text = client.get("/metrics").text
    total = THREADS * PER_THREAD
    ok = re.search(r'http_requests_total\{method="POST",path="/predict",status="200"\} (\d+)', text)
    batch = re.search(
        r'http_requests_total\{method="POST",path="/predict/batch",status="200"\} (\d+)', text
    )
    assert ok
    assert batch
    assert int(ok.group(1)) + int(batch.group(1)) == total + 1
