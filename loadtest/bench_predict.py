"""Micro-benchmark of model.predict latency inside one process (no HTTP).

Explains where per-request time goes: the number of boosting iterations and the number of
OpenMP threads scikit-learn uses for tiny inputs.

    python loadtest/bench_predict.py --hardware "<description>"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from mlserving.train import load_california

ROOT = Path(__file__).resolve().parents[1]


def time_predict(model: Any, x: Any, repeats: int) -> list[float]:
    model.predict(x)  # warm-up
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict(x)
        out.append((time.perf_counter() - t) * 1000.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, nargs="+", default=[300, 100])
    ap.add_argument("--repeats", type=int, default=300)
    ap.add_argument("--hardware", default="unspecified")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "runs")
    args = ap.parse_args()

    ds = load_california(ROOT / ".cache" / "sklearn_data")
    x_tr, x_te, y_tr, _ = train_test_split(ds.x, ds.y, test_size=0.2, random_state=42)
    rows: list[dict[str, Any]] = []
    for max_iter in args.iters:
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=0.1,
            max_leaf_nodes=31,
            early_stopping=False,
            random_state=42,
        ).fit(x_tr, y_tr)
        for threads in ("default", 1):
            limit = threadpool_limits(limits=None if threads == "default" else threads)
            for n_rows in (1, 32):
                samples = time_predict(model, x_te[:n_rows], args.repeats)
                rows.append(
                    {
                        "max_iter": max_iter,
                        "threads": threads,
                        "rows": n_rows,
                        "median_ms": round(statistics.median(samples), 3),
                        "p95_ms": round(float(np.percentile(samples, 95)), 3),
                    }
                )
                print(rows[-1], flush=True)
            limit.restore_original_limits()
    started = datetime.now(UTC)
    result = {
        "date": f"{started:%Y-%m-%d}",
        "hardware": args.hardware,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "repeats": args.repeats,
        "note": "single process, no HTTP; 'default' = scikit-learn's OpenMP default thread count",
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{started:%Y-%m-%d}-predict-latency.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
