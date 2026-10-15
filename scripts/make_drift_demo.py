"""Precompute the drift demo shown in docs/index.html.

Held-out California Housing rows (the model's own test split, loaded at run time and never
committed) are sent through the real FastAPI app with request logging on. Three windows are
used: unchanged rows, a mild simulated shift and a strong simulated shift. Each request log is
then analysed with the drift CLI, exactly as in production. Only aggregate statistics and
histogram proportions are written to the repository.

    python scripts/make_drift_demo.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from mlserving import registry
from mlserving.app import create_app
from mlserving.cli import main as cli_main
from mlserving.config import Settings
from mlserving.drift import ReferenceProfile, compute_drift, render_html
from mlserving.schema import FEATURES
from mlserving.train import load_california

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch" / "drift"
N_ROWS = 1000
COL = {name: i for i, name in enumerate(FEATURES)}


def scenario_rows(kind: str, x: np.ndarray) -> np.ndarray:
    out = x.copy()
    if kind == "mild":
        out[:, COL["MedInc"]] += 0.4
    elif kind == "strong":
        out[:, COL["MedInc"]] = out[:, COL["MedInc"]] * 1.4 + 0.6
        out[:, COL["HouseAge"]] -= 10.0
        out[:, COL["AveOccup"]] *= 1.3
    return out


SCENARIOS = [
    ("none", "No drift", "1,000 held-out rows, unchanged."),
    (
        "mild",
        "Mild simulated drift",
        "MedInc raised by 0.4 (about 0.2 standard deviations) on 1,000 held-out rows.",
    ),
    (
        "strong",
        "Strong simulated drift",
        "MedInc x 1.4 + 0.6, HouseAge minus 10 years and AveOccup x 1.3 on 1,000 held-out rows.",
    ),
]


def calibration(
    profile: ReferenceProfile, pool: np.ndarray, sigma_income: float, lo: np.ndarray, hi: np.ndarray
) -> dict[str, object]:
    """How often windows are flagged, unchanged and with MedInc shifted by k std devs."""
    rng = np.random.default_rng(1)
    n_windows = 200
    out: list[dict[str, object]] = []
    for k in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        warn = drift = 0
        for _ in range(n_windows):
            w = pool[rng.choice(len(pool), size=N_ROWS, replace=False)].copy()
            w[:, COL["MedInc"]] += k * sigma_income
            status = compute_drift(profile, np.clip(w, lo, hi))["overall_status"]
            warn += status in ("warning", "drift")
            drift += status == "drift"
        out.append(
            {
                "shift_sd": k,
                "flagged_warning_or_drift": warn / n_windows,
                "flagged_drift": drift / n_windows,
            }
        )
    return {
        "windows_per_shift": n_windows,
        "rows_per_window": N_ROWS,
        "feature": "MedInc",
        "note": (
            "Windows are random 1,000-row draws from the 4,128 held-out rows (so they overlap); "
            "the shift is added to MedInc in units of its training standard deviation, then "
            "clipped to the training range. 'flagged' = any feature at warning or drift."
        ),
        "rows": out,
    }


def main() -> int:
    model_dir = ROOT / "models"
    version = registry.resolve_version(model_dir, "production")
    meta = registry.verify_version(model_dir, version)
    schema = meta["feature_schema"]
    lo = np.array([f["min"] for f in schema])
    hi = np.array([f["max"] for f in schema])

    ds = load_california(ROOT / ".cache" / "sklearn_data")
    _, x_te, _, _ = train_test_split(
        ds.x, ds.y, test_size=meta["split"]["test_size"], random_state=meta["seed"]
    )
    profile = ReferenceProfile.from_dict(
        json.loads(
            (model_dir / version / meta["artifacts"]["reference"]["file"]).read_text(
                encoding="utf-8"
            )
        )
    )
    rng = np.random.default_rng(0)
    base = x_te[rng.choice(len(x_te), size=N_ROWS, replace=False)]

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    results = []
    for kind, name, description in SCENARIOS:
        rows = np.clip(scenario_rows(kind, base), lo, hi)  # stay inside the service's range guard
        log = SCRATCH / f"{kind}.jsonl"
        settings = Settings(model_dir=model_dir, request_log_path=log, rate_limit_per_minute=0)
        with TestClient(create_app(settings)) as client:
            for start in range(0, len(rows), 100):
                chunk = [
                    dict(zip(FEATURES, map(float, r), strict=True))
                    for r in rows[start : start + 100]
                ]
                resp = client.post("/predict/batch", json={"instances": chunk})
                assert resp.status_code == 200, resp.text
        out = SCRATCH / kind
        code = cli_main(
            ["drift", "--model-dir", str(model_dir), "--log", str(log), "--out", str(out)]
        )
        assert code == 0
        report = json.loads((out / "drift_report.json").read_text(encoding="utf-8"))
        report["window"]["source"] = f"{kind}.jsonl (simulated window)"
        report["window"].pop("first_ts", None)
        report["window"].pop("last_ts", None)
        results.append({"id": kind, "name": name, "description": description, "report": report})

    strong = next(r for r in results if r["id"] == "strong")["report"]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "drift_report_strong.html").write_text(
        render_html(strong), encoding="utf-8", newline="\n"
    )
    payload = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "model_version": version,
        "seed": meta["seed"],
        "rows_per_window": N_ROWS,
        "note": (
            "Simulated: held-out test rows (never committed) were perturbed as described, "
            "sent through the service, logged and analysed with the drift CLI."
        ),
        "scenarios": results,
        "calibration": calibration(profile, x_te, profile.std["MedInc"], lo, hi),
    }
    out_path = ROOT / "docs" / "data" / "drift_demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        rep = r["report"]
        print(
            r["id"],
            rep["overall_status"],
            "drifted:",
            rep["drifted_features"],
            "warning:",
            rep["warning_features"],
        )
    for row in payload["calibration"]["rows"]:
        print("calibration", row)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
