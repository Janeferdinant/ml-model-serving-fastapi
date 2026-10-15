"""The features the service logs must feed the drift module end to end."""

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from mlserving import drift
from mlserving.app import create_app
from mlserving.cli import main
from mlserving.config import Settings
from mlserving.schema import FEATURES

from .conftest import synthetic_dataset


def serve_rows(model_dir: Path, log: Path, rows: np.ndarray) -> np.ndarray:
    """Send rows through the API and return what was actually sent (clipped to the schema).

    The range guard rejects values outside the training range, so synthetic traffic is
    clipped to it first; drift that stays inside the range is what the service can log.
    """
    schema = json.loads((model_dir / "v1" / "metadata.json").read_text())["feature_schema"]
    rows = np.clip(rows, [f["min"] for f in schema], [f["max"] for f in schema])
    settings = Settings(model_dir=model_dir, request_log_path=log, rate_limit_per_minute=0)
    with TestClient(create_app(settings)) as c:
        for start in range(0, len(rows), 100):
            chunk = rows[start : start + 100]
            payload = {
                "instances": [dict(zip(FEATURES, map(float, r), strict=True)) for r in chunk]
            }
            assert c.post("/predict/batch", json=payload).status_code == 200
    return rows


def fresh_rows(seed: int, n: int) -> np.ndarray:
    return synthetic_dataset(n=n, seed=seed).x


def reference_profile(model_dir: Path) -> drift.ReferenceProfile:
    return drift.ReferenceProfile.from_dict(
        json.loads((model_dir / "v1" / "reference.json").read_text())
    )


def test_logged_features_round_trip_into_the_drift_module(model_dir: Path, tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    rows = serve_rows(model_dir, log, fresh_rows(seed=5, n=300))
    window = drift.load_request_log(log, list(FEATURES))
    assert window.x.shape == (300, len(FEATURES))
    np.testing.assert_allclose(window.x, rows)
    report = drift.compute_drift(reference_profile(model_dir), window.x, model_version="v1")
    assert report["n_current"] == 300
    assert [f["name"] for f in report["features"]] == list(FEATURES)


def test_in_distribution_traffic_does_not_trigger_drift(model_dir: Path, tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    serve_rows(model_dir, log, fresh_rows(seed=6, n=1500))
    window = drift.load_request_log(log, list(FEATURES))
    report = drift.compute_drift(reference_profile(model_dir), window.x)
    assert report["drifted_features"] == []


def test_shifted_traffic_through_the_service_is_detected(model_dir: Path, tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    rows = fresh_rows(seed=7, n=1500)
    rows[:, 0] = np.clip(rows[:, 0] * 1.4 + 0.8, 0.5, 9.0)  # MedInc up, inside the range guard
    rows[:, 1] = np.clip(rows[:, 1] - 20.0, 1.0, 52.0)  # HouseAge down
    serve_rows(model_dir, log, rows)
    window = drift.load_request_log(log, list(FEATURES))
    report = drift.compute_drift(reference_profile(model_dir), window.x)
    assert {"MedInc", "HouseAge"} <= set(report["drifted_features"])
    assert "Latitude" not in report["drifted_features"]
    assert report["overall_status"] == "drift"


def test_cli_writes_reports_and_exit_code(model_dir: Path, tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    rows = fresh_rows(seed=8, n=800)
    rows[:, 2] = np.clip(rows[:, 2] + 2.0, 2.0, 9.0)  # AveRooms shifted, inside the range guard
    serve_rows(model_dir, log, rows)
    out = tmp_path / "report"
    args = ["drift", "--model-dir", str(model_dir), "--log", str(log), "--out", str(out)]
    assert main(args) == 0
    data = json.loads((out / "drift_report.json").read_text())
    assert "AveRooms" in data["drifted_features"]
    assert (out / "drift_report.html").exists()
    assert main([*args, "--fail-on-drift"]) == 2
    assert main([*args, "--last", "100"]) == 0


def test_cli_fails_cleanly_on_too_few_rows(model_dir: Path, tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    serve_rows(model_dir, log, fresh_rows(seed=9, n=10))
    code = main(
        ["drift", "--model-dir", str(model_dir), "--log", str(log), "--out", str(tmp_path / "o")]
    )
    assert code == 1
