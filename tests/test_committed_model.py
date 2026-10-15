"""The model artifact that ships in the repository (and in the Docker image) must be healthy."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlserving import registry
from mlserving.app import create_app
from mlserving.config import Settings

from .conftest import VALID_PAYLOAD

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_committed_model_is_small_and_verified() -> None:
    meta = registry.verify_version(MODELS, "v1")
    assert (MODELS / "v1" / "model.joblib").stat().st_size < 5 * 1024 * 1024
    assert meta["dataset"]["name"] == "California Housing"
    assert meta["metrics"]["test"]["rmse"] < meta["metrics"]["baselines"]["mean"]["rmse"]
    assert meta["git"]["commit"]
    assert meta["trained_at"]
    assert registry.load_registry(MODELS)["production"] == "v1"


def test_committed_model_serves_plausible_predictions() -> None:
    settings = Settings(model_dir=MODELS, rate_limit_per_minute=0)
    with TestClient(create_app(settings)) as c:
        assert c.get("/ready").json()["status"] == "ready"
        rich = c.post("/predict", json={**VALID_PAYLOAD, "MedInc": 9.0}).json()["prediction"]
        poor = c.post("/predict", json={**VALID_PAYLOAD, "MedInc": 1.5}).json()["prediction"]
    assert 0.0 < poor < rich < 6.0


def test_committed_metadata_has_no_local_paths() -> None:
    text = (MODELS / "v1" / "metadata.json").read_text()
    assert "Users" not in text
    json.loads(text)


def test_registry_file_is_valid() -> None:
    reg = registry.load_registry(MODELS)
    assert reg["versions"]["v1"]["stage"] == "production"
    assert reg["history"], "promotion should be recorded"


@pytest.mark.parametrize("name", ["model.joblib", "metadata.json", "reference.json"])
def test_expected_files_exist(name: str) -> None:
    assert (MODELS / "v1" / name).is_file()
