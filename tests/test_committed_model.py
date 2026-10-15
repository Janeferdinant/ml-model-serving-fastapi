"""The model artifacts that ship in the repository (and in the Docker image) must be healthy."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlserving import registry
from mlserving.app import create_app
from mlserving.config import Settings

from .conftest import VALID_PAYLOAD

MODELS = Path(__file__).resolve().parents[1] / "models"
VERSIONS = ["v1", "v2"]


@pytest.mark.parametrize("version", VERSIONS)
def test_committed_versions_are_small_and_verified(version: str) -> None:
    meta = registry.verify_version(MODELS, version)
    assert (MODELS / version / "model.joblib").stat().st_size < 5 * 1024 * 1024
    assert meta["dataset"]["name"] == "California Housing"
    assert meta["metrics"]["test"]["rmse"] < meta["metrics"]["baselines"]["mean"]["rmse"]
    assert meta["git"]["commit"]
    assert meta["trained_at"]


def test_v2_is_in_production_and_v1_is_archived() -> None:
    reg = registry.load_registry(MODELS)
    assert reg["production"] == "v2"
    assert reg["versions"]["v2"]["stage"] == "production"
    assert reg["versions"]["v1"]["stage"] == "archived"
    assert [h["action"] for h in reg["history"]].count("promote") == 2


def test_committed_model_serves_plausible_predictions() -> None:
    settings = Settings(model_dir=MODELS, rate_limit_per_minute=0)
    with TestClient(create_app(settings)) as c:
        assert c.get("/ready").json()["status"] == "ready"
        assert c.get("/model-info").json()["version"] == "v2"
        rich = c.post("/predict", json={**VALID_PAYLOAD, "MedInc": 9.0}).json()["prediction"]
        poor = c.post("/predict", json={**VALID_PAYLOAD, "MedInc": 1.5}).json()["prediction"]
    assert 0.0 < poor < rich < 6.0


@pytest.mark.parametrize("version", VERSIONS)
def test_committed_metadata_has_no_local_paths(version: str) -> None:
    text = (MODELS / version / "metadata.json").read_text()
    assert "Users" not in text
    json.loads(text)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("name", ["model.joblib", "metadata.json", "reference.json"])
def test_expected_files_exist(version: str, name: str) -> None:
    assert (MODELS / version / name).is_file()
