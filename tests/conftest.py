"""Shared fixtures. Every test runs offline on a small synthetic dataset."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlserving.app import create_app
from mlserving.config import Settings
from mlserving.registry import promote
from mlserving.schema import FEATURES
from mlserving.train import Dataset, TrainConfig, train

VALID_PAYLOAD: dict[str, float] = {
    "MedInc": 3.5,
    "HouseAge": 28.0,
    "AveRooms": 5.4,
    "AveBedrms": 1.1,
    "Population": 1400.0,
    "AveOccup": 3.0,
    "Latitude": 34.2,
    "Longitude": -118.4,
}


def synthetic_dataset(n: int = 1500, seed: int = 0) -> Dataset:
    """Synthetic stand-in for California Housing with similar feature ranges."""
    rng = np.random.default_rng(seed)
    x = np.column_stack(
        [
            np.clip(rng.lognormal(1.2, 0.5, n), 0.5, 15.0),  # MedInc
            rng.integers(1, 53, n).astype(float),  # HouseAge
            np.clip(rng.normal(5.4, 1.2, n), 2.0, 12.0),  # AveRooms
            np.clip(rng.normal(1.1, 0.1, n), 0.8, 2.0),  # AveBedrms
            np.clip(rng.lognormal(7.0, 0.6, n), 5.0, 20000.0),  # Population
            np.clip(rng.normal(3.0, 0.7, n), 1.0, 10.0),  # AveOccup
            rng.uniform(32.5, 42.0, n),  # Latitude
            rng.uniform(-124.3, -114.3, n),  # Longitude
        ]
    )
    y = np.clip(0.45 * x[:, 0] + 0.01 * x[:, 1] + rng.normal(0, 0.3, n), 0.15, 5.0)
    return Dataset(
        x=x,
        y=y,
        feature_names=list(FEATURES),
        target_name="MedHouseVal",
        provenance={"name": "synthetic test data", "source_url": "n/a"},
    )


def fast_config(out_dir: Path, version: str | None = None, seed: int = 42) -> TrainConfig:
    return TrainConfig(
        out_dir=out_dir,
        version=version,
        seed=seed,
        max_iter=30,
        bootstrap=50,
        use_mlflow=False,
    )


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A model directory with v1 (production) and v2 (staging)."""
    d = tmp_path_factory.mktemp("models")
    ds = synthetic_dataset()
    train(fast_config(d, "v1", seed=1), dataset=ds)
    train(fast_config(d, "v2", seed=2), dataset=ds)
    promote(d, "v1")
    return d


@pytest.fixture
def model_dir_copy(model_dir: Path, tmp_path: Path) -> Path:
    """A private, mutable copy of the model directory."""
    target = tmp_path / "models"
    shutil.copytree(model_dir, target)
    return target


@pytest.fixture
def settings(model_dir: Path) -> Settings:
    return Settings(model_dir=model_dir, rate_limit_per_minute=0)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict[str, float]:
    return dict(VALID_PAYLOAD)
