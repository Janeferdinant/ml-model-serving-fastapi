import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from mlserving import registry
from mlserving.schema import FEATURES
from mlserving.train import TrainConfig, load_california, train

from .conftest import fast_config, synthetic_dataset


def test_writes_versioned_artifact_folder(tmp_path: Path) -> None:
    result = train(fast_config(tmp_path, "v1"), dataset=synthetic_dataset())
    folder = tmp_path / "v1"
    assert result.version == "v1"
    assert result.version_dir == folder
    assert {p.name for p in folder.iterdir()} == {
        "model.joblib",
        "metadata.json",
        "reference.json",
    }
    assert (tmp_path / "registry.json").exists()
    assert (folder / "model.joblib").stat().st_size < 5 * 1024 * 1024


def test_metadata_contents(tmp_path: Path) -> None:
    train(fast_config(tmp_path, "v1"), dataset=synthetic_dataset())
    meta = json.loads((tmp_path / "v1" / "metadata.json").read_text())
    assert meta["version"] == "v1"
    assert meta["feature_names"] == list(FEATURES)
    assert [f["name"] for f in meta["feature_schema"]] == list(FEATURES)
    for f in meta["feature_schema"]:
        assert f["min"] <= f["mean"] <= f["max"]
        assert f["std"] >= 0
    assert meta["seed"] == 42
    assert meta["trained_at"].endswith("Z")
    assert "commit" in meta["git"]
    assert meta["metrics"]["test"]["rmse"] > 0
    lo, hi = meta["metrics"]["test"]["rmse_ci95"]
    assert lo <= meta["metrics"]["test"]["rmse"] <= hi
    assert set(meta["metrics"]["baselines"]) == {"mean", "ridge"}
    assert meta["dataset"]["name"] == "synthetic test data"
    assert meta["artifacts"]["model"]["sha256"] == registry.sha256_file(
        tmp_path / "v1" / "model.joblib"
    )
    assert meta["library_versions"]["scikit-learn"]


def test_model_beats_mean_baseline(tmp_path: Path) -> None:
    train(fast_config(tmp_path, "v1"), dataset=synthetic_dataset())
    meta = json.loads((tmp_path / "v1" / "metadata.json").read_text())
    assert meta["metrics"]["test"]["rmse"] < meta["metrics"]["baselines"]["mean"]["rmse"]


def test_seeded_training_is_reproducible(tmp_path: Path) -> None:
    ds = synthetic_dataset()
    a = train(fast_config(tmp_path / "a", "v1", seed=7), dataset=ds)
    b = train(fast_config(tmp_path / "b", "v1", seed=7), dataset=ds)
    assert a.metadata["metrics"] == b.metadata["metrics"]
    xa = joblib.load(a.version_dir / "model.joblib").predict(ds.x[:50])
    xb = joblib.load(b.version_dir / "model.joblib").predict(ds.x[:50])
    np.testing.assert_array_equal(xa, xb)


def test_different_seed_changes_split(tmp_path: Path) -> None:
    ds = synthetic_dataset()
    a = train(fast_config(tmp_path / "a", "v1", seed=1), dataset=ds)
    b = train(fast_config(tmp_path / "b", "v1", seed=2), dataset=ds)
    assert a.metadata["metrics"]["test"]["rmse"] != b.metadata["metrics"]["test"]["rmse"]


def test_auto_version_increments_and_never_overwrites(tmp_path: Path) -> None:
    ds = synthetic_dataset()
    assert train(fast_config(tmp_path), dataset=ds).version == "v1"
    assert train(fast_config(tmp_path), dataset=ds).version == "v2"
    with pytest.raises(FileExistsError):
        train(fast_config(tmp_path, "v1"), dataset=ds)


def test_invalid_version_name_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="version"):
        train(fast_config(tmp_path, "../x"), dataset=synthetic_dataset())


def test_new_version_starts_as_staging(tmp_path: Path) -> None:
    train(fast_config(tmp_path, "v1"), dataset=synthetic_dataset())
    assert registry.load_registry(tmp_path)["versions"]["v1"]["stage"] == "staging"
    assert registry.load_registry(tmp_path)["production"] is None


def test_reference_profile_matches_training_features(tmp_path: Path) -> None:
    train(fast_config(tmp_path, "v1"), dataset=synthetic_dataset())
    ref = json.loads((tmp_path / "v1" / "reference.json").read_text())
    assert list(ref["quantiles"]) == list(FEATURES)
    assert len(ref["quantiles"]["MedInc"]) == 1001


def test_mlflow_logs_params_metrics_and_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    import mlflow

    uri = (tmp_path / "mlruns").resolve().as_uri()
    cfg = TrainConfig(
        out_dir=tmp_path / "models",
        version="v1",
        max_iter=20,
        bootstrap=20,
        mlflow_uri=uri,
        mlflow_experiment="unit-test",
    )
    result = train(cfg, dataset=synthetic_dataset())
    run_id = result.metadata["mlflow"]["run_id"]
    client = mlflow.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.params["seed"] == "42"
    assert run.data.params["max_iter"] == "20"
    assert float(run.data.metrics["test_rmse"]) == pytest.approx(
        result.metadata["metrics"]["test"]["rmse"]
    )
    assert "git_commit" in run.data.tags
    names = {a.path for a in client.list_artifacts(run_id, "model_artifact")}
    assert "model_artifact/model.joblib" in names
    assert "model_artifact/metadata.json" in names


def test_load_california_rejects_unexpected_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from sklearn import datasets

    class Fake:
        data = np.zeros((10, 3))
        target = np.zeros(10)
        feature_names = ("a", "b", "c")

    monkeypatch.setattr(datasets, "fetch_california_housing", lambda **_: Fake())
    with pytest.raises(ValueError, match="features"):
        load_california(Path("."))
