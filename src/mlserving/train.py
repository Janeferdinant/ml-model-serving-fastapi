"""Seeded training script: fits the model, logs to MLflow and writes a versioned artifact."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from numpy.typing import NDArray
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlserving import registry
from mlserving.drift import ReferenceProfile
from mlserving.schema import FEATURE_DESCRIPTIONS, FEATURES, TARGET, TARGET_UNIT

Array = NDArray[np.float64]

CALIFORNIA_SOURCE = {
    "name": "California Housing",
    "loader": "sklearn.datasets.fetch_california_housing",
    "source_url": "https://ndownloader.figshare.com/files/5976036",
    "upstream": "StatLib, https://lib.stat.cmu.edu/datasets/houses.zip",
    "reference": (
        "Pace, R. Kelley and Barry, Ronald (1997). Sparse Spatial Autoregressions. "
        "Statistics and Probability Letters, 33, 291-297."
    ),
    "origin": "1990 U.S. Census, one row per block group",
    "license": "none stated; see THIRD_PARTY_NOTICES.md",
}


@dataclass
class Dataset:
    x: Array
    y: Array
    feature_names: list[str]
    target_name: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    out_dir: Path = Path("models")
    version: str | None = None
    seed: int = 42
    test_size: float = 0.2
    max_iter: int = 300
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    bootstrap: int = 1000
    data_dir: Path = Path(".cache/sklearn_data")
    use_mlflow: bool = True
    mlflow_uri: str | None = None
    mlflow_experiment: str = "california-housing-serving"


@dataclass
class TrainResult:
    version: str
    version_dir: Path
    metadata: dict[str, Any]


def load_california(data_dir: Path) -> Dataset:
    """Download California Housing (scikit-learn verifies the archive's SHA-256)."""
    from sklearn import datasets

    bunch = datasets.fetch_california_housing(data_home=str(data_dir), download_if_missing=True)
    names = list(bunch.feature_names)
    if names != list(FEATURES) or bunch.data.shape[1] != len(FEATURES):
        raise ValueError(f"unexpected features in dataset: {names}")
    provenance = dict(CALIFORNIA_SOURCE)
    try:
        from sklearn.datasets._california_housing import ARCHIVE

        provenance["download_sha256"] = ARCHIVE.checksum
    except ImportError:  # pragma: no cover - private import moved in a future release
        pass
    return Dataset(
        x=np.asarray(bunch.data, dtype=np.float64),
        y=np.asarray(bunch.target, dtype=np.float64),
        feature_names=names,
        target_name=TARGET,
        provenance=provenance,
    )


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, timeout=20
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run(
        "status",
        "--porcelain",
        "--",
        ".",
        ":(exclude)models",
        ":(exclude)mlruns",
        ":(exclude).cache",
    )
    return {"commit": commit or "unknown", "dirty": bool(status) if status is not None else None}


def _metrics(y_true: Array, y_pred: Array) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _bootstrap_rmse_ci(y_true: Array, y_pred: Array, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sq = (y_true - y_pred) ** 2
    idx = rng.integers(0, len(sq), size=(n, len(sq)))
    rmses = np.sqrt(sq[idx].mean(axis=1))
    lo, hi = np.percentile(rmses, [2.5, 97.5])
    return [float(lo), float(hi)]


def _next_version(out_dir: Path) -> str:
    existing = [
        int(p.name[1:]) for p in out_dir.glob("v*") if registry.VERSION_RE.fullmatch(p.name)
    ]
    return f"v{max(existing, default=0) + 1}"


def _library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def _mlflow_run(config: TrainConfig, version: str) -> AbstractContextManager[Any]:
    import os

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = config.mlflow_uri or (Path("mlruns").resolve().as_uri())
    if uri.startswith("file:"):
        # MLflow 3.x puts the local file store in maintenance mode and refuses it unless opted
        # in. The file store is what this project documents; pass --mlflow-uri sqlite:///... to
        # use the database backend instead.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config.mlflow_experiment)
    return mlflow.start_run(run_name=version)


def _mlflow_log(config: TrainConfig, meta: dict[str, Any], version_dir: Path) -> None:
    import mlflow

    test = meta["metrics"]["test"]
    mlflow.log_params(
        {
            "seed": config.seed,
            "test_size": config.test_size,
            "max_iter": config.max_iter,
            "learning_rate": config.learning_rate,
            "max_leaf_nodes": config.max_leaf_nodes,
            "model_type": meta["model_type"],
            "n_train": meta["metrics"]["n_train"],
            "n_test": meta["metrics"]["n_test"],
            "dataset": meta["dataset"]["name"],
            "dataset_content_sha256": meta["dataset"]["content_sha256"],
        }
    )
    mlflow.log_metrics(
        {
            "test_rmse": test["rmse"],
            "test_mae": test["mae"],
            "test_r2": test["r2"],
            "test_rmse_ci_low": test["rmse_ci95"][0],
            "test_rmse_ci_high": test["rmse_ci95"][1],
            "train_rmse": meta["metrics"]["train"]["rmse"],
            "baseline_mean_rmse": meta["metrics"]["baselines"]["mean"]["rmse"],
            "baseline_ridge_rmse": meta["metrics"]["baselines"]["ridge"]["rmse"],
        }
    )
    mlflow.set_tags(
        {
            "version": meta["version"],
            "git_commit": meta["git"]["commit"],
            "git_dirty": str(meta["git"]["dirty"]),
            "model_sha256": meta["artifacts"]["model"]["sha256"],
        }
    )
    mlflow.log_artifacts(str(version_dir), artifact_path="model_artifact")


def train(config: TrainConfig, dataset: Dataset | None = None) -> TrainResult:
    """Train, evaluate on a held-out split, and write ``<out_dir>/<version>/``."""
    out_dir = Path(config.out_dir)
    version = config.version or _next_version(out_dir)
    if not registry.VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid version {version!r}; expected v<number>")
    version_dir = out_dir / version
    if version_dir.exists():
        raise FileExistsError(f"{version_dir} already exists; versions are immutable")

    ds = dataset or load_california(config.data_dir)
    x_tr, x_te, y_tr, y_te = train_test_split(
        ds.x, ds.y, test_size=config.test_size, random_state=config.seed
    )

    model = HistGradientBoostingRegressor(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        early_stopping=False,
        random_state=config.seed,
    )
    model.fit(x_tr, y_tr)
    pred_te = model.predict(x_te)
    baselines: dict[str, dict[str, float]] = {}
    for name, base in (
        ("mean", DummyRegressor(strategy="mean")),
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ):
        base.fit(x_tr, y_tr)
        baselines[name] = _metrics(y_te, base.predict(x_te))

    feature_schema = [
        {
            "name": name,
            "dtype": "float64",
            "min": float(x_tr[:, i].min()),
            "max": float(x_tr[:, i].max()),
            "mean": float(x_tr[:, i].mean()),
            "std": float(x_tr[:, i].std()),
            "description": FEATURE_DESCRIPTIONS.get(name, ""),
        }
        for i, name in enumerate(ds.feature_names)
    ]
    profile = ReferenceProfile.from_matrix(x_tr, ds.feature_names)
    content_sha = hashlib.sha256(
        np.ascontiguousarray(ds.x).tobytes() + np.ascontiguousarray(ds.y).tobytes()
    ).hexdigest()

    run_ctx: AbstractContextManager[Any] = (
        _mlflow_run(config, version) if config.use_mlflow else nullcontext(None)
    )
    with run_ctx as run:
        partial = out_dir / f".{version}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        model_path = partial / "model.joblib"
        joblib.dump(model, model_path, compress=3)
        ref_path = partial / "reference.json"
        ref_path.write_text(json.dumps(profile.to_dict()) + "\n", encoding="utf-8", newline="\n")

        from datetime import UTC, datetime

        meta: dict[str, Any] = {
            "schema_version": 1,
            "version": version,
            "model_type": type(model).__name__,
            "task": "regression",
            "target": {"name": ds.target_name, "unit": TARGET_UNIT},
            "feature_names": list(ds.feature_names),
            "feature_schema": feature_schema,
            "hyperparameters": {
                "max_iter": config.max_iter,
                "learning_rate": config.learning_rate,
                "max_leaf_nodes": config.max_leaf_nodes,
                "early_stopping": False,
            },
            "seed": config.seed,
            "split": {"test_size": config.test_size, "method": "train_test_split, shuffled"},
            "metrics": {
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "train": _metrics(y_tr, model.predict(x_tr)),
                "test": {
                    **_metrics(y_te, pred_te),
                    "rmse_ci95": _bootstrap_rmse_ci(y_te, pred_te, config.bootstrap, config.seed),
                    "bootstrap_resamples": config.bootstrap,
                },
                "baselines": baselines,
            },
            "dataset": {
                **ds.provenance,
                "n_rows": len(ds.y),
                "n_features": int(ds.x.shape[1]),
                "content_sha256": content_sha,
            },
            "git": _git_info(),
            "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "library_versions": _library_versions(),
            "artifacts": {
                "model": {
                    "file": "model.joblib",
                    "sha256": registry.sha256_file(model_path),
                    "bytes": model_path.stat().st_size,
                },
                "reference": {
                    "file": "reference.json",
                    "sha256": registry.sha256_file(ref_path),
                    "bytes": ref_path.stat().st_size,
                },
            },
        }
        if run is not None:
            meta["mlflow"] = {"run_id": run.info.run_id, "experiment": config.mlflow_experiment}
        (partial / "metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        partial.rename(version_dir)
        if run is not None:
            _mlflow_log(config, meta, version_dir)

    registry.register_version(out_dir, version, meta)
    return TrainResult(version=version, version_dir=version_dir, metadata=meta)
