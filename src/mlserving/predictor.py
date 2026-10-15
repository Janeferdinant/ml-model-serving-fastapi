"""Load a verified model version and run predictions."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from mlserving import registry
from mlserving.registry import ArtifactError
from mlserving.schema import FEATURES, FeatureSpec, RangeChecker

log = logging.getLogger("mlserving")


@dataclass
class ModelBundle:
    """A model version that passed checksum verification, plus its schema."""

    version: str
    stage: str
    metadata: dict[str, Any]
    model: Any
    specs: list[FeatureSpec]
    range_checker: RangeChecker
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))

    def to_matrix(self, rows: Sequence[Mapping[str, float]]) -> NDArray[np.float64]:
        """Rows (dicts) to a float matrix in the training feature order."""
        return np.asarray(
            [[row[name] for name in self.feature_names] for row in rows], dtype=np.float64
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(x), dtype=np.float64)


def load_bundle(model_dir: Path, spec: str, range_margin: float = 0.1) -> ModelBundle:
    """Resolve ``spec`` (production, latest or vN), verify checksums, then unpickle.

    The pickle is only deserialized after its SHA-256 matched ``metadata.json``.
    """
    model_dir = Path(model_dir)
    version = registry.resolve_version(model_dir, spec)
    meta = registry.verify_version(model_dir, version)
    names = list(meta.get("feature_names", []))
    if names != list(FEATURES):
        raise ArtifactError(
            f"feature order mismatch in {version}: metadata has {names}, service expects "
            f"{list(FEATURES)}"
        )
    specs = [FeatureSpec(**f) for f in meta["feature_schema"]]
    if [s.name for s in specs] != names:
        raise ArtifactError(f"feature_schema of {version} does not follow feature_names")
    model = joblib.load(model_dir / version / meta["artifacts"]["model"]["file"])
    trained_with = meta.get("library_versions", {}).get("scikit-learn")
    import sklearn

    if trained_with and trained_with != sklearn.__version__:
        log.warning(
            "scikit-learn version differs from training",
            extra={"fields": {"trained_with": trained_with, "running": sklearn.__version__}},
        )
    stage = registry.load_registry(model_dir)["versions"].get(version, {}).get("stage", "staging")
    return ModelBundle(
        version=version,
        stage=stage,
        metadata=meta,
        model=model,
        specs=specs,
        range_checker=RangeChecker(specs, margin=range_margin),
    )
