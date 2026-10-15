"""Versioned model artifacts and a tiny JSON registry with a promotion step.

Layout::

    models/
      registry.json          {"production": "v1", "versions": {...}, "history": [...]}
      v1/model.joblib        serialized model
      v1/metadata.json       schema, metrics, git commit, date, SHA-256 of the artifacts
      v1/reference.json      training quantile sketch used by the drift module
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRY_FILE = "registry.json"
VERSION_RE = re.compile(r"^v[0-9]+$")
STAGES = ("staging", "production", "archived")


class RegistryError(Exception):
    """A version or registry operation is invalid."""


class ArtifactError(Exception):
    """An artifact folder is missing files or has inconsistent contents."""


class ChecksumMismatchError(ArtifactError):
    """An artifact's SHA-256 differs from the value recorded in ``metadata.json``."""


@dataclass(frozen=True)
class VersionInfo:
    version: str
    stage: str
    created_at: str | None
    model_sha256: str | None


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def version_key(version: str) -> int:
    return int(version[1:])


def check_version_name(version: str) -> str:
    if not VERSION_RE.fullmatch(version):
        raise RegistryError(f"invalid version name {version!r}; expected v<number>, e.g. v1")
    return version


def _empty() -> dict[str, Any]:
    return {"schema_version": 1, "production": None, "versions": {}, "history": []}


def load_registry(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / REGISTRY_FILE
    if not path.exists():
        return _empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "versions" not in data:
        raise RegistryError(f"{path} is not a registry file")
    return data


def save_registry(model_dir: Path, registry: dict[str, Any]) -> None:
    """Write atomically: a crash never leaves a half-written registry."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = model_dir / f"{REGISTRY_FILE}.tmp"
    tmp.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, model_dir / REGISTRY_FILE)


def register_version(model_dir: Path, version: str, metadata: dict[str, Any]) -> None:
    """Add a freshly trained version as ``staging``."""
    reg = load_registry(model_dir)
    reg["versions"][check_version_name(version)] = {
        "stage": "staging",
        "created_at": metadata.get("trained_at"),
        "model_sha256": metadata["artifacts"]["model"]["sha256"],
    }
    reg["history"].append({"action": "register", "version": version, "at": utc_now()})
    save_registry(model_dir, reg)


def list_versions(model_dir: Path) -> list[VersionInfo]:
    reg = load_registry(model_dir)
    out = [
        VersionInfo(
            version=v,
            stage=str(info.get("stage", "staging")),
            created_at=info.get("created_at"),
            model_sha256=info.get("model_sha256"),
        )
        for v, info in reg["versions"].items()
    ]
    return sorted(out, key=lambda i: version_key(i.version))


def resolve_version(model_dir: Path, spec: str) -> str:
    """Map ``production``, ``latest`` or ``vN`` to a concrete registered version."""
    reg = load_registry(model_dir)
    if spec == "production":
        prod = reg.get("production")
        if not prod:
            raise RegistryError("no version has been promoted to production; run `promote`")
        return str(prod)
    if spec == "latest":
        if not reg["versions"]:
            raise RegistryError("the registry has no versions")
        return max(reg["versions"], key=version_key)  # type: ignore[no-any-return]
    check_version_name(spec)
    if spec not in reg["versions"]:
        known = ", ".join(sorted(reg["versions"], key=version_key)) or "none"
        raise RegistryError(f"version {spec} is not registered (known: {known})")
    return spec


def verify_version(model_dir: Path, version: str) -> dict[str, Any]:
    """Check every artifact against its recorded SHA-256 and return the metadata."""
    check_version_name(version)
    folder = Path(model_dir) / version
    meta_path = folder / "metadata.json"
    if not meta_path.is_file():
        raise ArtifactError(f"{meta_path} is missing")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        artifacts = meta["artifacts"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ArtifactError(f"{meta_path} is unreadable or incomplete: {exc!r}") from exc
    for name, info in artifacts.items():
        file = folder / info["file"]
        if not file.is_file():
            raise ArtifactError(f"artifact '{name}' ({file.name}) is missing from {folder.name}")
        actual = sha256_file(file)
        if actual != info["sha256"]:
            raise ChecksumMismatchError(
                f"checksum mismatch for {version}/{file.name}: "
                f"expected {info['sha256'][:12]}..., found {actual[:12]}..."
            )
    return meta  # type: ignore[no-any-return]


def promote(model_dir: Path, version: str) -> dict[str, Any]:
    """Mark ``version`` as production after verifying its artifacts.

    The previous production version becomes ``archived``. Promoting the version that is
    already in production is a no-op apart from a history entry.
    """
    version = resolve_version(model_dir, version)
    verify_version(model_dir, version)
    reg = load_registry(model_dir)
    previous = reg.get("production")
    if previous and previous != version and previous in reg["versions"]:
        reg["versions"][previous]["stage"] = "archived"
    reg["versions"][version]["stage"] = "production"
    reg["production"] = version
    reg["history"].append(
        {"action": "promote", "version": version, "previous": previous, "at": utc_now()}
    )
    save_registry(model_dir, reg)
    return reg
