import json
import shutil
from pathlib import Path

import pytest

from mlserving import registry
from mlserving.registry import ArtifactError, ChecksumMismatchError, RegistryError


def test_lists_versions_with_stage(model_dir_copy: Path) -> None:
    versions = {v.version: v for v in registry.list_versions(model_dir_copy)}
    assert set(versions) == {"v1", "v2"}
    assert versions["v1"].stage == "production"
    assert versions["v2"].stage == "staging"


def test_resolve_specs(model_dir_copy: Path) -> None:
    assert registry.resolve_version(model_dir_copy, "production") == "v1"
    assert registry.resolve_version(model_dir_copy, "latest") == "v2"
    assert registry.resolve_version(model_dir_copy, "v2") == "v2"


def test_resolve_unknown_version(model_dir_copy: Path) -> None:
    with pytest.raises(RegistryError):
        registry.resolve_version(model_dir_copy, "v99")


@pytest.mark.parametrize("bad", ["../v1", "v1/../v2", "V1", "1", "v", "v1 ", ""])
def test_resolve_rejects_malformed_names(model_dir_copy: Path, bad: str) -> None:
    with pytest.raises(RegistryError):
        registry.resolve_version(model_dir_copy, bad)


def test_latest_is_numeric_not_lexicographic(model_dir_copy: Path) -> None:
    shutil.copytree(model_dir_copy / "v2", model_dir_copy / "v10")
    reg = json.loads((model_dir_copy / "registry.json").read_text())
    reg["versions"]["v10"] = {"stage": "staging"}
    (model_dir_copy / "registry.json").write_text(json.dumps(reg))
    assert registry.resolve_version(model_dir_copy, "latest") == "v10"


def test_promote_switches_production_and_archives_previous(model_dir_copy: Path) -> None:
    reg = registry.promote(model_dir_copy, "v2")
    assert reg["production"] == "v2"
    versions = {v.version: v.stage for v in registry.list_versions(model_dir_copy)}
    assert versions == {"v1": "archived", "v2": "production"}
    assert registry.resolve_version(model_dir_copy, "production") == "v2"
    last = reg["history"][-1]
    assert last["action"] == "promote"
    assert last["version"] == "v2"
    assert last["previous"] == "v1"
    assert "at" in last


def test_promote_is_idempotent(model_dir_copy: Path) -> None:
    registry.promote(model_dir_copy, "v1")
    reg = registry.load_registry(model_dir_copy)
    assert reg["production"] == "v1"


def test_promote_unknown_version(model_dir_copy: Path) -> None:
    with pytest.raises(RegistryError):
        registry.promote(model_dir_copy, "v7")


def test_promote_refuses_corrupted_artifact(model_dir_copy: Path) -> None:
    with (model_dir_copy / "v2" / "model.joblib").open("ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(ChecksumMismatchError):
        registry.promote(model_dir_copy, "v2")
    assert registry.resolve_version(model_dir_copy, "production") == "v1"


def test_production_without_promotion(tmp_path: Path) -> None:
    (tmp_path / "registry.json").write_text(
        json.dumps({"schema_version": 1, "production": None, "versions": {}, "history": []})
    )
    with pytest.raises(RegistryError, match="production"):
        registry.resolve_version(tmp_path, "production")


def test_registry_write_is_atomic_and_valid_json(model_dir_copy: Path) -> None:
    registry.promote(model_dir_copy, "v2")
    assert not list(model_dir_copy.glob("*.tmp"))
    json.loads((model_dir_copy / "registry.json").read_text())


def test_verify_version_returns_metadata(model_dir_copy: Path) -> None:
    meta = registry.verify_version(model_dir_copy, "v1")
    assert meta["version"] == "v1"
    assert len(meta["artifacts"]["model"]["sha256"]) == 64


def test_verify_detects_changed_model(model_dir_copy: Path) -> None:
    path = model_dir_copy / "v1" / "model.joblib"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ChecksumMismatchError):
        registry.verify_version(model_dir_copy, "v1")


def test_verify_detects_missing_files(model_dir_copy: Path) -> None:
    (model_dir_copy / "v1" / "model.joblib").unlink()
    with pytest.raises(ArtifactError):
        registry.verify_version(model_dir_copy, "v1")
    (model_dir_copy / "v2" / "metadata.json").unlink()
    with pytest.raises(ArtifactError):
        registry.verify_version(model_dir_copy, "v2")


def test_verify_detects_corrupt_metadata(model_dir_copy: Path) -> None:
    (model_dir_copy / "v1" / "metadata.json").write_text("{not json")
    with pytest.raises(ArtifactError):
        registry.verify_version(model_dir_copy, "v1")


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert (
        registry.sha256_file(p)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
