import json
from pathlib import Path

import pytest

from mlserving import cli
from mlserving.cli import main

from .conftest import synthetic_dataset


@pytest.fixture(autouse=True)
def _offline_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_california", lambda _dir: synthetic_dataset())


def test_train_then_promote_then_versions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "models"
    train_args = [
        "train",
        "--out",
        str(out),
        "--no-mlflow",
        "--max-iter",
        "20",
        "--bootstrap",
        "20",
    ]
    assert main(train_args) == 0
    assert main(train_args) == 0
    assert (out / "v1" / "metadata.json").exists()
    assert (out / "v2" / "metadata.json").exists()
    assert main(["promote", "v2", "--model-dir", str(out)]) == 0
    capsys.readouterr()
    assert main(["versions", "--model-dir", str(out), "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    stages = {row["version"]: row["stage"] for row in listing}
    assert stages == {"v1": "staging", "v2": "production"}


def test_train_with_promote_flag(tmp_path: Path) -> None:
    out = tmp_path / "models"
    code = main(
        ["train", "--out", str(out), "--no-mlflow", "--max-iter", "10", "--bootstrap", "10",
         "--promote"]
    )  # fmt: skip
    assert code == 0
    assert json.loads((out / "registry.json").read_text())["production"] == "v1"


def test_promote_unknown_version_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["promote", "v9", "--model-dir", str(tmp_path)]) == 1
    assert "v9" in capsys.readouterr().err


def test_verify_ok_and_tampered(model_dir_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "v1", "--model-dir", str(model_dir_copy)]) == 0
    assert "OK" in capsys.readouterr().out
    with (model_dir_copy / "v1" / "model.joblib").open("ab") as fh:
        fh.write(b"x")
    assert main(["verify", "v1", "--model-dir", str(model_dir_copy)]) == 1
    assert "checksum" in capsys.readouterr().err.lower()


def test_versions_table(model_dir_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["versions", "--model-dir", str(model_dir_copy)]) == 0
    out = capsys.readouterr().out
    assert "v1" in out
    assert "production" in out


def test_serve_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_run(app: str, **kwargs: object) -> None:
        calls["app"] = app
        calls.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert main(["serve", "--port", "9001", "--workers", "3"]) == 0
    assert calls["app"] == "mlserving.app:create_app"
    assert calls["factory"] is True
    assert calls["port"] == 9001
    assert calls["workers"] == 3


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()
