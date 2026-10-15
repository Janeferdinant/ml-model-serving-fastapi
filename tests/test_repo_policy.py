"""Guards for the delivery pipeline: the image must never be published by these workflows."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_dockerfile_is_multistage_nonroot_pinned_with_healthcheck() -> None:
    text = read("Dockerfile")
    froms = re.findall(r"^FROM\s+(\S+)", text, flags=re.MULTILINE)
    assert len(froms) >= 2, "multi-stage build expected"
    assert all("@sha256:" in f for f in froms), "base image must be pinned by digest"
    assert "HEALTHCHECK" in text
    user = re.findall(r"^USER\s+(\S+)", text, flags=re.MULTILINE)
    assert user, "a USER instruction is required"
    assert user[-1] not in {"root", "0"}


def test_release_workflow_builds_but_never_publishes() -> None:
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert re.search(r"tags:\s*\n\s*-\s*['\"]v", text), "release must trigger on version tags"
    assert "branches:" not in text
    assert "workflow_dispatch" not in text
    forbidden = [
        r"docker\s+push",
        r"push:\s*true",
        r"docker/login-action",
        r"docker\s+login",
        r"gh\s+release",
        r"softprops/action-gh-release",
        r"ghcr\.io",
        r"packages:\s*write",
        r"contents:\s*write",
        r"id-token:\s*write",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, text), f"release.yml must not contain {pattern!r}"
    assert "docker build" in text


def test_ci_never_pushes_images() -> None:
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert not re.search(r"docker\s+push|push:\s*true|docker/login-action", text)
    for needed in ("ruff", "mypy", "pytest", "hadolint", "pip-audit", "smoke"):
        assert needed in text, f"ci.yml should mention {needed}"


def test_no_secrets_or_local_paths_in_tracked_text() -> None:
    suspicious = [re.compile(r"[A-Za-z]:\\Users\\"), re.compile(r"ghp_[A-Za-z0-9]{20,}")]
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or any(part in {".git", ".venv", "mlruns", ".mypy_cache", ".ruff_cache"} for part in path.parts)
            or path.suffix not in {".py", ".md", ".yml", ".yaml", ".json", ".toml", ".html", ".txt", ""}
            or path.name == "test_repo_policy.py"
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
            or path.parts[-2:-1] == ("scratch",)
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pat in suspicious:
            assert not pat.search(text), f"{path.relative_to(ROOT)} matches {pat.pattern}"


def test_license_and_notices_present() -> None:
    assert "Attribution-NonCommercial-NoDerivatives 4.0" in read("LICENSE")
    notices = read("THIRD_PARTY_NOTICES.md")
    assert "California Housing" in notices
    assert "scikit-learn" in notices
