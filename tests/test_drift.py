import json
from pathlib import Path

import numpy as np
import pytest

from mlserving import drift
from mlserving.drift import ReferenceProfile

NAMES = ["a", "b", "c"]


def rng_matrix(seed: int, n: int = 5000, shift: float = 0.0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (n, 3))
    base[:, 0] = base[:, 0] * scale + shift
    return base


@pytest.fixture
def profile() -> ReferenceProfile:
    return ReferenceProfile.from_matrix(rng_matrix(0, 20000), NAMES)


# ------------------------------------------------------------------------- PSI / KS


def test_psi_near_zero_for_same_distribution() -> None:
    a = np.random.default_rng(1).normal(size=20000)
    b = np.random.default_rng(2).normal(size=20000)
    assert drift.psi(a, b) < 0.01


def test_psi_grows_with_shift() -> None:
    a = np.random.default_rng(1).normal(size=20000)
    small = drift.psi(a, np.random.default_rng(2).normal(0.2, 1, 20000))
    large = drift.psi(a, np.random.default_rng(2).normal(1.5, 1, 20000))
    assert 0.0 < small < 0.25 < large


def test_psi_is_non_negative_and_zero_for_identical() -> None:
    a = np.random.default_rng(1).normal(size=1000)
    assert drift.psi(a, a) == pytest.approx(0.0, abs=1e-9)


def test_psi_handles_values_outside_reference_range() -> None:
    a = np.random.default_rng(1).normal(size=5000)
    assert drift.psi(a, a + 100.0) > 1.0


def test_psi_with_constant_reference_feature() -> None:
    ref = np.full(1000, 5.0)
    assert drift.psi(ref, np.full(500, 5.0)) == pytest.approx(0.0)
    assert drift.psi(ref, np.full(500, 6.0)) > 0.25


def test_psi_with_discrete_feature_has_unique_edges() -> None:
    ref = np.random.default_rng(1).integers(0, 3, 5000).astype(float)
    cur = np.random.default_rng(2).integers(0, 3, 5000).astype(float)
    assert drift.psi(ref, cur) < 0.01


def test_ks_detects_shift_and_not_null() -> None:
    a = np.random.default_rng(1).normal(size=5000)
    null_stat, null_p = drift.ks(a, np.random.default_rng(2).normal(size=5000))
    shift_stat, shift_p = drift.ks(a, np.random.default_rng(3).normal(0.5, 1, 5000))
    assert null_p > 0.01
    assert shift_p < 1e-10
    assert shift_stat > null_stat


# ------------------------------------------------------------------------- profile


def test_profile_roundtrip_and_pseudo_sample(profile: ReferenceProfile) -> None:
    again = ReferenceProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
    assert again.feature_names == NAMES
    assert again.n == 20000
    sample = again.pseudo_sample("a")
    assert len(sample) == 1001
    assert sample[0] <= np.median(sample) <= sample[-1]
    assert abs(float(np.median(sample))) < 0.05


def test_profile_from_matrix_validates_shape() -> None:
    with pytest.raises(ValueError, match="columns"):
        ReferenceProfile.from_matrix(np.zeros((10, 2)), NAMES)
    with pytest.raises(ValueError, match="empty"):
        ReferenceProfile.from_matrix(np.zeros((0, 3)), NAMES)


# ------------------------------------------------------------------------- report


def test_no_drift_window_is_ok(profile: ReferenceProfile) -> None:
    report = drift.compute_drift(profile, rng_matrix(11, 2000))
    assert report["overall_status"] == "ok"
    assert all(f["status"] in ("ok", "warning") for f in report["features"])
    assert not any(f["status"] == "drift" for f in report["features"])
    assert report["n_current"] == 2000


def test_simulated_drift_is_detected_per_feature(profile: ReferenceProfile) -> None:
    cur = rng_matrix(12, 2000, shift=1.5)  # feature "a" shifted by 1.5 sigma
    report = drift.compute_drift(profile, cur)
    by_name = {f["name"]: f for f in report["features"]}
    assert report["overall_status"] == "drift"
    assert by_name["a"]["status"] == "drift"
    assert by_name["a"]["psi"] > 0.25
    assert by_name["a"]["ks_pvalue"] < 1e-6
    assert by_name["b"]["status"] != "drift"
    assert by_name["c"]["status"] != "drift"
    assert report["drifted_features"] == ["a"]


def test_variance_drift_is_detected(profile: ReferenceProfile) -> None:
    report = drift.compute_drift(profile, rng_matrix(13, 3000, scale=2.5))
    assert report["drifted_features"] == ["a"]


def test_report_contents(profile: ReferenceProfile) -> None:
    report = drift.compute_drift(profile, rng_matrix(14, 500, shift=0.5), model_version="v1")
    assert report["model_version"] == "v1"
    assert report["thresholds"]["psi_drift"] == 0.25
    feat = report["features"][0]
    for key in ("name", "psi", "ks_statistic", "ks_pvalue", "status", "reference", "current"):
        assert key in feat
    hist = feat["histogram"]
    assert len(hist["edges"]) == len(hist["reference"]) + 1 == len(hist["current"]) + 1
    assert sum(hist["reference"]) == pytest.approx(1.0)
    assert sum(hist["current"]) == pytest.approx(1.0)
    json.dumps(report)  # must be JSON-serialisable


def test_too_few_samples_is_an_error(profile: ReferenceProfile) -> None:
    with pytest.raises(ValueError, match="at least"):
        drift.compute_drift(profile, rng_matrix(1, 10))


def test_wrong_column_count_is_an_error(profile: ReferenceProfile) -> None:
    with pytest.raises(ValueError, match="columns"):
        drift.compute_drift(profile, np.zeros((100, 2)))


def test_nan_rows_are_dropped(profile: ReferenceProfile) -> None:
    cur = rng_matrix(15, 500)
    cur[0, 1] = np.nan
    report = drift.compute_drift(profile, cur)
    assert report["n_current"] == 499


def test_html_report_lists_features_and_status(profile: ReferenceProfile) -> None:
    report = drift.compute_drift(profile, rng_matrix(16, 1000, shift=1.5))
    html = drift.render_html(report)
    assert html.startswith("<!doctype html>")
    for name in NAMES:
        assert name in html
    assert "drift" in html.lower()
    assert "<script" not in html.lower()


def test_write_report_creates_both_files(profile: ReferenceProfile, tmp_path: Path) -> None:
    report = drift.compute_drift(profile, rng_matrix(17, 500))
    json_path, html_path = drift.write_report(report, tmp_path / "out")
    assert json.loads(json_path.read_text())["overall_status"] == report["overall_status"]
    assert html_path.read_text().startswith("<!doctype html>")


# ------------------------------------------------------------------------- request log


def write_log(path: Path, matrix: np.ndarray, start_minute: int = 0) -> None:
    with path.open("w") as fh:
        for i, row in enumerate(matrix):
            rec = {
                "ts": f"2026-10-15T10:{start_minute + i // 60:02d}:{i % 60:02d}Z",
                "request_id": f"r{i}",
                "model_version": "v1",
                "features": dict(zip(NAMES, map(float, row), strict=True)),
                "prediction": 1.0,
            }
            fh.write(json.dumps(rec) + "\n")


def test_load_request_log_window_and_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    write_log(path, rng_matrix(1, 120))
    with path.open("a") as fh:
        fh.write("not json\n")
        fh.write(json.dumps({"ts": "x", "features": {"a": 1}}) + "\n")  # missing features
        fh.write("\n")
    win = drift.load_request_log(path, NAMES)
    assert win.x.shape == (120, 3)
    assert win.n_skipped == 2
    assert win.first_ts == "2026-10-15T10:00:00Z"
    assert win.last_ts == "2026-10-15T10:01:59Z"

    last = drift.load_request_log(path, NAMES, last=50)
    assert last.x.shape == (50, 3)
    np.testing.assert_array_equal(last.x, win.x[-50:])

    since = drift.load_request_log(path, NAMES, since="2026-10-15T10:01:00Z")
    assert since.x.shape == (60, 3)
    until = drift.load_request_log(path, NAMES, until="2026-10-15T10:00:59Z")
    assert until.x.shape == (60, 3)


def test_load_request_log_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        drift.load_request_log(tmp_path / "none.jsonl", NAMES)
