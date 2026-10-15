import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlserving.schema import (
    FEATURES,
    BatchPredictionRequest,
    FeatureSpec,
    PredictionRequest,
    RangeChecker,
)

from .conftest import VALID_PAYLOAD


def specs_from(model_dir: Path) -> list[FeatureSpec]:
    meta = json.loads((model_dir / "v1" / "metadata.json").read_text())
    return [FeatureSpec(**f) for f in meta["feature_schema"]]


def test_features_constant() -> None:
    assert len(FEATURES) == 8
    assert FEATURES[0] == "MedInc"


def test_request_accepts_valid_payload() -> None:
    assert PredictionRequest(**VALID_PAYLOAD).MedInc == 3.5


def test_request_accepts_ints_for_float_fields() -> None:
    payload = {**VALID_PAYLOAD, "HouseAge": 28}
    assert PredictionRequest(**payload).HouseAge == 28.0


@pytest.mark.parametrize("bad", ["3.5", None, True, [1], {"a": 1}])
def test_request_rejects_non_numbers(bad: object) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**VALID_PAYLOAD, "MedInc": bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_request_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**VALID_PAYLOAD, "MedInc": bad})


def test_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**VALID_PAYLOAD, Extra=1.0)  # type: ignore[call-arg]


def test_batch_requires_at_least_one_instance() -> None:
    with pytest.raises(ValidationError):
        BatchPredictionRequest(instances=[])


def test_range_checker_accepts_in_range(model_dir: Path) -> None:
    checker = RangeChecker(specs_from(model_dir), margin=0.1)
    assert checker.violations(VALID_PAYLOAD) == []


def test_range_checker_flags_out_of_range(model_dir: Path) -> None:
    checker = RangeChecker(specs_from(model_dir), margin=0.1)
    bad = {**VALID_PAYLOAD, "MedInc": 1e6, "Latitude": -90.0}
    names = [v.feature for v in checker.violations(bad)]
    assert names == ["MedInc", "Latitude"]
    assert "training range" in checker.violations(bad)[0].message


def test_margin_widens_bounds(model_dir: Path) -> None:
    specs = specs_from(model_dir)
    strict = RangeChecker(specs, margin=0.0)
    lenient = RangeChecker(specs, margin=0.5)
    lat = next(s for s in specs if s.name == "Latitude")
    value = lat.max + 0.2 * (lat.max - lat.min)
    row = {**VALID_PAYLOAD, "Latitude": value}
    assert [v.feature for v in strict.violations(row)] == ["Latitude"]
    assert lenient.violations(row) == []


def test_bounds_are_reported(model_dir: Path) -> None:
    checker = RangeChecker(specs_from(model_dir), margin=0.1)
    lo, hi = checker.bounds("Latitude")
    assert lo < hi
