"""Request/response schemas and the training-derived input range checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

FEATURES: tuple[str, ...] = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)
TARGET = "MedHouseVal"
TARGET_UNIT = "100000 USD"

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "MedInc": "Median income in the block group (tens of thousands of USD).",
    "HouseAge": "Median house age in the block group (years).",
    "AveRooms": "Average number of rooms per household.",
    "AveBedrms": "Average number of bedrooms per household.",
    "Population": "Block group population.",
    "AveOccup": "Average number of household members.",
    "Latitude": "Block group latitude (degrees).",
    "Longitude": "Block group longitude (degrees).",
}


def _feature(name: str, example: float) -> float:
    return Field(  # type: ignore[no-any-return]
        description=FEATURE_DESCRIPTIONS[name], examples=[example], allow_inf_nan=False
    )


class PredictionRequest(BaseModel):
    """One block group. All eight features are required and must be finite numbers."""

    model_config = ConfigDict(extra="forbid", strict=True)

    MedInc: float = _feature("MedInc", 3.5)
    HouseAge: float = _feature("HouseAge", 28.0)
    AveRooms: float = _feature("AveRooms", 5.4)
    AveBedrms: float = _feature("AveBedrms", 1.1)
    Population: float = _feature("Population", 1400.0)
    AveOccup: float = _feature("AveOccup", 3.0)
    Latitude: float = _feature("Latitude", 34.2)
    Longitude: float = _feature("Longitude", -118.4)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[PredictionRequest] = Field(min_length=1, description="Block groups to score.")


class PredictionResponse(BaseModel):
    prediction: float = Field(description="Predicted median house value, in units of USD 100,000.")
    unit: str = TARGET_UNIT
    model_version: str
    request_id: str


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    count: int
    unit: str = TARGET_UNIT
    model_version: str
    request_id: str


class ValidationIssue(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(BaseModel):
    detail: str | list[ValidationIssue]
    request_id: str | None = None


@dataclass(frozen=True)
class FeatureSpec:
    """Per-feature training statistics stored in ``metadata.json``."""

    name: str
    min: float
    max: float
    mean: float
    std: float
    dtype: str = "float64"
    description: str = ""


@dataclass(frozen=True)
class RangeViolation:
    feature: str
    value: float
    low: float
    high: float
    message: str


class RangeChecker:
    """Rejects inputs far outside what the model saw in training.

    The allowed interval is the training [min, max] widened by ``margin`` times the
    training range on each side. This is a guard against garbage and unit mistakes,
    not a guarantee that predictions near the edges are accurate.
    """

    def __init__(self, specs: Sequence[FeatureSpec], margin: float = 0.1) -> None:
        self._specs = {s.name: s for s in specs}
        self._bounds: dict[str, tuple[float, float]] = {}
        for s in specs:
            pad = margin * (s.max - s.min)
            self._bounds[s.name] = (s.min - pad, s.max + pad)

    def bounds(self, name: str) -> tuple[float, float]:
        return self._bounds[name]

    def violations(self, row: Mapping[str, float]) -> list[RangeViolation]:
        found: list[RangeViolation] = []
        for name, spec in self._specs.items():
            value = row[name]
            low, high = self._bounds[name]
            if not low <= value <= high:
                found.append(
                    RangeViolation(
                        feature=name,
                        value=value,
                        low=low,
                        high=high,
                        message=(
                            f"Value {value:g} is outside the training range "
                            f"[{spec.min:g}, {spec.max:g}] (accepted: [{low:g}, {high:g}])."
                        ),
                    )
                )
        return found
