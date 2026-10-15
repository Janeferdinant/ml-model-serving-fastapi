import json
import math

import joblib
import numpy as np
from fastapi.testclient import TestClient

from mlserving.schema import FEATURES


def test_predict_success(client: TestClient, valid_payload: dict[str, float]) -> None:
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == "v1"
    assert body["unit"] == "100000 USD"
    assert body["request_id"] == r.headers["x-request-id"]


def test_predict_matches_the_model_directly(
    client: TestClient, valid_payload: dict[str, float], model_dir: object
) -> None:
    model = joblib.load(f"{model_dir}/v1/model.joblib")
    expected = float(model.predict(np.array([[valid_payload[f] for f in FEATURES]]))[0])
    got = client.post("/predict", json=valid_payload).json()["prediction"]
    assert got == expected


def test_batch_success_and_order(client: TestClient, valid_payload: dict[str, float]) -> None:
    rows = [{**valid_payload, "MedInc": m} for m in (1.0, 3.0, 6.0)]
    r = client.post("/predict/batch", json={"instances": rows})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    singles = [client.post("/predict", json=row).json()["prediction"] for row in rows]
    assert body["predictions"] == singles
    assert body["predictions"][0] < body["predictions"][2]


def test_batch_limit(client: TestClient, valid_payload: dict[str, float]) -> None:
    from mlserving.app import create_app
    from mlserving.config import Settings

    s = Settings(model_dir=client.app.state.settings.model_dir, max_batch_size=2)  # type: ignore[attr-defined]
    with TestClient(create_app(s)) as small:
        r = small.post("/predict/batch", json={"instances": [valid_payload] * 3})
    assert r.status_code == 422
    assert "at most 2" in json.dumps(r.json())


def test_batch_empty_is_422(client: TestClient) -> None:
    r = client.post("/predict/batch", json={"instances": []})
    assert r.status_code == 422


def test_missing_field_is_422_with_location(
    client: TestClient, valid_payload: dict[str, float]
) -> None:
    del valid_payload["Latitude"]
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["loc"] == ["body", "Latitude"]
    assert "required" in err["msg"].lower()
    assert r.json()["request_id"]


def test_wrong_type_is_422(client: TestClient, valid_payload: dict[str, object]) -> None:
    valid_payload["MedInc"] = "high"
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "MedInc"]


def test_extra_field_is_422(client: TestClient, valid_payload: dict[str, float]) -> None:
    r = client.post("/predict", json={**valid_payload, "Zip": 90210})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "Zip"]


def test_nan_is_422(client: TestClient, valid_payload: dict[str, float]) -> None:
    raw = json.dumps(valid_payload).replace("3.5", "NaN")
    r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_out_of_range_is_422_and_explains(
    client: TestClient, valid_payload: dict[str, float]
) -> None:
    valid_payload["MedInc"] = 1e6
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["loc"] == ["body", "MedInc"]
    assert err["type"] == "value_error.out_of_range"
    assert "training range" in err["msg"]


def test_batch_out_of_range_reports_row_index(
    client: TestClient, valid_payload: dict[str, float]
) -> None:
    bad = {**valid_payload, "HouseAge": 500.0}
    r = client.post("/predict/batch", json={"instances": [valid_payload, bad]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "instances", 1, "HouseAge"]


def test_malformed_json_is_400(client: TestClient) -> None:
    r = client.post("/predict", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert "malformed" in r.json()["detail"].lower()


def test_empty_body_is_422(client: TestClient) -> None:
    r = client.post("/predict", content=b"", headers={"content-type": "application/json"})
    assert r.status_code in (400, 422)


def test_non_object_body_is_422(client: TestClient) -> None:
    r = client.post("/predict", json=[1, 2, 3])
    assert r.status_code == 422


def test_binary_garbage_is_handled(client: TestClient) -> None:
    r = client.post(
        "/predict", content=b"\xff\xfe\x00\x01", headers={"content-type": "application/json"}
    )
    assert r.status_code in (400, 422)


def test_unknown_route_and_method(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404
    assert client.get("/predict").status_code == 405


def test_predict_not_available_when_model_missing(
    model_dir_copy: object, valid_payload: dict[str, float]
) -> None:
    import shutil
    from pathlib import Path

    from mlserving.app import create_app
    from mlserving.config import Settings

    shutil.rmtree(Path(str(model_dir_copy)) / "v1")
    s = Settings(model_dir=Path(str(model_dir_copy)), strict_startup=False)
    with TestClient(create_app(s)) as c:
        r = c.post("/predict", json=valid_payload)
        assert r.status_code == 503
        assert c.post("/predict/batch", json={"instances": [valid_payload]}).status_code == 503


def test_unhandled_error_returns_json_500(
    settings: object, valid_payload: dict[str, float]
) -> None:
    from mlserving.app import create_app

    app = create_app(settings)  # type: ignore[arg-type]
    with TestClient(app, raise_server_exceptions=False) as c:
        c.app.state.bundle.predict = None  # type: ignore[attr-defined]
        r = c.post("/predict", json=valid_payload)
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error"
    assert r.json()["request_id"]
