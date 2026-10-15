import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlserving.app import create_app
from mlserving.config import Settings
from mlserving.registry import ArtifactError, RegistryError


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_when_model_verified(client: TestClient) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["model_version"] == "v1"
    assert body["checksum_verified"] is True


def test_model_info(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "v1"
    assert body["stage"] == "production"
    assert body["model_type"] == "HistGradientBoostingRegressor"
    assert [f["name"] for f in body["feature_schema"]][0] == "MedInc"
    assert len(body["artifacts"]["model"]["sha256"]) == 64
    assert body["metrics"]["test"]["rmse"] > 0
    assert body["trained_at"].endswith("Z")
    assert "commit" in body["git"]
    assert body["range_margin"] == 0.1


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("production", "v1"), ("v2", "v2"), ("latest", "v2"), ("v1", "v1")],
)
def test_loads_requested_version(model_dir: Path, spec: str, expected: str) -> None:
    app = create_app(Settings(model_dir=model_dir, model_version=spec))
    with TestClient(app) as c:
        assert c.get("/model-info").json()["version"] == expected


def test_model_version_follows_promotion(model_dir_copy: Path) -> None:
    from mlserving.registry import promote

    promote(model_dir_copy, "v2")
    with TestClient(create_app(Settings(model_dir=model_dir_copy))) as c:
        body = c.get("/model-info").json()
    assert body["version"] == "v2"
    assert body["stage"] == "production"


def test_unknown_version_refuses_to_start(model_dir: Path) -> None:
    with (
        pytest.raises(RegistryError),
        TestClient(create_app(Settings(model_dir=model_dir, model_version="v42"))),
    ):
        pass


def corrupt(model_dir: Path) -> None:
    path = model_dir / "v1" / "model.joblib"
    data = bytearray(path.read_bytes())
    data[len(data) // 3] ^= 0x55
    path.write_bytes(bytes(data))


def test_checksum_mismatch_refuses_to_start(model_dir_copy: Path) -> None:
    corrupt(model_dir_copy)
    with (
        pytest.raises(ArtifactError, match="checksum"),
        TestClient(create_app(Settings(model_dir=model_dir_copy))),
    ):
        pass


def test_checksum_mismatch_is_not_ready_when_not_strict(
    model_dir_copy: Path, valid_payload: dict[str, float]
) -> None:
    corrupt(model_dir_copy)
    settings = Settings(model_dir=model_dir_copy, strict_startup=False)
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").status_code == 200
        ready = c.get("/ready")
        assert ready.status_code == 503
        assert "checksum" in ready.json()["reason"].lower()
        assert c.post("/predict", json=valid_payload).status_code == 503
        assert c.get("/model-info").status_code == 503
        assert "model_ready 0.0" in c.get("/metrics").text


def test_feature_order_mismatch_is_refused(model_dir_copy: Path) -> None:
    import json

    from mlserving.registry import sha256_file

    meta_path = model_dir_copy / "v1" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["feature_names"] = list(reversed(meta["feature_names"]))
    meta_path.write_text(json.dumps(meta))
    assert sha256_file(model_dir_copy / "v1" / "model.joblib")  # checksum itself still fine
    with (
        pytest.raises(ArtifactError, match="feature"),
        TestClient(create_app(Settings(model_dir=model_dir_copy))),
    ):
        pass


def test_metrics_text_format(client: TestClient, valid_payload: dict[str, float]) -> None:
    for _ in range(3):
        assert client.post("/predict", json=valid_payload).status_code == 200
    client.post("/predict", json={})  # a 422
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert "# TYPE http_requests_total counter" in text
    assert 'http_requests_total{method="POST",path="/predict",status="200"} 3.0' in text
    assert 'http_requests_total{method="POST",path="/predict",status="422"} 1.0' in text
    assert "# TYPE http_request_duration_seconds histogram" in text
    assert re.search(
        r'http_request_duration_seconds_bucket\{le="0\.005",method="POST",path="/predict"\} \d',
        text,
    )
    assert 'http_request_duration_seconds_count{method="POST",path="/predict"} 4.0' in text
    assert "# TYPE model_prediction_value histogram" in text
    assert 'model_prediction_value_count{model_version="v1"} 3.0' in text
    assert 'model_predictions_total{model_version="v1"} 3.0' in text
    assert "model_ready 1.0" in text
    assert 'model_info{sha256=' in text


def test_metrics_do_not_explode_label_cardinality(client: TestClient) -> None:
    for i in range(5):
        client.get(f"/random/{i}")
    text = client.get("/metrics").text
    assert 'path="/random/' not in text
    assert 'path="unmatched"' in text


def test_metrics_include_predict_batch_counts(
    client: TestClient, valid_payload: dict[str, float]
) -> None:
    client.post("/predict/batch", json={"instances": [valid_payload] * 4})
    text = client.get("/metrics").text
    assert 'model_predictions_total{model_version="v1"} 4.0' in text


def test_openapi_documents_endpoints(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    for path in ("/predict", "/predict/batch", "/health", "/ready", "/model-info", "/metrics"):
        assert path in spec["paths"]
    assert spec["info"]["title"]
