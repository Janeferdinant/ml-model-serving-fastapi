import io
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlserving.app import create_app
from mlserving.config import Settings
from mlserving.middleware import TokenBucketLimiter


def make_client(model_dir: Path, **kwargs: object) -> TestClient:
    return TestClient(create_app(Settings(model_dir=model_dir, **kwargs)))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- auth


def test_auth_disabled_by_default(client: TestClient, valid_payload: dict[str, float]) -> None:
    assert client.post("/predict", json=valid_payload).status_code == 200


def test_auth_required_when_key_configured(
    model_dir: Path, valid_payload: dict[str, float]
) -> None:
    with make_client(model_dir, api_key="s3cret", rate_limit_per_minute=0) as c:
        assert c.post("/predict", json=valid_payload).status_code == 401
        assert c.post("/predict/batch", json={"instances": [valid_payload]}).status_code == 401
        assert c.get("/model-info").status_code == 401
        bad = c.post("/predict", json=valid_payload, headers={"X-API-Key": "nope"})
        assert bad.status_code == 401
        assert "api key" in bad.json()["detail"].lower()
        ok = c.post("/predict", json=valid_payload, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
        assert c.get("/model-info", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_probes_and_metrics_stay_open_with_auth(model_dir: Path) -> None:
    with make_client(model_dir, api_key="s3cret", rate_limit_per_minute=0) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 200
        assert c.get("/metrics").status_code == 200


def test_auth_runs_before_validation(model_dir: Path) -> None:
    with make_client(model_dir, api_key="s3cret", rate_limit_per_minute=0) as c:
        assert c.post("/predict", json={}).status_code == 401


def test_api_key_is_never_logged_or_echoed(model_dir: Path) -> None:
    buf = io.StringIO()
    app = create_app(
        Settings(model_dir=model_dir, api_key="s3cret-value", rate_limit_per_minute=0),
        log_stream=buf,
    )
    with TestClient(app) as c:
        r = c.post("/predict", json={}, headers={"X-API-Key": "s3cret-value-wrong"})
    assert "s3cret" not in buf.getvalue()
    assert "s3cret" not in r.text


# ----------------------------------------------------------------------------- rate limit


def test_rate_limit_returns_429(model_dir: Path, valid_payload: dict[str, float]) -> None:
    with make_client(model_dir, rate_limit_per_minute=5) as c:
        codes = [c.post("/predict", json=valid_payload).status_code for _ in range(8)]
        assert codes[:5] == [200] * 5
        assert codes[5:] == [429] * 3
        r = c.post("/predict", json=valid_payload)
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1
        assert r.json()["detail"].lower().startswith("rate limit")
        assert r.json()["request_id"]


def test_rate_limit_exempts_probes_and_metrics(
    model_dir: Path, valid_payload: dict[str, float]
) -> None:
    with make_client(model_dir, rate_limit_per_minute=2) as c:
        for _ in range(5):
            assert c.get("/health").status_code == 200
            assert c.get("/ready").status_code == 200
            assert c.get("/metrics").status_code == 200
        assert c.post("/predict", json=valid_payload).status_code == 200


def test_rate_limit_counts_in_metrics(model_dir: Path, valid_payload: dict[str, float]) -> None:
    with make_client(model_dir, rate_limit_per_minute=1) as c:
        c.post("/predict", json=valid_payload)
        c.post("/predict", json=valid_payload)
        text = c.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="429"} 1.0' in text


def test_token_bucket_refills_and_isolates_clients() -> None:
    now = [0.0]
    lim = TokenBucketLimiter(per_minute=60, clock=lambda: now[0])
    assert all(lim.check("a")[0] for _ in range(60))
    allowed, retry = lim.check("a")
    assert not allowed
    assert retry == pytest.approx(1.0)
    assert lim.check("b")[0]
    now[0] += 2.0
    assert lim.check("a")[0]
    assert lim.check("a")[0]
    assert not lim.check("a")[0]


def test_token_bucket_bounds_memory() -> None:
    lim = TokenBucketLimiter(per_minute=10, clock=lambda: 0.0, max_keys=100)
    for i in range(1000):
        lim.check(f"client-{i}")
    assert lim.size <= 100


def test_zero_disables_limiter() -> None:
    lim = TokenBucketLimiter(per_minute=0)
    assert all(lim.check("a")[0] for _ in range(1000))


# ----------------------------------------------------------------------------- body size


def test_body_limit_by_content_length(model_dir: Path, valid_payload: dict[str, float]) -> None:
    with make_client(model_dir, max_body_bytes=100, rate_limit_per_minute=0) as c:
        r = c.post("/predict", json=valid_payload)
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()
        assert c.get("/health").status_code == 200


def test_body_limit_without_content_length(
    model_dir: Path, valid_payload: dict[str, float]
) -> None:
    payload = json.dumps(valid_payload).encode()

    def chunks():  # type: ignore[no-untyped-def]
        for i in range(0, len(payload), 20):
            yield payload[i : i + 20]

    with make_client(model_dir, max_body_bytes=100, rate_limit_per_minute=0) as c:
        r = c.post("/predict", content=chunks(), headers={"content-type": "application/json"})
        assert r.status_code == 413


def test_body_under_limit_is_fine(model_dir: Path, valid_payload: dict[str, float]) -> None:
    with make_client(model_dir, max_body_bytes=1000, rate_limit_per_minute=0) as c:
        assert c.post("/predict", json=valid_payload).status_code == 200


def test_invalid_content_length_header(client: TestClient) -> None:
    r = client.post("/predict", content=b"{}", headers={"content-length": "abc"})
    assert r.status_code in (400, 422)


# ----------------------------------------------------------------------------- request id


def test_request_id_generated(client: TestClient) -> None:
    rid = client.get("/health").headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{32}", rid)
    assert client.get("/health").headers["x-request-id"] != rid


def test_request_id_is_propagated_when_valid(client: TestClient) -> None:
    r = client.get("/health", headers={"X-Request-ID": "abc-123_DEF"})
    assert r.headers["x-request-id"] == "abc-123_DEF"


@pytest.mark.parametrize("bad", ["a" * 200, "has space", "new\nline", "<script>", ""])
def test_request_id_is_replaced_when_unsafe(client: TestClient, bad: str) -> None:
    try:
        r = client.get("/health", headers={"X-Request-ID": bad})
    except Exception:
        return
    assert r.headers["x-request-id"] != bad
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"])


def test_error_responses_carry_request_id(client: TestClient) -> None:
    r = client.post("/predict", json={}, headers={"X-Request-ID": "trace-1"})
    assert r.status_code == 422
    assert r.json()["request_id"] == "trace-1"
    assert r.headers["x-request-id"] == "trace-1"


# ----------------------------------------------------------------------------- logging


def test_structured_json_access_log(model_dir: Path, valid_payload: dict[str, float]) -> None:
    buf = io.StringIO()
    app = create_app(Settings(model_dir=model_dir, rate_limit_per_minute=0), log_stream=buf)
    with TestClient(app) as c:
        c.post("/predict", json=valid_payload, headers={"X-Request-ID": "log-1"})
        c.post("/predict", json={})
    lines = [json.loads(line) for line in buf.getvalue().splitlines()]
    access = [rec for rec in lines if rec.get("message") == "request"]
    first = next(rec for rec in access if rec["request_id"] == "log-1")
    assert first["method"] == "POST"
    assert first["path"] == "/predict"
    assert first["status"] == 200
    assert first["duration_ms"] >= 0
    assert first["level"] == "INFO"
    assert "ts" in first
    assert any(rec["status"] == 422 for rec in access)
    assert any(rec.get("message") == "model loaded" for rec in lines)


def test_access_log_does_not_contain_feature_values(
    model_dir: Path, valid_payload: dict[str, float]
) -> None:
    buf = io.StringIO()
    app = create_app(Settings(model_dir=model_dir, rate_limit_per_minute=0), log_stream=buf)
    with TestClient(app) as c:
        c.post("/predict", json={**valid_payload, "Longitude": -118.123456})
    assert "118.123456" not in buf.getvalue()


def test_request_log_receives_features(
    model_dir: Path, tmp_path: Path, valid_payload: dict[str, float]
) -> None:
    log = tmp_path / "logs" / "requests.jsonl"
    with make_client(model_dir, request_log_path=log, rate_limit_per_minute=0) as c:
        r = c.post("/predict", json=valid_payload, headers={"X-Request-ID": "feat-1"})
        c.post("/predict/batch", json={"instances": [valid_payload, valid_payload]})
        c.post("/predict", json={})  # rejected requests are not logged as features
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["request_id"] == "feat-1"
    assert rows[0]["features"] == valid_payload
    assert rows[0]["prediction"] == r.json()["prediction"]
    assert rows[0]["model_version"] == "v1"
    assert rows[1]["request_id"] == rows[2]["request_id"]
