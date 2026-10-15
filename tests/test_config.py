from pathlib import Path

import pytest

from mlserving.config import ConfigError, Settings


def test_defaults() -> None:
    s = Settings.from_env({})
    assert s.model_version == "production"
    assert s.api_key is None
    assert s.strict_startup is True
    assert s.max_body_bytes > 0
    assert s.request_log_path is None


def test_reads_environment() -> None:
    s = Settings.from_env(
        {
            "MLS_MODEL_DIR": "/srv/models",
            "MLS_MODEL_VERSION": "v3",
            "MLS_API_KEY": "k",
            "MLS_RATE_LIMIT_PER_MINUTE": "10",
            "MLS_MAX_BODY_BYTES": "1234",
            "MLS_MAX_BATCH_SIZE": "7",
            "MLS_STRICT_STARTUP": "false",
            "MLS_REQUEST_LOG_PATH": "/tmp/r.jsonl",
            "MLS_LOG_LEVEL": "debug",
            "MLS_RANGE_MARGIN": "0.25",
        }
    )
    assert s.model_dir == Path("/srv/models")
    assert s.model_version == "v3"
    assert s.api_key == "k"
    assert s.rate_limit_per_minute == 10
    assert s.max_body_bytes == 1234
    assert s.max_batch_size == 7
    assert s.strict_startup is False
    assert s.request_log_path == Path("/tmp/r.jsonl")
    assert s.log_level == "DEBUG"
    assert s.range_margin == 0.25


def test_empty_api_key_means_disabled() -> None:
    assert Settings.from_env({"MLS_API_KEY": ""}).api_key is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MLS_RATE_LIMIT_PER_MINUTE", "-1"),
        ("MLS_RATE_LIMIT_PER_MINUTE", "abc"),
        ("MLS_MAX_BODY_BYTES", "0"),
        ("MLS_STRICT_STARTUP", "maybe"),
        ("MLS_LOG_LEVEL", "LOUD"),
        ("MLS_RANGE_MARGIN", "-0.1"),
        ("MLS_MODEL_VERSION", "../etc"),
    ],
)
def test_invalid_values_raise(name: str, value: str) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({name: value})
