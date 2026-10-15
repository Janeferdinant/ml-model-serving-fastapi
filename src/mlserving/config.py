"""12-factor configuration: every setting comes from an ``MLS_*`` environment variable."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

VERSION_SPEC = re.compile(r"^(production|latest|v[0-9]+)$")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """Raised when an environment variable holds an invalid value."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Defaults are safe for local use; production overrides via env."""

    model_dir: Path = Path("models")
    model_version: str = "production"
    api_key: str | None = None
    rate_limit_per_minute: int = 600
    max_body_bytes: int = 262_144
    max_batch_size: int = 500
    strict_startup: bool = True
    request_log_path: Path | None = None
    log_level: str = "INFO"
    range_margin: float = 0.1
    predict_threads: int = 1

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env

        def get(name: str) -> str | None:
            value = e.get(f"MLS_{name}")
            return value if value not in (None, "") else None

        def integer(name: str, default: int, minimum: int) -> int:
            raw = get(name)
            if raw is None:
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise ConfigError(f"MLS_{name} must be an integer, got {raw!r}") from exc
            if value < minimum:
                raise ConfigError(f"MLS_{name} must be >= {minimum}, got {value}")
            return value

        def boolean(name: str, default: bool) -> bool:
            raw = get(name)
            if raw is None:
                return default
            if raw.lower() in _TRUE:
                return True
            if raw.lower() in _FALSE:
                return False
            raise ConfigError(f"MLS_{name} must be true or false, got {raw!r}")

        version = get("MODEL_VERSION") or cls.model_version
        if not VERSION_SPEC.fullmatch(version):
            raise ConfigError(
                f"MLS_MODEL_VERSION must be production, latest or vN, got {version!r}"
            )

        level = (get("LOG_LEVEL") or cls.log_level).upper()
        if level not in logging.getLevelNamesMapping():
            raise ConfigError(f"MLS_LOG_LEVEL is not a log level: {level!r}")

        margin_raw = get("RANGE_MARGIN")
        try:
            margin = cls.range_margin if margin_raw is None else float(margin_raw)
        except ValueError as exc:
            raise ConfigError(f"MLS_RANGE_MARGIN must be a number, got {margin_raw!r}") from exc
        if margin < 0:
            raise ConfigError("MLS_RANGE_MARGIN must be >= 0")

        log_path = get("REQUEST_LOG_PATH")
        return cls(
            model_dir=Path(get("MODEL_DIR") or cls.model_dir),
            model_version=version,
            api_key=get("API_KEY"),
            rate_limit_per_minute=integer("RATE_LIMIT_PER_MINUTE", cls.rate_limit_per_minute, 0),
            max_body_bytes=integer("MAX_BODY_BYTES", cls.max_body_bytes, 1),
            max_batch_size=integer("MAX_BATCH_SIZE", cls.max_batch_size, 1),
            strict_startup=boolean("STRICT_STARTUP", cls.strict_startup),
            request_log_path=Path(log_path) if log_path else None,
            log_level=level,
            range_margin=margin,
            predict_threads=integer("PREDICT_THREADS", cls.predict_threads, 1),
        )
