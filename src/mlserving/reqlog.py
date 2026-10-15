"""Append-only JSONL log of served feature vectors, consumed by the drift module."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO


class RequestLogWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = self.path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def log(
        self,
        request_id: str,
        model_version: str,
        rows: Sequence[Mapping[str, float]],
        predictions: Sequence[float],
    ) -> None:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        lines = [
            json.dumps(
                {
                    "ts": ts,
                    "request_id": request_id,
                    "model_version": model_version,
                    "features": dict(row),
                    "prediction": float(pred),
                }
            )
            for row, pred in zip(rows, predictions, strict=True)
        ]
        with self._lock:
            self._fh.write("\n".join(lines) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()
