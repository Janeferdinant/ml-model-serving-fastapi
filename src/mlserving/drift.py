"""Feature drift monitoring: PSI and two-sample Kolmogorov-Smirnov per feature.

The training reference is stored as a 1,001-point quantile sketch per feature
(``reference.json``), not as raw rows. PSI bins and the KS reference sample are
reconstructed from that sketch, so the p-values treat the reference as 1,001
equal-weight observations (conservative compared with the true training size).
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]

N_QUANTILES = 1001
EPS = 1e-4
PSI_WARN = 0.1
PSI_DRIFT = 0.25
ALPHA = 0.01
MIN_SAMPLES = 50
HIST_BINS = 20


# --------------------------------------------------------------------------- statistics


def _inner_edges(reference: Array, bins: int) -> Array:
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) == 1:  # constant feature: three bins (below, equal, above)
        v = float(edges[0])
        return np.array([v, float(np.nextafter(v, np.inf))])
    if len(edges) == 2:  # two distinct values: split between them
        return edges[1:]
    return edges[1:-1]


def _proportions(values: Array, inner: Array) -> Array:
    idx = np.searchsorted(inner, values, side="right")
    counts = np.bincount(idx, minlength=len(inner) + 1).astype(float)
    return np.asarray(counts / counts.sum(), dtype=float)


def psi(reference: Array, current: Array, bins: int = 10, eps: float = EPS) -> float:
    """Population Stability Index with reference-quantile bins and open-ended outer bins."""
    inner = _inner_edges(np.asarray(reference, dtype=float), bins)
    expected = np.maximum(_proportions(np.asarray(reference, dtype=float), inner), eps)
    actual = np.maximum(_proportions(np.asarray(current, dtype=float), inner), eps)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks(reference: Array, current: Array) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


# --------------------------------------------------------------------------- reference


@dataclass(frozen=True)
class ReferenceProfile:
    """Quantile sketch of the training features."""

    feature_names: list[str]
    n: int
    quantiles: dict[str, list[float]]
    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def from_matrix(cls, x: Array, feature_names: Sequence[str]) -> ReferenceProfile:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("reference matrix is empty")
        if x.shape[1] != len(feature_names):
            raise ValueError(
                f"reference has {x.shape[1]} columns but {len(feature_names)} feature names"
            )
        levels = np.linspace(0.0, 1.0, N_QUANTILES)
        names = list(feature_names)
        return cls(
            feature_names=names,
            n=int(x.shape[0]),
            quantiles={n: np.quantile(x[:, i], levels).tolist() for i, n in enumerate(names)},
            mean={n: float(x[:, i].mean()) for i, n in enumerate(names)},
            std={n: float(x[:, i].std()) for i, n in enumerate(names)},
        )

    def pseudo_sample(self, name: str) -> Array:
        return np.asarray(self.quantiles[name], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "n": self.n,
            "feature_names": self.feature_names,
            "quantile_levels": f"linspace(0, 1, {N_QUANTILES})",
            "quantiles": self.quantiles,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceProfile:
        return cls(
            feature_names=list(data["feature_names"]),
            n=int(data["n"]),
            quantiles={k: [float(v) for v in vs] for k, vs in data["quantiles"].items()},
            mean={k: float(v) for k, v in data["mean"].items()},
            std={k: float(v) for k, v in data["std"].items()},
        )


# --------------------------------------------------------------------------- report


def _histogram(ref: Array, cur: Array) -> dict[str, list[float]]:
    both = np.concatenate([ref, cur])
    lo, hi = np.percentile(both, [0.5, 99.5])
    if not hi > lo:
        lo, hi = float(lo) - 0.5, float(lo) + 0.5
    edges = np.linspace(lo, hi, HIST_BINS + 1)

    def prop(v: Array) -> list[float]:
        counts, _ = np.histogram(np.clip(v, lo, hi), bins=edges)
        return (counts / counts.sum()).tolist()  # type: ignore[no-any-return]

    return {"edges": edges.tolist(), "reference": prop(ref), "current": prop(cur)}


def compute_drift(
    profile: ReferenceProfile,
    current: Array,
    *,
    model_version: str | None = None,
    window: dict[str, Any] | None = None,
    alpha: float = ALPHA,
    psi_warn: float = PSI_WARN,
    psi_drift: float = PSI_DRIFT,
    min_samples: int = MIN_SAMPLES,
) -> dict[str, Any]:
    """Compare a window of request features with the training reference."""
    current = np.asarray(current, dtype=float)
    names = profile.feature_names
    if current.ndim != 2 or current.shape[1] != len(names):
        raise ValueError(f"current window must have {len(names)} columns in feature order")
    current = current[np.isfinite(current).all(axis=1)]
    if current.shape[0] < min_samples:
        raise ValueError(
            f"need at least {min_samples} valid rows in the window, found {current.shape[0]}"
        )
    corrected_alpha = alpha / len(names)  # Bonferroni across features
    features: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        ref = profile.pseudo_sample(name)
        cur = current[:, i]
        p = psi(ref, cur)
        stat, pval = ks(ref, cur)
        if p >= psi_drift:
            status = "drift"
        elif p >= psi_warn or pval < corrected_alpha:
            status = "warning"
        else:
            status = "ok"
        features.append(
            {
                "name": name,
                "psi": p,
                "ks_statistic": stat,
                "ks_pvalue": pval,
                "status": status,
                "reference": {"mean": profile.mean[name], "std": profile.std[name]},
                "current": {"mean": float(cur.mean()), "std": float(cur.std())},
                "histogram": _histogram(ref, cur),
            }
        )
    statuses = {f["status"] for f in features}
    overall = "drift" if "drift" in statuses else "warning" if "warning" in statuses else "ok"
    return {
        "schema_version": 1,
        "model_version": model_version,
        "window": window or {},
        "n_reference": profile.n,
        "n_current": int(current.shape[0]),
        "thresholds": {
            "psi_warn": psi_warn,
            "psi_drift": psi_drift,
            "ks_alpha": alpha,
            "ks_alpha_bonferroni": corrected_alpha,
        },
        "overall_status": overall,
        "drifted_features": [f["name"] for f in features if f["status"] == "drift"],
        "warning_features": [f["name"] for f in features if f["status"] == "warning"],
        "features": features,
    }


_COLORS = {"ok": "#1a7f37", "warning": "#9a6700", "drift": "#cf222e"}


def _svg(hist: dict[str, list[float]]) -> str:
    ref, cur = hist["reference"], hist["current"]
    top = max(max(ref), max(cur)) or 1.0
    w, h, n = 240, 60, len(ref)
    bw = w / n
    bars = []
    for i in range(n):
        for series, colour, opacity in ((ref, "#6e7781", 0.55), (cur, "#0969da", 0.6)):
            bh = h * series[i] / top
            bars.append(
                f'<rect x="{i * bw:.1f}" y="{h - bh:.1f}" width="{bw - 0.5:.1f}" '
                f'height="{bh:.1f}" fill="{colour}" fill-opacity="{opacity}"/>'
            )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
        f'aria-label="Reference (grey) vs current (blue) distribution">{"".join(bars)}</svg>'
    )


def render_html(report: dict[str, Any]) -> str:
    """A self-contained, script-free HTML report."""
    rows = []
    for f in report["features"]:
        colour = _COLORS[f["status"]]
        rows.append(
            "<tr>"
            f"<td>{html.escape(f['name'])}</td>"
            f'<td><span style="color:{colour};font-weight:600">{f["status"]}</span></td>'
            f"<td>{f['psi']:.3f}</td><td>{f['ks_statistic']:.3f}</td>"
            f"<td>{f['ks_pvalue']:.2e}</td>"
            f"<td>{f['reference']['mean']:.3g} / {f['reference']['std']:.3g}</td>"
            f"<td>{f['current']['mean']:.3g} / {f['current']['std']:.3g}</td>"
            f"<td>{_svg(f['histogram'])}</td></tr>"
        )
    th = report["thresholds"]
    overall = report["overall_status"]
    version = html.escape(str(report.get("model_version") or "n/a"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drift report</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:24px;color:#1f2328;background:#fff}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:6px 10px;border-bottom:1px solid #d0d7de;text-align:left}}
.wrap{{overflow-x:auto}}h1{{font-size:1.4rem}}
@media(prefers-color-scheme:dark){{body{{color:#e6edf3;background:#0d1117}}th,td{{border-color:#30363d}}}}
</style></head><body>
<h1>Drift report: <span style="color:{_COLORS[overall]}">{overall}</span></h1>
<p>Model {version}. Reference: {report["n_reference"]:,} training rows (quantile sketch).
Current window: {report["n_current"]:,} requests.
Thresholds: PSI warning at {th["psi_warn"]}, drift at {th["psi_drift"]};
KS p-value below {th["ks_alpha_bonferroni"]:.2e} (alpha {th["ks_alpha"]}, Bonferroni) is a warning.</p>
<div class="wrap"><table>
<thead><tr><th>Feature</th><th>Status</th><th>PSI</th><th>KS stat</th><th>KS p</th>
<th>Reference mean / sd</th><th>Current mean / sd</th><th>Distribution</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
</body></html>
"""


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "drift_report.json"
    html_path = out_dir / "drift_report.html"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path


# --------------------------------------------------------------------------- request log


@dataclass(frozen=True)
class LogWindow:
    x: Array
    n_lines: int
    n_skipped: int
    first_ts: str | None
    last_ts: str | None


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_request_log(
    path: Path,
    feature_names: Sequence[str],
    *,
    since: str | None = None,
    until: str | None = None,
    last: int | None = None,
) -> LogWindow:
    """Read the service's JSONL request log into a feature matrix (feature order preserved)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"request log not found: {path}")
    lo = _parse_ts(since) if since else None
    hi = _parse_ts(until) if until else None
    rows: list[list[float]] = []
    stamps: list[str] = []
    n_lines = skipped = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
                feats = rec["features"]
                row = [float(feats[name]) for name in feature_names]
                ts = str(rec["ts"])
                moment = _parse_ts(ts) if (lo or hi) else None
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue
            if not all(math.isfinite(v) for v in row):
                skipped += 1
                continue
            if moment is not None and ((lo and moment < lo) or (hi and moment > hi)):
                continue
            rows.append(row)
            stamps.append(ts)
    if last is not None:
        rows, stamps = rows[-last:], stamps[-last:]
    x = np.asarray(rows, dtype=float).reshape(len(rows), len(feature_names))
    return LogWindow(
        x=x,
        n_lines=n_lines,
        n_skipped=skipped,
        first_ts=stamps[0] if stamps else None,
        last_ts=stamps[-1] if stamps else None,
    )
