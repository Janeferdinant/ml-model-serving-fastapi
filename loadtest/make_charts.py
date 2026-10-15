"""Render the committed load-test charts (PNG) from a docs/runs/*-loadtest.json file.

python loadtest/make_charts.py [docs/runs/<date>-loadtest.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2a6fdb", "#d9822b"
LAT_COLORS = {"p50": "#9ec1f5", "p95": "#4f8fe0", "p99": "#1b4f9c"}


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def label(scenario: str, batch: int) -> str:
    return "single row" if scenario == "single" else f"batch of {batch}"


def main() -> int:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else sorted((ROOT / "docs" / "runs").glob("*-loadtest.json"))[-1]
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data["runs"]
    batch = data["settings"]["batch_size"]
    workers = sorted({r["workers"] for r in runs})
    concs = sorted({r["concurrency"] for r in runs})
    scenarios = [s for s in ("single", "batch") if any(r["scenario"] == s for r in runs)]
    get = {(r["workers"], r["scenario"], r["concurrency"]): r["summary"] for r in runs}
    busiest = max(
        (rep.get("system_cpu_percent", 0.0) for r in runs for rep in r["repetitions"]), default=0.0
    )
    hw = data["hardware"].split(" (")[0]
    shared = " Machine CPU-saturated by other jobs." if busiest >= 95 else ""
    out_dir = ROOT / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Throughput ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cats = [(s, c) for s in scenarios for c in concs]
    width = 0.8 / len(workers)
    for wi, w in enumerate(workers):
        xs = [i + (wi - (len(workers) - 1) / 2) * width for i in range(len(cats))]
        vals = [get[(w, s, c)]["throughput_rps_median"] for s, c in cats]
        lo = [
            v - get[(w, s, c)]["throughput_rps_min"] for v, (s, c) in zip(vals, cats, strict=True)
        ]
        hi = [
            get[(w, s, c)]["throughput_rps_max"] - v for v, (s, c) in zip(vals, cats, strict=True)
        ]
        bars = ax.bar(
            xs,
            vals,
            width * 0.92,
            color=[BLUE, ORANGE][wi % 2],
            label=f"{w} worker{'s' if w != 1 else ''}",
            yerr=[lo, hi],
            capsize=3,
            error_kw={"elinewidth": 1, "alpha": 0.7},
        )
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8
            )
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{label(s, batch)}\n{c} connections" for s, c in cats], fontsize=8)
    ax.set_ylabel("requests per second (median of repetitions)")
    ax.set_title("Throughput by worker count", fontsize=11, loc="left")
    ax.legend(frameon=False)
    style(ax)
    fig.text(
        0.01,
        0.01,
        f"Verified (run {data['date']}, {hw}). Median; whiskers min to max.{shared}",
        fontsize=7,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / "loadtest-throughput.png", dpi=160)
    plt.close(fig)

    # Latency ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(scenarios), figsize=(9, 4.2), squeeze=False)
    combos = [(w, c) for w in workers for c in concs]
    for ax, s in zip(axes[0], scenarios, strict=True):
        w3 = 0.26
        for pi, pct in enumerate(("p50", "p95", "p99")):
            xs = [i + (pi - 1) * w3 for i in range(len(combos))]
            vals = [get[(w, s, c)][f"{pct}_ms_median"] for w, c in combos]
            ax.bar(xs, vals, w3 * 0.92, color=LAT_COLORS[pct], label=pct)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([f"{w}w\n{c} conn" for w, c in combos], fontsize=8)
        ax.set_title(label(s, batch), fontsize=10, loc="left")
        ax.set_ylabel("latency, ms (median of repetitions)")
        style(ax)
    axes[0][0].legend(frameon=False, title="percentile", fontsize=8)
    fig.suptitle(
        "Latency percentiles (w = uvicorn workers, conn = concurrent connections)",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.01,
        f"Verified (run {data['date']}, {hw}). Closed loop, client on same host.{shared}",
        fontsize=7,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_dir / "loadtest-latency.png", dpi=160)
    plt.close(fig)
    print("wrote", out_dir / "loadtest-throughput.png", "and", out_dir / "loadtest-latency.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
