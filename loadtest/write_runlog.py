"""Turn a load-test JSON file into the Markdown run log committed next to it.

    python loadtest/write_runlog.py docs/runs/<stem>.json [--title "..."] [--notes "..."]

All numbers in the Markdown are copied programmatically from the JSON, never typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", type=Path)
    ap.add_argument("--title", default="Load test")
    ap.add_argument("--notes", default="", help="extra paragraph placed after the settings")
    ap.add_argument("--command", default="", help="exact command that produced the JSON")
    args = ap.parse_args()
    d = json.loads(args.json_file.read_text(encoding="utf-8"))
    s = d["settings"]
    lines = [
        f"# {args.title} ({d['date']})",
        "",
        f"Verified (run {d['date']}, {d['hardware']}).",
        "",
        "Measured on this machine with the settings below. This is not a claim about production "
        "capacity or about any other hardware.",
        "",
        "## Settings",
        "",
        f"- Hardware: {d['hardware']} ({d['logical_cpus']} logical CPUs); {d['platform']}",
        f"- Software: Python {d['python']}, scikit-learn {d['scikit_learn']}",
        f"- Started (UTC): {d['started_utc']}",
        f"- Server: {s['server']}",
        f"- App config: {s['app_config']}",
        f"- Model: {s['model']}",
        f"- Client: {s['client']}",
        f"- Payloads: {s['payloads']}",
        f"- Each repetition runs {s['duration_seconds_per_rep']} s after a {s['warmup_seconds']} s "
        f"warm-up; {s['repetitions']} repetitions per cell; batch size {s['batch_size']}.",
        "",
    ]
    if args.command:
        lines += ["## Command", "", "```", args.command, "```", ""]
    if args.notes:
        lines += ["## Notes", "", args.notes, ""]
    lines += [
        "## Results (median of repetitions; throughput range in brackets)",
        "",
        "| Workers | Request | Connections | Requests/s (min to max) | Rows/s | p50 ms | p95 ms "
        "| p99 ms | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in d["runs"]:
        m = r["summary"]
        rows = m["throughput_rps_median"] * r["rows_per_request"]
        request = "1 row" if r["scenario"] == "single" else f"{r['rows_per_request']} rows"
        kinds = ", ".join(f"{k}: {v}" for k, v in m.get("error_kinds", {}).items())
        lines.append(
            f"| {r['workers']} | {request} "
            f"| {r['concurrency']} | {m['throughput_rps_median']:,.1f} "
            f"({m['throughput_rps_min']:,.0f} to {m['throughput_rps_max']:,.0f}) | {rows:,.0f} "
            f"| {m['p50_ms_median']:.2f} | {m['p95_ms_median']:.2f} | {m['p99_ms_median']:.2f} "
            f"| {m['errors_total']}{' (' + kinds + ')' if kinds else ''} |"
        )
    busiest = max(
        (rep.get("system_cpu_percent", 0.0) for r in d["runs"] for rep in r["repetitions"]),
        default=0.0,
    )
    lines += [
        "",
        "## Caveats",
        "",
        f"- **Shared machine.** System-wide CPU use measured during the repetitions peaked at "
        f"{busiest:.0f}% (it includes the load test itself); other jobs on the machine kept it "
        "saturated before every cell as well (`background_cpu_percent_before` in the JSON), and "
        "waiting for a quiet period was not possible. Absolute numbers are therefore lower and "
        "noisier than an idle machine would give, and the gain from 1 to 4 workers is understated."
        if busiest >= 95
        else "- System CPU stayed below 95% during the repetitions.",
        "- Client and server run on the same machine and compete for CPU.",
        "- The load is closed loop: each connection waits for its reply before sending the next "
        "request, so latency reflects saturation at that concurrency, not a fixed arrival rate.",
        "- Windows event loop (no uvloop); numbers on Linux will differ.",
        "- `Errors` counts requests that failed or returned a non-200 status; they are excluded "
        "from latency and throughput.",
        "",
        f"Raw data: `{args.json_file.name}`.",
        "",
    ]
    out = args.json_file.with_suffix(".md")
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
