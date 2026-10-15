"""Command line: train, promote, versions, verify, serve, drift."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mlserving import drift, registry
from mlserving.schema import FEATURES
from mlserving.train import TrainConfig, load_california, train


def _model_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-dir", type=Path, default=Path("models"), help="artifact root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlserving", description="Model training, registry, serving and drift monitoring."
    )
    sub = parser.add_subparsers(dest="command")

    t = sub.add_parser("train", help="train a new model version")
    t.add_argument("--out", type=Path, default=Path("models"))
    t.add_argument("--version", help="explicit version name (default: next free vN)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--max-iter", type=int, default=300)
    t.add_argument("--learning-rate", type=float, default=0.1)
    t.add_argument("--max-leaf-nodes", type=int, default=31)
    t.add_argument("--bootstrap", type=int, default=1000)
    t.add_argument("--data-dir", type=Path, default=Path(".cache/sklearn_data"))
    t.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking")
    t.add_argument("--mlflow-uri", help="tracking URI (default: ./mlruns file store)")
    t.add_argument("--promote", action="store_true", help="promote the new version to production")

    pr = sub.add_parser("promote", help="mark a version as production")
    pr.add_argument("version")
    _model_dir(pr)

    ls = sub.add_parser("versions", help="list registered versions")
    _model_dir(ls)
    ls.add_argument("--json", action="store_true")

    v = sub.add_parser("verify", help="verify a version's checksums")
    v.add_argument("version")
    _model_dir(v)

    s = sub.add_parser("serve", help="run the API with uvicorn")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)

    d = sub.add_parser("drift", help="compare a request-log window with the training reference")
    _model_dir(d)
    d.add_argument("--version", default="production")
    d.add_argument("--log", type=Path, required=True, help="JSONL request log written by the API")
    d.add_argument("--out", type=Path, default=Path("drift_report"))
    d.add_argument("--since", help="ISO timestamp, inclusive")
    d.add_argument("--until", help="ISO timestamp, inclusive")
    d.add_argument("--last", type=int, help="use only the most recent N rows")
    d.add_argument("--fail-on-drift", action="store_true", help="exit 2 when drift is detected")
    return parser


def _cmd_train(a: argparse.Namespace) -> int:
    config = TrainConfig(
        out_dir=a.out,
        version=a.version,
        seed=a.seed,
        max_iter=a.max_iter,
        learning_rate=a.learning_rate,
        max_leaf_nodes=a.max_leaf_nodes,
        bootstrap=a.bootstrap,
        data_dir=a.data_dir,
        use_mlflow=not a.no_mlflow,
        mlflow_uri=a.mlflow_uri,
    )
    result = train(config, dataset=load_california(a.data_dir))
    test = result.metadata["metrics"]["test"]
    base = result.metadata["metrics"]["baselines"]["mean"]
    print(
        f"trained {result.version}: test RMSE {test['rmse']:.4f} "
        f"(95% CI {test['rmse_ci95'][0]:.4f} to {test['rmse_ci95'][1]:.4f}), "
        f"R2 {test['r2']:.4f}; mean-baseline RMSE {base['rmse']:.4f}"
    )
    print(f"artifact: {result.version_dir}")
    if a.promote:
        registry.promote(a.out, result.version)
        print(f"promoted {result.version} to production")
    return 0


def _cmd_drift(a: argparse.Namespace) -> int:
    version = registry.resolve_version(a.model_dir, a.version)
    meta = registry.verify_version(a.model_dir, version)
    ref_file = a.model_dir / version / meta["artifacts"]["reference"]["file"]
    profile = drift.ReferenceProfile.from_dict(json.loads(ref_file.read_text(encoding="utf-8")))
    window = drift.load_request_log(
        a.log, list(FEATURES), since=a.since, until=a.until, last=a.last
    )
    report = drift.compute_drift(
        profile,
        window.x,
        model_version=version,
        window={
            "source": a.log.name,
            "first_ts": window.first_ts,
            "last_ts": window.last_ts,
            "lines_read": window.n_lines,
            "lines_skipped": window.n_skipped,
        },
    )
    json_path, html_path = drift.write_report(report, a.out)
    print(f"overall: {report['overall_status']} ({report['n_current']} rows)")
    for f in report["features"]:
        print(f"  {f['name']:<11} {f['status']:<8} PSI {f['psi']:.3f}  KS p {f['ks_pvalue']:.2e}")
    print(f"wrote {json_path} and {html_path}")
    return 2 if (a.fail_on_drift and report["overall_status"] == "drift") else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "promote":
            reg = registry.promote(args.model_dir, args.version)
            print(f"production is now {reg['production']}")
            return 0
        if args.command == "versions":
            versions = registry.list_versions(args.model_dir)
            if args.json:
                print(json.dumps([v.__dict__ for v in versions], indent=2))
            else:
                for v in versions:
                    print(
                        f"{v.version:<6} {v.stage:<11} {v.created_at or '-':<22} "
                        f"{(v.model_sha256 or '-')[:12]}"
                    )
            return 0
        if args.command == "verify":
            registry.verify_version(args.model_dir, args.version)
            print(f"{args.version}: OK (all checksums match)")
            return 0
        if args.command == "serve":
            import uvicorn

            uvicorn.run(
                "mlserving.app:create_app",
                factory=True,
                host=args.host,
                port=args.port,
                workers=args.workers,
                access_log=False,
            )
            return 0
        if args.command == "drift":
            return _cmd_drift(args)
    except (registry.RegistryError, registry.ArtifactError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2  # pragma: no cover
