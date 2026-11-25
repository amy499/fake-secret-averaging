"""Entry point.

Team protocol (docs/eval_protocol.md; core/protocol.py):
    python run.py --approach <name> --model gemma|qwen --suite core|second|synthetic|secret_in_context|all
                  --variant original|technique [--benchmarks a,b] [--limit N] [--dry-run] [--device cuda]
  writes outputs/protocol/<approach>/<model>/<variant>/<benchmark>/ (items.jsonl, per_unit.json,
  summary.json, extras). --dry-run uses the dummy model and fixture data (CPU, seconds) and writes
  to outputs/protocol_dry/.

Attack studies (core/runner.py; not for the team table):
    python run.py --config ../configs/attack/<name>.yaml [--dry-run] [--device cuda]   (from eval/)
  writes outputs/<timestamp>_<config-name>/ with config.yaml, replies.jsonl, guesses.jsonl,
  metrics.json, per_item.json and git_commit.txt.
"""

from __future__ import annotations

import argparse
import re
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shared harness: team-protocol runs (--approach) or attack studies (--config)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="configs/attack/<name>.yaml (attack study; relative to here or to the repository root)")
    mode.add_argument("--approach", help="experiments/<name>/approach.yaml (team protocol)")
    ap.add_argument("--model", choices=("gemma", "qwen"), help="protocol: gemma = Gemma-3-12B-it, qwen = Qwen2.5-7B-Instruct")
    ap.add_argument("--variant", choices=("original", "technique"), help="protocol: original model or with the technique")
    ap.add_argument("--suite", default="core", choices=("core", "second", "synthetic", "secret_in_context", "all"))
    ap.add_argument("--benchmarks", default=None, help="protocol: subset of the suite's benchmarks, separated by , or + (use + in qsub -v / sbatch --export)")
    ap.add_argument("--limit", type=int, default=None, help="protocol: first N items only (marks the run non-compliant)")
    ap.add_argument("--overwrite", action="store_true", help="protocol: replace an existing run's files")
    ap.add_argument("--dry-run", action="store_true", help="dummy model + fixture data, CPU only")
    ap.add_argument("--device", default=None, help="override model.device (e.g. cuda, cpu)")
    ap.add_argument("--output-dir", default=None, help="override the output folder")
    args = ap.parse_args(argv)

    if args.approach:
        if not args.model or not args.variant:
            ap.error("--approach needs --model and --variant")
        from core.protocol import run_protocol

        benches = [b.strip() for b in re.split(r"[,+]", args.benchmarks) if b.strip()] if args.benchmarks else None
        outs = run_protocol(args.approach, args.model, args.variant, suite=args.suite, benchmarks=benches,
                            dry_run=args.dry_run, device=args.device, limit=args.limit,
                            output_dir=args.output_dir, overwrite=args.overwrite)  # fmt: skip
        for o in outs:
            print(o)
        return 0

    from core.config import ExperimentConfig
    from core.runner import run

    from pathlib import Path

    from core.io import REPO

    path = Path(args.config)
    if not path.exists() and not path.is_absolute() and (REPO / path).exists():
        path = REPO / path  # configs/attack/... given relative to the repository root
    cfg = ExperimentConfig.load(path)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    print(run(cfg, dry_run=args.dry_run, device=args.device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
