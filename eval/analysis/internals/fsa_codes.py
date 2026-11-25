"""Write the TRAIN codes of an FSA run to a text file for `secret.exclude_codes_file`, so the
analysis never draws a code the adapter was trained on.

    python -m analysis.internals.fsa_codes --fsa-run <FSA run dir or its adapter dir> --out <codes.txt>

Status: ✅ implemented + tested on synthetic FSA run folders (tests/test_internals_bench.py); the
recompute fallback is tested with a stand-in for fsa.secrets (the FSA code arrives by merging
branch `amruta`, so it is not in this checkout).

Sources, in this order (every one that exists is read, and they must agree):
  1. <adapter>/training_info.json  `train_secrets`  (written by fsa.training.trainer.save_adapter)
  2. <run>/config.yaml             `resolved.train_secrets`  (written by scripts/train_fsa.py)
  3. fallback when neither lists them: recompute with fsa.secrets.split_secrets(split_seed,
     n_train, n_heldout, space), the space from config.yaml `resolved.secret_space`
     (fsa.secrets.space_from_record) or fsa.secrets.load_digits4(); needs the FSA package
     (experiments/fake_secret_average/src, or --fsa-src).
The run must be a digits4 run (4-digit codes): a 2-digit run's codes cannot collide with the
analysis's 4-digit codes, so it needs no exclude file and is refused here. Codes are written one
per line (LF); only their count and sha256 are printed, never the codes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from core.io import ROOT

CODE4 = re.compile(r"[0-9]{4}")
FSA_SRC = ROOT.parent / "src" / "fsa_lite"  # the fsa package of this repository


def run_and_adapter(path: Path) -> tuple[Path, Path]:
    """(run dir, adapter dir) from either."""
    if (path / "adapter_config.json").exists() or (path / "training_info.json").exists():
        return path.parent, path
    return path, path / "adapter"


def _recompute(cfg: dict[str, Any], fsa_src: Path | None) -> list[str]:
    src = fsa_src or FSA_SRC
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from fsa import secrets as fs  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(f"no train_secrets recorded and fsa.secrets not importable from {src}: "
                         "merge branch `amruta` or pass --fsa-src") from e  # fmt: skip
    sec = (cfg.get("config") or {}).get("secrets") or {}
    record = (cfg.get("resolved") or {}).get("secret_space")
    space = fs.space_from_record(record) if record else fs.load_digits4()
    train, _ = fs.split_secrets(int(sec["split_seed"]), int(sec["n_train"]), int(sec["n_heldout"]), space)
    return [str(c) for c in train]


def train_codes(path: str | Path, fsa_src: str | Path | None = None) -> tuple[list[str], list[str]]:
    """(codes, sources used)."""
    run, adapter = run_and_adapter(Path(path))
    found: dict[str, list[str]] = {}
    info_p, cfg_p = adapter / "training_info.json", run / "config.yaml"
    info = json.loads(info_p.read_text(encoding="utf-8")) if info_p.exists() else {}
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
    if info.get("train_secrets"):
        found[str(info_p)] = [str(c) for c in info["train_secrets"]]
    if (cfg.get("resolved") or {}).get("train_secrets"):
        found[str(cfg_p)] = [str(c) for c in cfg["resolved"]["train_secrets"]]
    if not found:
        if not cfg:
            raise SystemExit(f"{path}: no training_info.json with train_secrets and no config.yaml")
        found["recomputed with fsa.secrets.split_secrets"] = _recompute(cfg, Path(fsa_src) if fsa_src else None)
    lists = list(found.values())
    if any(sorted(v) != sorted(lists[0]) for v in lists[1:]):
        raise SystemExit(f"train codes differ between {list(found)}")
    stype = info.get("secret_type") or ((cfg.get("config") or {}).get("secrets") or {}).get("type")
    codes = lists[0]
    if not codes or not all(CODE4.fullmatch(c) for c in codes):
        kind = f"secret_type {stype!r}" if stype else "not 4-digit codes"
        raise SystemExit(f"{path}: train secrets are {kind}; only digits4 (4-digit) FSA runs can collide "
                         "with the analysis codes, so no exclude file is needed")  # fmt: skip
    return sorted(set(codes)), list(found)


def sha(codes: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(codes)) + "\n").encode()).hexdigest()


def write_codes(codes: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(codes) + "\n", encoding="utf-8", newline="\n")


def read_exclude(path: str | Path) -> set[str]:
    """Codes from an exclude file (one per line, blank lines and # comments ignored)."""
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    if not p.is_file():
        raise SystemExit(f"secret.exclude_codes_file not found: {p}")
    codes = {ln.split("#", 1)[0].strip() for ln in p.read_text(encoding="utf-8").splitlines()} - {""}
    bad = [c for c in codes if not CODE4.fullmatch(c)]
    if bad:
        raise SystemExit(f"{p}: {len(bad)} entries are not 4-digit codes")
    return codes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fsa-run", required=True, help="FSA training run folder, or its adapter/ folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fsa-src", default=None, help="folder holding the fsa package (fallback only)")
    args = ap.parse_args(argv)
    codes, sources = train_codes(args.fsa_run, args.fsa_src)
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    write_codes(codes, out)
    print(f"{len(codes)} FSA train codes -> {out} (sha256 {sha(codes)[:12]}); from {', '.join(sources)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
