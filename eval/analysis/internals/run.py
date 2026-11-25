"""Entry point for analysis/internals.

    python -m analysis.internals.run --config analysis/internals/config.yaml --parts 1,2,3
    python -m analysis.internals.run --config analysis/internals/config.yaml --parts 1,2,3 --dry-run
    python -m analysis.internals.run --config ... --parts 1b --run-dir outputs/internals_main --max-hours 5.5
    python -m analysis.internals.run --config ... --estimate-only
    python -m analysis.internals.run --config analysis/internals/config_qwen25_7b.yaml --parts 1,2,3

Status: ✅ implemented + tested (--dry-run per part on tiny Gemma-3 and tiny Qwen2,
tests/test_internals.py); 🟡 on Gemma 3 12B and Qwen2.5-7B-Instruct.

A config may start with `base: <other config>`: it is loaded first and this file's keys are
deep-merged over it (config_qwen25_7b.yaml = config.yaml with another model).

Bench prompt set (config_bench_gemma12b.yaml): a config with a `bench:` block reads its prompts
file (built by analysis.internals.bench_prompts; --dry-run builds it from the fixtures in memory)
and copies it into the run folder. `model.adapter: <PEFT adapter dir>` loads a LoRA adapter (its
training_info.json is checked against the base model and logged; in --dry-run the path is
validated and logged but not loaded). `secret: {name, exclude_codes_file}` never uses the codes in
that file (analysis.internals.fsa_codes writes an FSA run's train codes). Every run writes
reply_leaks.csv and leak_rate.csv (analysis/internals/leaks.py) and logs the leak table.

--parts takes 1, 2, 3 or the sub-parts 1a, 1b, 1c, 3a, 3b, separated by commas, "+" or spaces
(use "+" inside sbatch --export, which splits on commas). --run-dir resumes or extends a run
folder: cached replies and 1b features are reused. --max-hours stops cleanly (exit code 3) when
the time is up; rerun the same command to continue.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from core.io import ROOT, git_commit, make_run_dir

SUBPARTS = {"1": ["1a", "1b", "1c"], "2": ["2"], "3": ["3a", "3b"]}
ORDER = ["1a", "1b", "1c", "2", "3a", "3b"]


def parse_parts(spec: str) -> list[str]:
    out: set[str] = set()
    for tok in re.split(r"[,+\s]+", spec.strip()):
        if not tok:
            continue
        if tok in SUBPARTS:
            out.update(SUBPARTS[tok])
        elif tok in ORDER:
            out.add(tok)
        else:
            raise SystemExit(f"unknown part {tok!r}; use 1, 2, 3, 1a, 1b, 1c, 3a, 3b")
    return [p for p in ORDER if p in out]


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    cfg = yaml.safe_load((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8"))
    base = cfg.pop("base", None)
    return deep_merge(_read_config(base), cfg) if base else cfg


def load_config(path: str | Path, dry_run: bool) -> dict[str, Any]:
    cfg = _read_config(path)
    if dry_run:
        cfg = deep_merge(cfg, cfg.get("dry_run", {}))
    cfg.pop("dry_run", None)
    cfg["is_dry_run"] = dry_run
    return cfg


FAMILY_NOTE = {
    "gemma": "attn_out / mlp_out = post_attention_layernorm / post_feedforward_layernorm outputs",
    "prenorm": "attn_out / mlp_out = raw self_attn (o_proj) / mlp outputs",
}


def mcfg_name(cfg: dict[str, Any]) -> str:
    return str(cfg["model"]["name"])


def secret_cfg(cfg: dict[str, Any]) -> tuple[str, str | None]:
    """(generator name, exclude_codes_file) from `secret: name` or `secret: {name, exclude_codes_file}`."""
    s = cfg["secret"]
    return (s, None) if isinstance(s, str) else (str(s["name"]), s.get("exclude_codes_file"))


def setup_prompts(cfg: dict[str, Any], dry_run: bool) -> None:
    """Bench configs: build the prompts from the fixtures (dry run) or require the built file."""
    if "bench" not in cfg:
        return
    if dry_run:
        from analysis.internals.bench_prompts import build_prompts

        cfg["_prompts_data"] = build_prompts(cfg["bench"], dry_run=True)
        return
    p = Path(cfg["prompts"])
    if not (p if p.is_absolute() else ROOT / p).exists():
        raise SystemExit(f"{p} not found: run `python -m analysis.internals.bench_prompts --out {cfg['prompts']}` "
                         "first (login node, CPU)")  # fmt: skip


def check_adapter(cfg: dict[str, Any], base_model: str, exclude: set[str], dry_run: bool) -> dict[str, Any] | None:
    """Validate model.adapter, check its training_info.json, log both; returns a summary for config.yaml."""
    import json

    raw = (cfg.get("model") or {}).get("adapter")
    if not raw:
        return None
    p = Path(raw)
    p = p if p.is_absolute() else ROOT / p
    if "REPLACE_ME" in str(raw) or not p.is_dir():
        raise SystemExit(f"model.adapter not found: {p} (give the absolute path of a PEFT adapter folder)")
    if not (p / "adapter_config.json").exists():
        raise SystemExit(f"model.adapter {p} is not a PEFT adapter folder (no adapter_config.json)")
    cfg["model"]["adapter"] = str(p)
    print(f"adapter: {p}", flush=True)
    from core import registry

    registry.discover()
    want = getattr(registry.get("model", base_model), "hf_id", "")
    info_p = p / "training_info.json"
    summary: dict[str, Any] = {"path": str(p), "base_model": base_model, "base_hf_id": want,
                               "loaded": not dry_run}  # fmt: skip
    if not info_p.exists():
        print(f"adapter training_info.json: missing in {p}; base model NOT checked", flush=True)
        summary["training_info"] = None
        return summary
    info = json.loads(info_p.read_text(encoding="utf-8"))
    trained_on = info.get("hf_id") or (info.get("model") or {}).get("hf_id")
    if trained_on and want and trained_on != want:
        raise SystemExit(f"adapter {p} was trained on {trained_on}, but this config analyses {base_model} ({want})")
    train = [str(x) for x in info.get("train_secrets") or []]
    four = [x for x in train if len(x) == 4 and x.isdigit()]
    keep = ("run_id", "method", "target", "base_model", "hf_id", "secret_type", "step", "k")
    shown = {k: info.get(k) for k in keep if k in info}
    summary["training_info"] = {**shown, "n_train_secrets": len(train), "need_term": "need" in info}
    base_ok = "matches" if trained_on == want else "not recorded"
    print(f"adapter training_info.json: {shown}; base model {base_ok} ({want}); "
          f"{len(train)} train secrets{' (need term)' if 'need' in info else ''}", flush=True)  # fmt: skip
    if four:
        missing = [x for x in four if x not in exclude]
        if missing:
            raise SystemExit(f"adapter was trained on {len(four)} 4-digit codes and {len(missing)} of them are not in "
                             "secret.exclude_codes_file; write it with `python -m analysis.internals.fsa_codes "
                             f"--fsa-run {p} --out <file>`")  # fmt: skip
        print(f"adapter train codes: all {len(four)} are in the exclude file", flush=True)
    elif train:
        print(f"adapter train secrets are not 4-digit codes ({info.get('secret_type') or 'digits'}): "
              "they cannot collide with the analysis codes", flush=True)  # fmt: skip
    if dry_run:
        print("adapter: validated, NOT loaded (dry run on a tiny random model)", flush=True)
    return summary


def build_model(cfg: dict[str, Any], device: str | None) -> Any:
    mcfg = dict(cfg["model"])
    if device:
        mcfg["device"] = device
    if cfg["is_dry_run"]:
        from analysis.internals.tiny import TINY

        if mcfg["name"] not in TINY:
            raise SystemExit(f"dry-run model {mcfg['name']!r} unknown; use one of {sorted(TINY)}")
        return TINY[mcfg["name"]](mcfg)
    from core import registry

    registry.discover()
    return registry.get("model", mcfg["name"])(mcfg)


def write_parquet(run: Path, name: str, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_parquet(run / f"{name}.parquet", index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--parts", default="1,2,3")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-dir", default=None, help="resume / extend this run folder")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args(argv)

    from analysis.internals.estimate import report

    cfg = load_config(args.config, args.dry_run)
    base_model = mcfg_name(load_config(args.config, False))  # the real model, also in --dry-run
    setup_prompts(cfg, args.dry_run)
    print(report(cfg), flush=True)
    if args.estimate_only:
        return 0
    parts = parse_parts(args.parts)

    from analysis.internals.parts import BudgetExceeded, Study
    from core import registry

    if args.run_dir:
        run = Path(args.run_dir)
        run = run if run.is_absolute() else ROOT / run
        run.mkdir(parents=True, exist_ok=True)
    else:
        run = make_run_dir(args.output_dir or cfg["output_dir"], cfg["name"])
    print(f"run folder: {run}\nparts: {', '.join(parts)}", flush=True)

    registry.discover()
    sec_name, excl_file = secret_cfg(cfg)
    secrets = registry.get("secret", sec_name)()
    codes = secrets.secrets()  # hash-checked against secret_types/hashes.json
    exclude: set[str] = set()
    if excl_file:
        from analysis.internals.fsa_codes import read_exclude, sha

        exclude = read_exclude(excl_file)
        print(f"exclude: {len(exclude)} codes from {excl_file} (sha256 {sha(sorted(exclude))[:12]})", flush=True)
    adapter = check_adapter(cfg, base_model, exclude, args.dry_run)
    hf = build_model(cfg, args.device)
    t0 = time.time()
    study = Study(cfg, run, hf, codes, args.max_hours, exclude=exclude)
    if excl_file:
        ne = study.n_excluded
        print(f"exclude: {ne['all']} of the {len(codes)} harness codes are excluded ({ne['train']} in the train "
              f"range, {ne['test']} in the test range); they are never drawn", flush=True)  # fmt: skip
    n_slide = study.h.layer_types.count("sliding")
    print(f"model: {mcfg_name(cfg)}, layer family: {study.h.family} ({FAMILY_NOTE[study.h.family]}); "
          f"{study.h.n_layers} layers ({n_slide} sliding, window {study.h.sliding_window}), "
          f"{study.h.n_heads} query heads, logit softcap {study.h.softcap}", flush=True)  # fmt: skip
    tok_check = study.check_tokenisation()
    print(f"tokenisation: every code -> {tok_check['span_tokens']} tokens at {tok_check['digit_pos']} "
          f"({tok_check['n_codes']} codes x {tok_check['prompts_checked']} prompts)", flush=True)  # fmt: skip
    prompts_data = cfg.pop("_prompts_data", None)
    if "bench" in cfg:  # keep the exact prompts with the run
        from analysis.internals.bench_prompts import write_prompts

        pp = Path(cfg["prompts"])
        src = prompts_data or yaml.safe_load((pp if pp.is_absolute() else ROOT / pp).read_text(encoding="utf-8"))
        kept = run / "prompts_bench.yaml"
        if kept.exists() and yaml.safe_load(kept.read_text(encoding="utf-8")) != src:
            raise SystemExit(f"{cfg['prompts']} differs from the prompts this run folder started with ({kept}); "
                             "use a new run folder or rebuild with the same seed")  # fmt: skip
        write_prompts(src, kept)
        cfg["prompts"] = str(kept)
        print(f"prompts: {cfg['prompts']} (" + ", ".join(f"{c} {len(v)}" for c, v in study.prompts["conds"].items())
              + f", D_no_secret {len(study.prompts['D'])})", flush=True)  # fmt: skip
    if study.replies:
        study.check_never_excluded()
        from analysis.internals.leaks import format_table

        print("leak table (cached replies):\n" + format_table(study.leak_report()), flush=True)
    extra: dict[str, Any] = {}
    if excl_file:
        extra["exclude_codes"] = {"file": str(excl_file), "n": len(exclude), "sha256": sha(sorted(exclude)),
                                  "n_excluded": study.n_excluded}  # fmt: skip
    if adapter:
        extra["adapter_info"] = adapter
    meta = {**cfg, **extra, "parts_requested": parts, "secret_sha256": secrets.sha256(), "tokenisation": tok_check,
            "layer_family": study.h.family, "layer_types": study.h.layer_types,
            "sliding_window": study.h.sliding_window,
            "n_layers": study.h.n_layers, "n_heads": study.h.n_heads}  # fmt: skip
    (run / "config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")
    gc = git_commit()
    (run / "git_commit.txt").write_text("\n".join(f"{k}: {v}" for k, v in gc.items()) + "\n", encoding="utf-8")

    try:
        want3 = {"3a", "3b"} & set(parts)
        if "1a" in parts or want3:
            n_codes = int(cfg["part1" if "1a" in parts else "part3"]["n_codes"])
            if want3 and "1a" in parts and int(cfg["part3"]["n_codes"]) != n_codes:
                raise SystemExit("part1.n_codes and part3.n_codes differ: run 1a and 3 separately")
            rows = study.sweep(n_codes, hidden="1a" in parts, components="3a" in parts, neurons="3b" in parts)
            if "1a" in parts:
                write_parquet(run, "hidden_sensitivity", rows["hidden"])
                write_parquet(run, "jsd", rows["jsd"])
            if "3a" in parts:
                write_parquet(run, "component_sensitivity", rows["components"])
            if "3b" in parts:
                write_parquet(run, "top_neurons", rows["neurons"])
                import pandas as pd

                pd.DataFrame(rows["neurons"]).to_csv(run / "top_neurons.csv", index=False)
            print(f"done: {', '.join(p for p in ('1a', '3a', '3b') if p in parts)} ({time.time() - t0:.0f} s)", flush=True)
        if "1b" in parts:
            write_parquet(run, "probe_accuracy", study.part1b())
            print(f"done: 1b ({time.time() - t0:.0f} s)", flush=True)
        if "1c" in parts:
            write_parquet(run, "logit_lens", study.part1c())
            print(f"done: 1c ({time.time() - t0:.0f} s)", flush=True)
        if "2" in parts:
            write_parquet(run, "attention_summary", study.part2())
            print(f"done: 2 ({time.time() - t0:.0f} s)", flush=True)
    except BudgetExceeded as e:
        print(str(e), file=sys.stderr)
        return 3

    from analysis.internals.leaks import format_table
    from analysis.internals.plots import plot_all

    n_checked = study.check_never_excluded()
    print(f"leak table ({n_checked} replies; reply_leaks.csv, leak_rate.csv):\n" + format_table(study.leak_report()),
          flush=True)  # fmt: skip
    made = plot_all(run)
    print("plots/tables: " + ", ".join(made))
    print(f"finished in {time.time() - t0:.0f} s -> {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
