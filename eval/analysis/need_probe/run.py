"""Entry point for analysis/need_probe: features (GPU, forward passes only) then probes (CPU).

    python -m analysis.need_probe.build --out outputs/need_probe/items.jsonl           # once, CPU
    python -m analysis.need_probe.run --config analysis/need_probe/config.yaml --run-dir outputs/need_probe_gemma3_12b
    python -m analysis.need_probe.run --config ... --run-dir DIR --stage probes        # probes only (CPU)
    python -m analysis.need_probe.run --config ... --dry-run                           # tiny model, fixtures
    python -m analysis.need_probe.run --config ... --estimate-only

Status: ✅ implemented + tested (--dry-run on tiny Gemma-3 and tiny Qwen2, tests/test_need_probe.py);
🟡 on gemma3-12b-it, gemma3-4b-it and qwen2.5-7b-instruct (not yet run on a GPU).

--run-dir resumes: finished feature batches (cache/) and assembled features (features/) are
reused. --max-hours stops cleanly with exit code 3; rerun the same command to continue.
This is analysis, not a defence: nothing here is registered with the harness registry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from core.io import ROOT, git_commit, make_run_dir


def estimate(cfg: dict[str, Any], items: list[dict[str, Any]]) -> str:
    from analysis.need_probe.build import MODEL_SHAPES
    from analysis.need_probe.probes import n_fits_per_job

    c, pc = cfg["cost"], cfg["probe"]
    name = cfg["model"]["name"]
    L, H = MODEL_SHAPES.get(name, (48, 4096))
    n_cond, n_pos = len(cfg.get("conditions", [0, 0])), len(cfg.get("positions", [0, 0]))
    n = len(items)
    tokens = n_cond * sum((len(it["system_template"]) + len(it["user"])) / float(c["chars_per_token"])
                          + float(c["template_tokens"]) for it in items)  # fmt: skip
    gpu_h = (float(c["load_s"]) + tokens / float(c["tokens_per_s"])) / 3600
    n_jobs = int(pc.get("n_jobs") or os.environ.get("SLURM_CPUS_PER_TASK", 0) or 8)
    fits = L * n_pos * n_fits_per_job(pc, n_cond)
    cpu_h = fits * float(c["sec_per_fit"]) / n_jobs / 3600
    feat_gb = n_cond * n_pos * L * n * H * 4 / 1e9
    layer_gb = n_cond * n * H * 4 / 1e9  # one probe job holds one layer of both conditions
    ram_gb = n_jobs * 4 * layer_gb + 4.0  # x4: copies during preprocessing; +4 GB process overhead
    tot = gpu_h + cpu_h
    lim = float(c["job_limit_h"])
    return "\n".join([
        f"Cost estimate ({name}: {L} layers x {H}; one A40; x3 = safety margin)",
        f"  items {n}, conditions {n_cond}, positions {n_pos}: ~{tokens:,.0f} prompt tokens",
        f"  features (GPU):  {gpu_h:.2f} h   (x3 {3 * gpu_h:.2f} h)   forward passes only",
        f"  probes (CPU):    {cpu_h:.2f} h   (x3 {3 * cpu_h:.2f} h)   {fits:,} logistic fits on {n_jobs} cores",
        f"  total:           {tot:.2f} h   (x3 {3 * tot:.2f} h) -> {'fits' if 3 * tot <= lim else 'DOES NOT FIT'} "
        f"one {lim:.0f} h job",
        f"  disk: features {feat_gb:.2f} GB (+ the same again in cache/ until assembled)",
        f"  RAM: ~{ram_gb:.1f} GB peak while probing ({n_jobs} jobs x one layer of both conditions); "
        f"{'fits' if ram_gb <= float(c['mem_gb']) else 'EXCEEDS'} --mem={float(c['mem_gb']):.0f}G",
    ])  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    from analysis.internals.run import FAMILY_NOTE, build_model, load_config
    from analysis.need_probe.build import build, read_items, summary, write_items

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None, help="resume / extend this run folder")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--items", default=None, help="override the config's items file")
    ap.add_argument("--stage", choices=["all", "features", "probes"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.dry_run)
    if args.dry_run:
        items = build(cfg["build"], dry_run=True)
    else:
        path = Path(args.items or cfg["items"])
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise SystemExit(f"{path} not found: run `python -m analysis.need_probe.build --out {args.items or cfg['items']}` first")
        items = read_items(path)
    print(estimate(cfg, items), flush=True)
    if args.estimate_only:
        return 0

    if args.run_dir:
        run = Path(args.run_dir)
        run = run if run.is_absolute() else ROOT / run
        run.mkdir(parents=True, exist_ok=True)
    else:
        run = make_run_dir(args.output_dir or cfg["output_dir"], cfg["name"])
    print(f"run folder: {run}", flush=True)
    write_items(items, run / "items.jsonl")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    gc = git_commit()
    (run / "git_commit.txt").write_text("\n".join(f"{k}: {v}" for k, v in gc.items()) + "\n", encoding="utf-8")
    s = summary(items)
    print(f"items: {s['n_items']} ({', '.join(f'{k} {v}' for k, v in s['by_split_group'].items())})", flush=True)
    t0 = time.time()

    meta_path = run / "features" / "meta.json"
    if args.stage in ("all", "features"):
        from analysis.need_probe.features import BudgetExceeded, Extractor, position_sanity

        hf = build_model(cfg, args.device)
        ex = Extractor(cfg, run, hf, items, args.max_hours)
        h = ex.h
        print(f"model: {cfg['model']['name']}, layer family: {h.family} ({FAMILY_NOTE[h.family]}); "
              f"{h.n_layers} layers ({h.layer_types.count('sliding')} sliding), hidden {h.hidden_size}", flush=True)  # fmt: skip
        for line in position_sanity(hf, items, ex.conditions):  # raises PositionError on a bad item
            print(line, flush=True)
        try:
            meta = ex.run()
        except BudgetExceeded as e:
            print(str(e), file=sys.stderr)
            return 3
        del ex, hf, h
        _free_gpu()
        print(f"done: features ({time.time() - t0:.0f} s)", flush=True)
    if args.stage in ("all", "probes"):
        if not meta_path.exists():
            raise SystemExit(f"{meta_path} missing: run the features stage first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["item_ids"] != [it["id"] for it in items]:
            raise SystemExit("features were extracted for a different item set")
        from analysis.need_probe.features import drop_cache
        from analysis.need_probe.plots import plot_all
        from analysis.need_probe.probes import run_probes

        res = run_probes(cfg, run, items, meta)
        print(f"done: probes ({time.time() - t0:.0f} s)", flush=True)
        for k, v in res["best_layer"].items():
            print(f"best layer {k}: {v}", flush=True)
        print(f"baselines: {res['baseline']}", flush=True)
        print(f"massive-dim handling, max |delta test AUROC| over layers: {res['massive_dims']}", flush=True)
        print("plots: " + ", ".join(plot_all(run)), flush=True)
        if not cfg.get("keep_cache", False):
            drop_cache(run)
    print(f"finished in {time.time() - t0:.0f} s -> {run}")
    return 0


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
