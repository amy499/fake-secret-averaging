"""Export the runs used in the report to results/ (committed): aggregate metrics only.

For every run in the list file (configs/training/export_runs.yaml) copies an allowlist of files:
evaluation runs -> results/runs/<run_id>/{config.yaml, metrics.json, diagnostics.json,
likelihood_attack*.json}; utility runs -> results/utility/<run_id>/{config.yaml,
utility_metrics.json}. Nothing else is ever copied (no samples.jsonl, train_samples.jsonl,
items.jsonl, reanalysis.json, adapters or logs). Every file is validated against its schema
first; one missing or invalid run aborts the export before anything is written.

Absolute home-directory paths (/home/<user>/..., /scratch/<user>/..., C:\\Users\\<user>\\...)
in a copied config.yaml are rewritten to repo-relative paths (the part after this checkout's root,
or after experiments/fake_secret_average/, the folder the committed results/fsa_v1 runs were
trained in; else <home>/...); export_manifest.json records which keys were
rewritten, plus source and copy sha256 of every file. A JSON file containing such a path stops
the export instead of being edited. Then scripts.collect_results writes results/summary.csv/.md
for exactly the exported runs, and results/README.md is regenerated from the template below.
outputs/ is only read. Imports nothing from the ml group.

    uv run python -m scripts.export_results --dry-run
    uv run python -m scripts.export_results [--prune]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fsa_runtime.settings import ROOT
from fsa.contract.schemas import (
    RUN_ID_PATTERN,
    EstimateCI,
    RunConfig,
    RunMetrics,
    UtilityEvalMetrics,
)
from fsa.eval.run_extras import (
    adapter_run,
    choose_attack,
    prompt_label,
    read_attack_files,
    summarize_attack,
)
from scripts import collect_results
from scripts._cli import utf8_stdio

DEFAULT_LIST = ROOT / "configs" / "training" / "export_runs.yaml"
DEFAULT_RUNS = ROOT / "outputs" / "runs"
DEFAULT_UTILITY = ROOT / "outputs" / "utility"
DEFAULT_OUT = ROOT / "results"
EVAL_FILES = ("config.yaml", "metrics.json", "diagnostics.json")  # + likelihood_attack*.json
UTILITY_FILES = ("config.yaml", "utility_metrics.json")
MANIFEST = "export_manifest.json"
README = "README.md"
EXPERIMENT_DIR = "experiments/fake_secret_average"  # legacy layout of the exported results/fsa_v1 runs
# Clusters whose homes have two levels (/home/<group>/<user>): FSA_EXPORT_HOME_GROUPS="group1 group2".
HOME_GROUPS = tuple(os.environ.get("FSA_EXPORT_HOME_GROUPS", "").split())

# A home directory and everything after it, up to whitespace or a quote.
HOME_RE = re.compile(
    r"(?:/home/[^/\s\"']+|/scratch/[^/\s\"']+|/Users/[^/\s\"']+|"
    r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+)(?:[\\/][^\s\"']*)?"
)
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)


class ExportError(RuntimeError):
    """The export cannot proceed; nothing has been written."""


# --------------------------------------------------------------------------- paths


def relativise(path: str) -> str:
    """One absolute home path -> repo-relative (after this checkout's root, or after the legacy
    experiments/fake_secret_average/) or <home>/<rest>."""
    p = path.replace("\\", "/")
    root = ROOT.as_posix().rstrip("/") + "/"
    if p.startswith(root):
        return p[len(root):] or "."
    marker = EXPERIMENT_DIR + "/"
    if marker in p:
        return p.split(marker, 1)[1] or "."
    if p.rstrip("/").endswith(EXPERIMENT_DIR):
        return "."
    groups = "".join(f"/home/{re.escape(g)}/[^/]+|" for g in HOME_GROUPS)  # two-level homes first
    m = re.match(
        rf"^(?:{groups}/home/[^/]+|/scratch/[^/]+|/Users/[^/]+|[A-Za-z]:/+Users/+[^/]+)/?(.*)$",
        p,
    )
    return f"<home>/{m.group(1)}" if m else "<home>"


def rewrite_home_paths(obj: Any, key: str = "") -> tuple[Any, list[dict[str, str]]]:
    """(copy of obj with home paths in string values rewritten, [{key, new}]). The old values
    are not returned: they would put the home path into the committed manifest."""
    if isinstance(obj, dict):
        out, changes = {}, []
        for k, v in obj.items():
            out[k], c = rewrite_home_paths(v, f"{key}.{k}" if key else str(k))
            changes += c
        return out, changes
    if isinstance(obj, list):
        out, changes = [], []
        for i, v in enumerate(obj):
            nv, c = rewrite_home_paths(v, f"{key}[{i}]")
            out.append(nv)
            changes += c
        return out, changes
    if isinstance(obj, str) and HOME_RE.search(obj):
        new = HOME_RE.sub(lambda m: relativise(m.group(0)), obj)
        return new, [{"key": key, "new": new}]
    return obj, []


# --------------------------------------------------------------------------- planning


@dataclass
class PlannedFile:
    name: str
    source: Path
    content: bytes
    rewritten: list[dict[str, str]] = field(default_factory=list)


@dataclass
class PlannedRun:
    kind: str  # "runs" | "utility"
    run_id: str
    label: str
    source: Path
    files: list[PlannedFile]
    info: dict[str, Any] = field(default_factory=dict)  # for the README table


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_list(path: Path) -> dict[str, list[dict[str, str]]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, list[dict[str, str]]] = {}
    for section in ("runs", "utility"):
        entries = raw.get(section) or []
        seen = set()
        for e in entries:
            if not isinstance(e, dict) or not _RUN_ID_RE.fullmatch(str(e.get("id", ""))):
                raise ExportError(f"{path.name}: {section}: bad entry {e!r} (need id: <run_id>)")
            if e["id"] in seen:
                raise ExportError(f"{path.name}: {section}: {e['id']} listed twice")
            seen.add(e["id"])
        out[section] = [
            {"id": str(e["id"]), "label": str(e.get("label") or e["id"])} for e in entries
        ]
    return out


def _config_file(src: Path) -> PlannedFile:
    raw = src.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ExportError(f"{src}: config.yaml is not a mapping")
    new, changes = rewrite_home_paths(data)
    content = raw
    if changes:
        content = yaml.safe_dump(new, sort_keys=False, allow_unicode=True).encode("utf-8")
    return PlannedFile("config.yaml", src, content, changes)


def _json_file(src: Path) -> PlannedFile:
    raw = src.read_bytes()
    text = raw.decode("utf-8")
    json.loads(text)
    m = HOME_RE.search(text)
    if m:
        raise ExportError(
            f"{src} contains an absolute home-directory path ({relativise(m.group(0))!r} after "
            "rewriting); JSON results are never edited, fix the source or drop the file"
        )
    return PlannedFile(src.name, src, raw)


def plan_eval_run(runs_dir: Path, run_id: str, label: str) -> PlannedRun:
    src = runs_dir / run_id
    for name in ("config.yaml", "metrics.json"):
        if not (src / name).is_file():
            raise ExportError(f"{run_id}: {name} missing in {src}")
    cfg = _config_file(src / "config.yaml")
    config = yaml.safe_load(cfg.content.decode("utf-8"))
    RunConfig.model_validate(config)
    metrics_file = _json_file(src / "metrics.json")
    metrics = RunMetrics.model_validate_json(metrics_file.content)
    if metrics.run_id != run_id:
        raise ExportError(f"{run_id}: metrics.json run_id is {metrics.run_id!r}")
    if metrics.run_type != "evaluation":
        raise ExportError(f"{run_id}: not an evaluation run ({metrics.run_type})")
    files = [cfg, metrics_file]
    if (src / "diagnostics.json").is_file():
        files.append(_json_file(src / "diagnostics.json"))
    attacks = read_attack_files(src)
    for a in attacks:
        if a.attack is None:
            raise ExportError(f"{run_id}/{a.path.name}: invalid ({a.error})")
        if a.attack.run_id != run_id:
            raise ExportError(f"{run_id}/{a.path.name}: run_id is {a.attack.run_id!r}")
        files.append(_json_file(a.path))
    summaries = [summarize_attack(a, config) for a in attacks]
    leak = metrics.leakage
    sig = leak.significance if leak else None
    info = {
        "prompt": prompt_label(config),
        "method": config.get("method"),
        "adapter_run": adapter_run(config),
        "threat_models": sorted({s.threat_model for s in summaries}),
        "heur_bits": leak.bits_leaked_lower_bound if leak else None,
        "heur_ci": (sig.bits_ci.low, sig.bits_ci.high) if sig else None,
        "heur_exact": leak.attacker.exact_match.value if leak else None,
        "attacks": {
            tm: choose_attack([s for s in summaries if s.threat_model == tm])
            for tm in ("grey_box", "public_weights", "surrogate")
        },
    }
    return PlannedRun("runs", run_id, label, src, files, info)


def plan_utility_run(utility_dir: Path, run_id: str, label: str) -> PlannedRun:
    src = utility_dir / run_id
    for name in UTILITY_FILES:
        if not (src / name).is_file():
            raise ExportError(f"utility {run_id}: {name} missing in {src}")
    cfg = _config_file(src / "config.yaml")
    m_file = _json_file(src / "utility_metrics.json")
    metrics = UtilityEvalMetrics.model_validate_json(m_file.content)
    if metrics.run_id != run_id:
        raise ExportError(f"utility {run_id}: utility_metrics.json run_id is {metrics.run_id!r}")
    info = {"method": metrics.method, "adapter_run": metrics.adapter_run, "metrics": metrics}
    return PlannedRun("utility", run_id, label, src, [cfg, m_file], info)


def plan(list_path: Path, runs_dir: Path, utility_dir: Path) -> list[PlannedRun]:
    """Read and validate everything first; raises ExportError listing every problem."""
    lst = load_list(list_path)
    planned, errors = [], []
    for kind, fn, base in (
        ("runs", plan_eval_run, runs_dir),
        ("utility", plan_utility_run, utility_dir),
    ):
        for e in lst[kind]:
            try:
                planned.append(fn(base, e["id"], e["label"]))
            except ExportError as err:
                errors.append(str(err))
            except (OSError, ValueError, yaml.YAMLError) as err:  # pydantic: ValueError
                errors.append(f"{e['id']}: {type(err).__name__}: {str(err).splitlines()[0]}")
    if errors:
        raise ExportError("nothing written:\n  " + "\n  ".join(errors))
    return planned


# --------------------------------------------------------------------------- writing


def check_out_dir(out_dir: Path, sources: Sequence[Path]) -> None:
    out = out_dir.resolve()
    for protected in (*sources, ROOT / "outputs"):
        p = Path(protected).resolve()
        if out == p or out.is_relative_to(p) or p.is_relative_to(out):
            raise ExportError(f"--out-dir {out_dir} overlaps {protected}; refusing to write there")


def stale_runs(out_dir: Path, planned: Sequence[PlannedRun]) -> list[Path]:
    wanted = {(p.kind, p.run_id) for p in planned}
    stale = []
    for kind in ("runs", "utility"):
        d = out_dir / kind
        if d.is_dir():
            stale += [
                x
                for x in sorted(d.iterdir())
                if x.is_dir() and _RUN_ID_RE.fullmatch(x.name) and (kind, x.name) not in wanted
            ]
    return stale


def write(out_dir: Path, planned: Sequence[PlannedRun], prune: bool = False) -> list[Path]:
    """Replace each planned run folder; remove stale ones only with prune. Returns stale ones."""
    stale = stale_runs(out_dir, planned)
    for p in planned:
        dest = out_dir / p.kind / p.run_id
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for f in p.files:
            (dest / f.name).write_bytes(f.content)
    if prune:
        for d in stale:
            shutil.rmtree(d)
    return stale


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(ROOT.resolve()).as_posix() if p.is_relative_to(ROOT.resolve()) else p.name


def manifest(planned: Sequence[PlannedRun], list_path: Path, now: datetime) -> dict[str, Any]:
    return {
        "exported_at": now.isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "list_file": _rel(list_path),
        "script": "scripts/export_results.py",
        "runs": [
            {
                "kind": p.kind,
                "run_id": p.run_id,
                "label": p.label,
                "source": _rel(p.source),
                "files": [
                    {
                        "name": f.name,
                        "source_sha256": _sha(f.source.read_bytes()),
                        "sha256": _sha(f.content),
                        "home_paths_relativised": f.rewritten,
                    }
                    for f in p.files
                ],
            }
            for p in planned
        ],
    }


# --------------------------------------------------------------------------- README

DASH = "—"
DAGGER_NOTE = (
    "† calibration temperature hit its search bound (0.01 or 10000), so the calibrated posterior "
    "is degenerate; raw value shown."
)
THREAT_LABEL = {
    "grey_box": "grey-box",
    "public_weights": "public weights",
    "surrogate": "surrogate",
}


def _bits(est: float | None, ci: tuple | None) -> str:
    if est is None:
        return DASH
    if not ci or ci[0] is None or ci[1] is None:
        return f"{est:.3f}"
    return f"{est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _attack_cell(s: Any) -> str:
    if s is None:
        return DASH
    tag = "cal" if s.posterior == "calibrated" else "*raw†*" if s.tau_at_search_bound else "*raw*"
    return f"{_bits(s.bits, (s.bits_ci_low, s.bits_ci_high))} {tag}"


def _est(ci: EstimateCI | None, pct: bool = True) -> str:
    if ci is None:
        return DASH
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.4f}")
    unit = "%" if pct else ""
    if ci.low is None or ci.high is None:
        return f"{f(ci.estimate)}{unit}"
    return f"{f(ci.estimate)}{unit} [{f(ci.low)}, {f(ci.high)}]"


def _utility_cell(m: UtilityEvalMetrics, bench: str, metric: str, cond: str) -> str:
    r = m.benchmarks.get(bench)
    c = r.conditions.get(cond) if r else None
    if c is None:
        return DASH
    if bench == "kl_to_base" and m.adapter is None and c.metrics[metric].estimate == 0:
        return "0 (base)"
    return _est(c.metrics[metric], pct=bench != "kl_to_base")


def _both(m: UtilityEvalMetrics, bench: str, metric: str) -> str:
    """'secret / no_secret' cells of one utility metric."""
    return f"{_utility_cell(m, bench, metric, 'secret')} / {_utility_cell(m, bench, metric, 'no_secret')}"


def _cell(x: Any) -> str:
    return str(x).replace("|", "\\|") if x not in (None, "") else DASH


def render_readme(planned: Sequence[PlannedRun], now: datetime) -> str:
    evals = [p for p in planned if p.kind == "runs"]
    utils = [p for p in planned if p.kind == "utility"]
    lines = [
        "# Results (FSA v1): fake-secret average",
        "",
        (
            "Exported aggregate metrics for every run used in the report. **Generated by "
            "`scripts/export_results.py`; do not edit by hand** (rerun the export instead). "
            f"Last export: {now.isoformat(timespec='seconds')}."
        ),
        "",
        "## What is here",
        "",
        "| Path | Contents |",
        "|---|---|",
        (
            "| `runs/<run_id>/` | `config.yaml`, `metrics.json` (heuristic attacker, significance), "
            "`diagnostics.json` (per-secret observed vs expected counts), `likelihood_attack*.json` "
            "(likelihood attacker: scores, posterior metrics, calibration) |"
        ),
        (
            "| `utility/<run_id>/` | `config.yaml`, `utility_metrics.json` (MMLU, IFEval, verification, "
            "KL to base, with 95% CIs) |"
        ),
        (
            "| `summary.csv`, `summary.md` | One row per (run, likelihood file), from "
            "`scripts.collect_results` over `runs/` |"
        ),
        (
            "| `export_manifest.json` | Source path, source and copy sha256 of every file, and which "
            "config keys had home-directory paths made repo-relative |"
        ),
        "",
        (
            "**Not here, on purpose:** raw generations (`samples.jsonl`, `train_samples.jsonl`, "
            "utility `items.jsonl`), re-analyses, Slurm logs and adapter weights. They are large, "
            "contain model outputs rather than results, and stay in `outputs/` on the Slurm cluster (following the "
            "team convention: aggregate metrics in git, raw data outside). The likelihood files keep "
            "their full score matrices, so `scripts.likelihood_attack --from-scores` can re-analyse "
            "them without a GPU."
        ),
        "",
        "## Runs",
        "",
        (
            "Gemma-3 12B, 50 held-out secrets each. ≥ bits = bits leaked (lower bound) = log2(100) − "
            "the attacker's mean cross-entropy on the true secret, clipped at 0, with its 95% "
            "bootstrap CI (max 6.644). Heuristic = the run's own attacker (metrics.json). Likelihood "
            "columns: one file per threat model, the calibrated one first, then the newest (same rule "
            "as the dashboard); files with identical scores count once. `summary.md` lists every file."
        ),
        "",
        (
            "| Run | run_id | Prompt | Method | Adapter (training run) | Threat models | "
            "Heuristic ≥ bits [95% CI] | Heuristic exact | Grey-box ≥ bits [95% CI] | "
            "Public weights ≥ bits [95% CI] |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in evals:
        i = p.info
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(p.label),
                    f"`{p.run_id}`",
                    _cell(i["prompt"]),
                    _cell(i["method"]),
                    _cell(i["adapter_run"]),
                    ", ".join(THREAT_LABEL[t] for t in i["threat_models"]) or DASH,
                    _bits(i["heur_bits"], i["heur_ci"]),
                    DASH if i["heur_exact"] is None else f"{100 * i['heur_exact']:.1f}%",
                    _attack_cell(i["attacks"]["grey_box"]),
                    _attack_cell(i["attacks"]["public_weights"]),
                ]
            )
            + " |"
        )
    shown = [a for p in evals for a in p.info["attacks"].values() if a is not None]
    if any(a.posterior == "raw" and a.tau_at_search_bound for a in shown):
        lines += ["", DAGGER_NOTE]
    if utils:
        lines += [
            "",
            "## Utility",
            "",
            (
                "Each cell: secret / no secret condition, estimate [95% bootstrap CI]. KL to base in "
                "nats per token (0 by definition without an adapter)."
            ),
            "",
            (
                "| Run | run_id | Method | Adapter | MMLU acc | IFEval prompt strict | "
                "IFEval instr strict | Verification acc | KL to base |"
            ),
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for p in utils:
            m = p.info["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(p.label),
                        f"`{p.run_id}`",
                        _cell(m.method),
                        _cell(m.adapter_run),
                        _both(m, "mmlu", "accuracy"),
                        _both(m, "ifeval", "prompt_level_strict"),
                        _both(m, "ifeval", "instruction_level_strict"),
                        _utility_cell(m, "verification", "accuracy", "secret"),
                        _both(m, "kl_to_base", "kl_per_token"),
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "## Calibrated vs raw",
        "",
        (
            "- **raw** (*italic*): posterior softmax(scores) over the 100 candidates. Scores are summed "
            "over ~1000 reply tokens, so it is far too confident and its bits are often meaningless "
            "(6.644 = certain)."
        ),
        (
            "- **cal**: softmax(scores / tau), one temperature tau fitted on the train secrets "
            "(`--calibrate`) and applied unchanged to the held-out ones. tau never reorders candidates."
        ),
        (
            "- **raw†**: the file has a calibration, but tau hit its search bound (0.01 or 10000; "
            "`tau_at_search_bound: true`). The calibrated posterior is then degenerate (6.644 bits at "
            "0.01, a uniform 0 at 10000), so the raw bits, CI, exact, top-5 and p are shown instead. "
            "The same rule picks the shown file and is used by the dashboard and `summary.md`."
        ),
        (
            "- Threat models: grey-box = the deployed model (base + the run's adapter); public weights "
            "= the base model only, without the adapter."
        ),
        "",
        "## Regenerate",
        "",
        (
            "From the repository root, after copying the runs down from the cluster into "
            "`outputs/`:"
        ),
        "",
        "```powershell",
        "uv run python -m scripts.export_results --dry-run   # check the list resolves",
        "uv run python -m scripts.export_results             # copy, then collect_results",
        "```",
        "",
        (
            "The run list is `configs/training/export_runs.yaml`. `--prune` removes exported runs no longer "
            "listed. `export_results` runs `scripts.collect_results --runs-dir results/runs --out-dir "
            "results` itself."
        ),
        "",
        "## Data credits",
        "",
        (
            "- **UnpredictaBench** (Hugging Face dataset `UnpredictaBench/UnpredictaBench`, CC-BY-4.0): "
            "the four prompts of the `unpredictabench_2digit` set, used verbatim; source lines and uids "
            "in `configs/training/prompts.yaml`. Please cite the dataset when using these results."
        ),
        "- **MMLU** (`cais/mmlu`, MIT): 300-question subject-stratified subset.",
        (
            "- **IFEval** (`google/IFEval`, Apache-2.0): 150-prompt subset, scored with the "
            "google-research `instruction_following_eval` checker (Apache-2.0), vendored with an NLTK "
            "`punkt_tab` patch."
        ),
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def export(
    list_path: Path = DEFAULT_LIST,
    runs_dir: Path = DEFAULT_RUNS,
    utility_dir: Path = DEFAULT_UTILITY,
    out_dir: Path = DEFAULT_OUT,
    *,
    prune: bool = False,
    dry_run: bool = False,
    summary: Callable[[Path], int] | None = None,
    log: Callable[[str], None] = print,
    now: datetime | None = None,
) -> list[PlannedRun]:
    now = now or datetime.now()
    check_out_dir(out_dir, [runs_dir, utility_dir])
    planned = plan(list_path, runs_dir, utility_dir)
    for p in planned:
        n_rw = sum(bool(f.rewritten) for f in p.files)
        log(
            f"{p.kind:7s} {p.run_id}  {len(p.files)} files"
            + (f", {n_rw} with home paths made relative" if n_rw else "")
            + f"  ({p.label})"
        )
    stale = stale_runs(out_dir, planned)
    for d in stale:
        log(
            f"{'removing' if prune and not dry_run else 'not listed'}: {d.relative_to(out_dir).as_posix()}"
        )
    if dry_run:
        log("dry run: nothing written")
        return planned
    write(out_dir, planned, prune=prune)
    (out_dir / MANIFEST).write_text(
        json.dumps(manifest(planned, list_path, now), indent=2) + "\n", encoding="utf-8"
    )
    run_summary = summary or (
        lambda out: collect_results.main(["--runs-dir", str(out / "runs"), "--out-dir", str(out)])
    )
    if run_summary(out_dir) != 0:
        raise ExportError("collect_results failed")
    (out_dir / README).write_bytes(render_readme(planned, now).encode("utf-8"))
    log(f"wrote {out_dir / MANIFEST} and {out_dir / README}")
    return planned


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", type=Path, default=DEFAULT_LIST, help="run list (YAML)")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--utility-dir", type=Path, default=DEFAULT_UTILITY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prune", action="store_true", help="remove exported runs not listed")
    parser.add_argument("--dry-run", action="store_true", help="validate and show; write nothing")
    args = parser.parse_args(argv)
    try:
        export(
            _resolve(args.list),
            _resolve(args.runs_dir),
            _resolve(args.utility_dir),
            _resolve(args.out_dir),
            prune=args.prune,
            dry_run=args.dry_run,
        )
    except ExportError as e:
        print(f"ERROR: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
