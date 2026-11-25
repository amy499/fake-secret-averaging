"""Team-protocol runner (docs/eval_protocol.md): one greedy reply per item, the protocol's
token cap, a fixed item order and batch size, pinned data, the original model and the model with
the technique on identical settings.

Status: ✅ implemented + tested (tests/test_protocol.py: dry run of every suite on the dummy
model, smoke run on the tiny random Gemma 3 / Qwen2, original vs technique pairing). 🟡 for real
models: never run on Gemma-3-12B-it / Qwen2.5-7B-Instruct from here yet.

    python run.py --approach <name> --model gemma|qwen --suite core|second|synthetic|secret_in_context|all
                  --variant original|technique [--benchmarks a,b] [--limit N] [--dry-run]

Approach spec: configs/eval/<name>.approach.yaml (repository root), or experiments/<name>/approach.yaml
for an approach written from experiments/_template/. Keys:
    name, owner
    defence: {name: <registered defence>, ...}       # the technique variant
    models:                                          # per model key (gemma, qwen)
      gemma: {model: {...}, defence: {...}}           # model: shared by both variants (revision,
                                                      # batch_size, attn_implementation);
                                                      # defence: technique only (e.g. adapter)
    synthetic: {used_secrets: private/used_secrets.txt}   # see held_out_filter
    secret_in_context: {secrets: private/held_out_secrets.txt}   # see held_out_secrets
    original_from: <approach>    # optional: this approach's technique is compared with THAT
                                 # approach's original runs (check_original_from: identical original
                                 # model options and secret-in-context file); recorded in summary.json
The original variant always uses defence `none` with the same model options.

Output: <output_dir>/<approach>/<model>/<variant>/<benchmark>/ with
    items.jsonl    one row per item (key, index, order, prompt hashes, reply, n_new_tokens,
                   finished, scores, meta), in item order
    per_unit.json  per-unit values of every metric (paired comparisons in stats/team_table.py)
    summary.json   everything needed to judge and pair the run (see `SUMMARY_KEYS`)
    + benchmark extras (confaide_vista_check.csv, mmlu_selection.json)
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core import registry
from core.interfaces import GenConfig, Item, ModelWrapper, SecretContext
from core.io import ROOT, git_commit, write_json, write_jsonl

SCHEMA_VERSION = 2  # 2: summary.environment (cluster, host, GPU, library versions, pin check)
PROTOCOL_DOC = "docs/eval_protocol.md"
PROTOCOL_BATCH_SIZE = 16
MODEL_KEYS = {"gemma": "gemma3-12b-it", "qwen": "qwen2.5-7b-instruct"}
MODEL_DEFAULTS = {"device": "cuda", "dtype": "bfloat16", "batch_size": PROTOCOL_BATCH_SIZE}
SUITES: dict[str, tuple[str, ...]] = {
    "core": ("password_eval_team", "confaide_t4_team", "mmlu_team", "ifeval_team"),
    "second": ("tt_extraction_team", "tt_hijacking_team", "tt_validity_team", "gsm8k_team"),
    "synthetic": ("hidden_code_team", "code_check_team"),
    # report only, both variants: the plain capability items with a held-out secret in context
    "secret_in_context": ("mmlu_team_sic", "ifeval_team_sic", "gsm8k_team_sic"),
}
SUITES["all"] = SUITES["core"] + SUITES["second"] + SUITES["synthetic"] + SUITES["secret_in_context"]
FIXTURE_SECRETS = ("fixture-secret-7391", "fixture-secret-2048", "fixture-secret-5560")  # dry / smoke runs only
DRY_RUN_LIMIT = 5
VARIANTS = ("original", "technique")
SUMMARY_KEYS = (
    "schema_version", "protocol_doc", "approach", "variant", "defence", "benchmark", "suite", "team_table",
    "model", "system_fold_check", "decoding", "batch_size", "n_items", "expected_n", "item_keys_sha256",
    "order_sha256", "prompts_sha256", "data_files", "versions", "git", "metrics", "primary", "truncation",
    "scorer_limits", "reference", "extras", "held_out_check", "settings_sha256", "protocol_compliant",
    "non_compliance", "dry_run", "fixtures", "limit", "started_utc", "wall_s", "environment", "defence_info",
    "sdpa_backend", "original_from",
)


class ProtocolError(RuntimeError):
    pass


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- approach spec


def approach_path(name: str) -> Path:
    from core.io import CONFIGS

    for p in (CONFIGS / "eval" / f"{name}.approach.yaml", ROOT / "experiments" / name / "approach.yaml"):
        if p.exists():
            return p
    raise ProtocolError(f"no approach spec for {name!r}: create experiments/{name}/approach.yaml "
                        f"(copy experiments/_template/)")  # fmt: skip


def load_approach(name: str, path: Path | None = None) -> dict[str, Any]:
    path = path or approach_path(name)
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    spec.setdefault("name", name)
    spec["_path"] = str(path)
    if not isinstance(spec.get("defence"), dict) or "name" not in spec["defence"]:
        raise ProtocolError(f"{path}: `defence: {{name: ...}}` is required")
    return spec


def variant_configs(spec: dict[str, Any], model_key: str, variant: str) -> tuple[dict, dict]:
    """(model cfg, defence cfg) for one variant. Model options are shared by both variants; the
    technique's defence options apply to the technique only."""
    if model_key not in MODEL_KEYS:
        raise ProtocolError(f"unknown model {model_key!r}; use one of {sorted(MODEL_KEYS)}")
    if variant not in VARIANTS:
        raise ProtocolError(f"unknown variant {variant!r}; use one of {VARIANTS}")
    per = (spec.get("models") or {}).get(model_key) or {}
    model_cfg = {"name": MODEL_KEYS[model_key], **MODEL_DEFAULTS, **(per.get("model") or {})}
    if variant == "original":
        return model_cfg, {"name": "none"}
    return model_cfg, {**spec["defence"], **(per.get("defence") or {})}


def check_original_from(spec: dict[str, Any]) -> dict[str, Any] | None:
    """An approach with `original_from: <base>` reuses <base>'s original runs as its baseline. That
    is valid only if its own original variant would be identical: the same model options for every
    model key (hence the same model, revision request, dtype, batch size; decoding, items, order and
    the SDPA backend are the benchmark's, the same for every approach) and the same secret-in-context
    file. Raises ProtocolError otherwise; returns the record written to summary.json."""
    base_name = spec.get("original_from")
    if not base_name:
        return None
    if base_name == spec.get("name"):
        raise ProtocolError(f"approach {spec.get('name')}: original_from names itself")
    base = load_approach(str(base_name))
    if base.get("original_from"):
        raise ProtocolError(f"original_from {base_name} itself borrows its original (from {base['original_from']}): not allowed")
    diffs = []
    for key in MODEL_KEYS:
        mine, theirs = variant_configs(spec, key, "original")[0], variant_configs(base, key, "original")[0]
        if mine != theirs:
            diffs.append(f"{key}: {mine} vs {theirs}")

    def sic_file(sp: dict[str, Any]) -> str | None:
        rel = (sp.get("secret_in_context") or {}).get("secrets")
        return str(_approach_file(sp, rel).resolve()) if rel else None

    if sic_file(spec) != sic_file(base):
        diffs.append(f"secret_in_context file {sic_file(spec)} vs {sic_file(base)}")
    if diffs:
        raise ProtocolError(f"approach {spec.get('name')}: original_from {base_name}, but its original variant would "
                            "differ: " + "; ".join(diffs))  # fmt: skip
    return {"approach": base_name, "spec": Path(base["_path"]).name,
            "checked": "original model options equal for every model key; same secret-in-context file",
            "original_model_options_sha256": sha256_json({k: variant_configs(base, k, "original")[0] for k in MODEL_KEYS})}  # fmt: skip


# --------------------------------------------------------------------------- one run


@dataclass
class Session:
    """A loaded model + defence for one (approach, model, variant), reused across benchmarks."""

    approach: dict[str, Any]
    model_key: str
    variant: str
    model_cfg: dict[str, Any]
    defence_cfg: dict[str, Any]
    base: ModelWrapper
    defence: Any
    model: ModelWrapper | None = None
    info: dict[str, Any] = field(default_factory=dict)
    fold: dict[str, Any] | None = None


def open_session(approach: dict[str, Any], model_key: str, variant: str, dry_run: bool = False,
                 device: str | None = None, model: ModelWrapper | None = None) -> Session:  # fmt: skip
    """Build the base model (dummy under dry_run, or the given instance) and the defence."""
    registry.discover()
    model_cfg, defence_cfg = variant_configs(approach, model_key, variant)
    if device:
        model_cfg["device"] = device
    if model is not None:
        base = model
    elif dry_run:
        base = registry.get("model", "dummy")({"replaces": model_cfg["name"]})
    else:
        base = registry.get("model", model_cfg["name"])(model_cfg)
    defence = registry.get("defence", defence_cfg["name"])(defence_cfg)
    return Session(approach, model_key, variant, model_cfg, defence_cfg, base, defence)


def _model_facts(s: Session) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not s.info:
        info = s.base.run_info() if hasattr(s.base, "run_info") else {"name": s.base.name, "class": type(s.base).__name__}
        s.info = {**info, "key": s.model_key, "registry_name": s.model_cfg["name"],
                  "batch_size": getattr(s.base, "batch_size", None)}  # fmt: skip
        s.fold = s.base.system_fold_check() if hasattr(s.base, "system_fold_check") else None
    return s.info, s.fold


def _rendered_sha(model: ModelWrapper, msgs: list[dict]) -> str | None:
    fn = getattr(model, "template", None)
    if fn is None:
        return None
    return hashlib.sha256(fn(msgs).encode("utf-8")).hexdigest()


def held_out_filter(approach: dict[str, Any], bench: Any, items: Sequence[Item]) -> tuple[list[Item], dict[str, Any]]:
    """Synthetic tasks: drop every item whose secret is listed in the approach's private
    used-secrets file (secrets used in training or model selection), so only held-out secrets are
    evaluated; both variants drop the same items. Matching is by value: digits_4 draws codes with
    replacement, so 21 of its 300 test codes also occur among the 800 train codes. Records only
    the file's sha256 and counts, never its contents."""
    if bench.team_table:
        return list(items), {"applies": False}
    got = _used_secrets(approach)
    if got is None:
        return list(items), {"applies": True, "declared": False, "note": "approach declares no used_secrets file"}
    path, data, used = got
    kept = [it for it in items if not (it.secret and it.secret in used)]
    if not kept:
        raise ProtocolError(f"every evaluated secret is listed in {path.name}: nothing held out to evaluate")
    return kept, {"applies": True, "declared": True, "file_sha256": hashlib.sha256(data).hexdigest(),
                  "n_used": len(used), "n_items_before": len(items), "n_excluded": len(items) - len(kept),
                  "overlap_after": 0}  # fmt: skip


def _approach_file(approach: dict[str, Any], rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else Path(approach["_path"]).parent / path


def _lines(data: bytes) -> list[str]:
    return [ln.strip() for ln in data.decode("utf-8").splitlines() if ln.strip()]


def _used_secrets(approach: dict[str, Any]) -> tuple[Path, bytes, set[str]] | None:
    """(path, bytes, set) of the approach's training / model-selection secrets, or None if undeclared."""
    rel = (approach.get("synthetic") or {}).get("used_secrets")
    if not rel:
        return None
    path = _approach_file(approach, rel)
    if not path.exists():
        raise ProtocolError(f"used_secrets file {path} is declared but missing")
    data = path.read_bytes()
    return path, data, set(_lines(data))


def _git_ignored(path: Path) -> bool | None:
    """True / False from the checkout's .gitignore rules (core.io.gitignored; no git call, so it
    also works on the PBS cluster compute nodes); None outside the checkout."""
    from core.io import gitignored

    return gitignored(path)


def held_out_secrets(approach: dict[str, Any], fixtures: bool) -> tuple[list[str], dict[str, Any]]:
    """The secrets for the secret-in-context benchmarks, from the approach's held-out file
    (`secret_in_context: {secrets: private/held_out_secrets.txt}`), one per line. Refused unless:
    the file sits in the approach's private/ folder and the repo's .gitignore ignores it (read directly;
    unknown outside a repo); it is not empty; the approach declares its used_secrets file
    (training or model selection) and no held-out secret is in it. Dry runs and fixture runs
    without a declared file use FIXTURE_SECRETS. The record holds sha256s and counts, never the
    secrets."""
    rel = (approach.get("secret_in_context") or {}).get("secrets")
    if not rel:
        if fixtures:
            return list(FIXTURE_SECRETS), {"source": "fixture secrets (dry run / smoke test only)",
                                           "n": len(FIXTURE_SECRETS), "file_sha256": None}  # fmt: skip
        raise ProtocolError("secret_in_context benchmarks need held-out secrets: add `secret_in_context: "
                            "{secrets: private/held_out_secrets.txt}` to approach.yaml, one secret per line, "
                            "none of them used in training or model selection")  # fmt: skip
    path = _approach_file(approach, rel)
    private = Path(approach["_path"]).parent / "private"
    if not path.exists():
        raise ProtocolError(f"held-out secrets file {path} is declared but missing")
    if private.resolve() not in path.resolve().parents:
        raise ProtocolError(f"held-out secrets file {path} must be inside {private} (git-ignored)")
    ignored = _git_ignored(path)
    if ignored is False:
        raise ProtocolError(f"held-out secrets file {path} is not git-ignored: it must never be committed")
    data = path.read_bytes()
    secrets = _lines(data)
    if not secrets:
        raise ProtocolError(f"held-out secrets file {path} is empty")
    used = _used_secrets(approach)
    if used is None:
        raise ProtocolError("secret_in_context: declare `synthetic: {used_secrets: ...}` too (the secrets used in "
                            "training or model selection), so the held-out secrets can be checked against it")  # fmt: skip
    overlap = sum(1 for x in secrets if x in used[2])
    if overlap:
        raise ProtocolError(f"{overlap} of the {len(secrets)} held-out secrets in {path.name} are also in "
                            f"{used[0].name} (used in training or model selection): remove them")  # fmt: skip
    return secrets, {"source": "approach held-out file", "file": str(rel),
                     "file_sha256": hashlib.sha256(data).hexdigest(), "n": len(secrets),
                     "n_distinct": len(set(secrets)), "used_secrets_sha256": hashlib.sha256(used[1]).hexdigest(),
                     "overlap_with_used": 0, "git_ignored": ignored}  # fmt: skip


def _bench_cfg(approach: dict[str, Any], bench_name: str, fixtures: bool) -> tuple[dict[str, Any], dict | None]:
    registry.discover()
    cfg: dict[str, Any] = {"download": not fixtures}
    rec = None
    if getattr(registry.get("benchmark", bench_name), "needs_held_out_secrets", False):
        cfg["held_out_secrets"], rec = held_out_secrets(approach, fixtures)
    return cfg, rec


def run_benchmark(s: Session, bench_name: str, out_root: Path, dry_run: bool = False, limit: int | None = None,
                  overwrite: bool = False, max_new_tokens: int | None = None, fixtures: bool | None = None) -> Path:  # fmt: skip
    """One benchmark for one session. dry_run: dummy model + fixture data. fixtures: fixture data
    with whatever model the session holds (smoke tests with a tiny real model)."""
    t0, started = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec="seconds")
    fixtures = dry_run if fixtures is None else fixtures
    cfg, secrets_rec = _bench_cfg(s.approach, bench_name, fixtures)
    bench = registry.get("benchmark", bench_name)(cfg, None, dry_run=fixtures)
    if not fixtures:
        _preflight(bench)
    items, held = held_out_filter(s.approach, bench, bench.items())
    full_n = len(items)
    if fixtures:
        limit = min(limit or DRY_RUN_LIMIT, DRY_RUN_LIMIT)
    if limit:
        items = items[:limit]
    out = out_root / s.approach["name"] / s.model_key / s.variant / bench_name
    if (out / "summary.json").exists() and not overwrite:
        raise ProtocolError(f"{out} already holds a run (pass --overwrite to replace its files)")
    out.mkdir(parents=True, exist_ok=True)

    s.defence.prepare(items, None)
    if s.model is None:
        if s.defence_cfg.get("train"):
            s.defence.train({**s.defence_cfg, "_dry_run": dry_run, "_base": s.base})
        s.model = s.defence.wrap(s.base)
    model = s.model
    info, fold = _model_facts(s)
    cap = int(max_new_tokens or bench.max_new_tokens)
    gen_cfg = GenConfig(max_new_tokens=cap, temperature=0.0)
    batch = int(info.get("batch_size") or PROTOCOL_BATCH_SIZE)

    order = bench.run_order(items)
    if sorted(order) != list(range(len(items))):
        raise ProtocolError(f"{bench_name}: run_order is not a permutation of the items")
    msgs = [it.messages() for it in items]
    results: dict[int, Any] = {}
    from models.hf import sdpa_backend

    with sdpa_backend(getattr(bench, "sdpa_backend", None)) as sdpa_rec:
        for start in range(0, len(order), batch):
            chunk = order[start : start + batch]
            gens = model.generate([msgs[i] for i in chunk], gen_cfg, [None] * len(chunk),
                                  secret_ctx=[SecretContext.find(msgs[i], items[i].secret) for i in chunk])
            for i, g in zip(chunk, gens):
                results[i] = g
    pos = {i: k for k, i in enumerate(order)}
    rows = []
    for i, it in enumerate(items):
        g = results[i]
        rows.append({"key": it.id, "index": i, "order": pos[i],
                     "prompt_sha256": sha256_json(msgs[i]), "rendered_sha256": _rendered_sha(s.base, msgs[i]),
                     "reply": g.text, "n_new_tokens": g.n_tokens, "finished": g.finished,
                     "scores": bench.score(it, g.text), "meta": it.metadata,
                     "secret": it.secret, "expected": it.expected_use})  # fmt: skip  (inputs for the re-scorer)
    write_jsonl(out / "items.jsonl", rows)
    metrics, per_unit = bench.summarize(items, rows)
    write_json(out / "per_unit.json", per_unit)
    extras = bench.write_extras(out, items, rows)
    if hasattr(bench, "checker_info"):
        extras["ifeval_checker"] = bench.checker_info()
    if secrets_rec is not None:
        extras["held_out_secrets"] = secrets_rec

    known = [r["finished"] for r in rows if r["finished"] is not None]
    truncation = {"rate": (sum(1 for f in known if not f) / len(known)) if known else None,
                  "n": len(known), "n_unknown": len(rows) - len(known), "cap": cap,
                  "definition": "share of replies with no EOS / end-of-turn token within the cap"}  # fmt: skip
    decoding = {"greedy": True, "do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                "repetition_penalty": 1.0, "max_new_tokens": cap, "replies_per_item": 1}  # fmt: skip
    keys = [it.id for it in items]
    data_files = bench.data_records
    reasons = []
    if dry_run:
        reasons.append("dry run (dummy model, fixture data)")
    elif fixtures:
        reasons.append("fixture data (smoke test)")
    if limit and len(items) < full_n:
        reasons.append(f"limit {limit}: {len(items)} of {full_n} items")
    if bench.expected_n is not None and full_n != bench.expected_n and not fixtures:
        reasons.append(f"{full_n} items, protocol expects {bench.expected_n}")
    if cap != bench.max_new_tokens:
        reasons.append(f"max_new_tokens {cap}, protocol {bench.max_new_tokens}")
    if data_files and not all(d["match"] for d in data_files):
        reasons.append("a data file does not match its pinned hash")
    if "checker_unavailable" in metrics:
        reasons.append("IFEval checker unavailable: " + str(metrics["checker_unavailable"]["note"]))
    from core.cluster import environment

    env = environment()
    if not dry_run and not env["pins"]["match"]:
        reasons.append("library versions differ from eval/requirements-gpu.txt: " + "; ".join(env["pins"]["differences"]))
    if not dry_run and not fixtures and not (env.get("gpu") or {}).get("name"):
        reasons.append("no CUDA GPU recorded (real protocol runs are GPU runs)")
    if fold is not None and s.model_key == "gemma" and fold["layout"] != "folded into first user turn":
        reasons.append(f"Gemma system text not folded into the first user turn ({fold['layout']})")
    # The pairing fingerprint: everything that must be identical between the original and the
    # technique run. The model class is left out on purpose (an adapter makes it PeftModel).
    settings = {"benchmark": bench_name, "model": {k: info.get(k) for k in ("hf_id", "revision_loaded", "dtype_loaded",
                                                                            "attn_implementation")},
                "batch_size": batch, "decoding": decoding, "item_keys": sha256_json(keys),
                "order": sha256_json([keys[i] for i in order]), "prompts": sha256_json([r["prompt_sha256"] for r in rows]),
                "data": sorted((d["name"], d["sha256"]) for d in data_files), "versions": info.get("versions"),
                "chat_template": info.get("chat_template_sha256"), "dry_run": dry_run, "fixtures": fixtures,
                "held_out_secrets": (secrets_rec or {}).get("file_sha256")}  # fmt: skip
    if sdpa_rec["backend"] != "default":  # only then: default benchmarks keep their earlier fingerprint
        settings["sdpa_backend"] = sdpa_rec["backend"]
    summary = {
        "schema_version": SCHEMA_VERSION, "protocol_doc": PROTOCOL_DOC, "approach": s.approach["name"],
        "variant": s.variant, "defence": {k: v for k, v in s.defence_cfg.items() if not str(k).startswith("_")},
        # what the defence checked after wrap() (e.g. the FSA bridge: adapter sha256, layout, counts)
        "defence_info": s.defence.info() if hasattr(s.defence, "info") else None,
        "benchmark": bench_name, "suite": bench.suite, "team_table": bench.team_table, "model": info,
        "system_fold_check": {k: v for k, v in (fold or {}).items() if k != "rendered"} or None,
        "decoding": decoding, "batch_size": batch, "n_items": len(items), "expected_n": bench.expected_n,
        "item_keys_sha256": settings["item_keys"], "order_sha256": settings["order"],
        "prompts_sha256": settings["prompts"], "data_files": data_files,
        "versions": {"python": platform.python_version(), **(info.get("versions") or {}), "numpy": _np_version()},
        "git": git_commit(), "metrics": metrics, "primary": list(getattr(bench, "primary", ())),
        "truncation": truncation, "scorer_limits": list(bench.scorer_limits), "reference": bench.reference,
        "sdpa_backend": sdpa_rec, "original_from": s.approach.get("_original_from"),
        "extras": extras, "held_out_check": held, "settings_sha256": sha256_json(settings),
        "protocol_compliant": not reasons, "non_compliance": reasons, "dry_run": dry_run, "fixtures": fixtures,
        "limit": limit,
        "started_utc": started, "wall_s": round(time.perf_counter() - t0, 3), "environment": env,
    }  # fmt: skip
    write_json(out / "summary.json", summary)
    return out


def _preflight(bench: Any) -> None:
    """Real-data runs only: fail before any generation if the benchmark cannot be scored."""
    try:
        bench.preflight()
    except Exception as e:  # noqa: BLE001 (the benchmark says what is missing)
        raise ProtocolError(f"{bench.name}: preflight failed, nothing generated: {e}") from e


def preflight(names: Sequence[str], approach: dict[str, Any] | None = None) -> None:
    """Real runs, before the model loads: every benchmark's preflight (e.g. the IFEval checker
    loads) and, for the secret-in-context benchmarks, the held-out secrets file checks. Raises
    ProtocolError."""
    for b in names:
        cfg, _ = _bench_cfg(approach or {}, b, fixtures=False)
        _preflight(registry.get("benchmark", b)(cfg, None, dry_run=False))


def _np_version() -> str:
    import numpy

    return numpy.__version__


def run_protocol(approach: str, model_key: str, variant: str, suite: str = "core",
                 benchmarks: Sequence[str] | None = None, dry_run: bool = False, device: str | None = None,
                 limit: int | None = None, output_dir: str | Path | None = None, overwrite: bool = False,
                 model: ModelWrapper | None = None, approach_spec: dict[str, Any] | None = None,
                 fixtures: bool | None = None) -> list[Path]:  # fmt: skip
    """Run every benchmark of `suite` (or the named `benchmarks`) for one variant."""
    spec = approach_spec or load_approach(approach)
    spec["_original_from"] = check_original_from(spec)  # before anything runs: refuses a mismatch
    names = list(benchmarks) if benchmarks else list(SUITES[suite])
    unknown = [b for b in names if b not in SUITES["all"]]
    if unknown:
        raise ProtocolError(f"not protocol benchmarks: {unknown}; known: {SUITES['all']}")
    out_root = Path(output_dir) if output_dir else ROOT / ("outputs/protocol_dry" if dry_run else "outputs/protocol")
    if not (dry_run if fixtures is None else fixtures):
        preflight(names, spec)  # before the model loads: a missing scorer fails in seconds
    s = open_session(spec, model_key, variant, dry_run=dry_run, device=device, model=model)
    return [run_benchmark(s, b, out_root, dry_run=dry_run, limit=limit, overwrite=overwrite, fixtures=fixtures)
            for b in names]
