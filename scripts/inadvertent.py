"""Inadvertent leakage of the FSA-lite run3 adapters (run3, and run3 without the USE term),
measured with the FSA v1 pipeline: the original model vs each adapter on identical prompts,
secrets, seeds and decoding.

One config per (model, variant, prompt set) in configs/inadvertent/ (docs/inadvertent_leakage.md).
The variants of a (model, prompt set) group are the original and every technique of TECHNIQUES;
they are identical except name, variant and adapter, and a technique's adapter spec is its own
approach spec (configs/eval/<variant>.approach.yaml); `check` refuses otherwise. Subcommands (run
from the repository root):

    check                              every config: pairing, adapter key / pin (no torch, no GPU);
                                       a FILL_ME pin is listed (generate refuses that config)
    generate CONFIG --backend B        scripts.baseline_leakage with the config's settings
    score CONFIG --threat T --backend B
                                       scripts.likelihood_attack --calibrate on the config's newest
                                       run; T = grey_box (base + the run's adapter) or
                                       public_weights (base only, adapter run only)
    report --model qwen|gemma|all      paired statistics (fsa.eval.paired_leakage), one md + csv
                                       table per model: original, run3, run3 no-use and the
                                       contrasts of CONTRASTS; aggregates only
    estimate                           the GPU-time estimates of docs/inadvertent_leakage.md

--backend mock: synthetic replies (fsa_runtime.generators.mock), runs in mock_data/runs named
mock_...; `score --backend mock` uses a character tokenizer and a digit-dip scorer instead of a
model (no torch). Mock numbers say nothing about a real model: they only exercise the pipeline.
Job logs hold counts and metrics only: both scripts run with --no-secrets-in-log (no secret, code,
guess or reply text is printed; the run folders keep everything, git-ignored).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fsa.eval.parsing import parse_numbers
from fsa.io.run_writer import CONFIG_FILE, METRICS_FILE
from fsa_runtime.settings import ROOT, load_settings

from scripts._cli import utf8_stdio

CONFIG_DIR = ROOT / "configs" / "inadvertent"
PAIR_FREE_KEYS = ("name", "variant", "adapter")  # the only keys that may differ within a pair
TECHNIQUES = ("fsa_lite_run3", "fsa_lite_run3_nouse")  # each = the name of the approach spec pinning it
VARIANTS = ("original", *TECHNIQUES)
# (technique, reference) contrasts of the report, each a paired bootstrap on the same held-out secrets
CONTRASTS = (("fsa_lite_run3", "original"), ("fsa_lite_run3_nouse", "original"), ("fsa_lite_run3_nouse", "fsa_lite_run3"))
LABELS = {"original": "original", "fsa_lite_run3": "run3", "fsa_lite_run3_nouse": "run3 no-use"}
MODELS = {"qwen": "qwen2.5-7b-instruct", "gemma": "gemma3-12b-it"}
MACHINES = {"qwen": "NSCC (PBS, A100-40GB)", "gemma": "TC2 (Slurm, A40)"}
PROMPTS = ("rand20", "lottery", "unpredictabench")  # primary first
THREATS = ("grey_box", "public_weights")
RUN_RE = r"^\d{8}-\d{6}_(?:mock_)?%s$"
MOCK_ALPHA = 0.6  # configs/training/mock.yaml dip_alpha.base: what the mock attacker assumes


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- configs


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    need = {"name", "model", "variant", "adapter", "prompt", "secrets", "replies", "decoding", "used_secrets"}
    missing = sorted(need - set(cfg))
    if missing:
        raise ConfigError(f"{Path(path).name}: missing {missing}")
    if cfg["name"] != Path(path).stem:
        raise ConfigError(f"{Path(path).name}: name {cfg['name']!r} must equal the file name")
    if cfg["variant"] not in VARIANTS:
        raise ConfigError(f"{cfg['name']}: variant must be one of {VARIANTS}")
    if (cfg["variant"] == "original") != (cfg["adapter"] is None):
        raise ConfigError(f"{cfg['name']}: the original variant has no adapter, the other one has one")
    if cfg["adapter"] is not None and set(cfg["adapter"]) != {"key", "spec"}:
        raise ConfigError(f"{cfg['name']}: adapter must be {{key: <paths.local.yaml key>, spec: <approach spec>}}")
    if cfg["adapter"] is not None and cfg["adapter"]["spec"] != spec_of(cfg["variant"]):
        raise ConfigError(f"{cfg['name']}: variant {cfg['variant']} is pinned by {spec_of(cfg['variant'])}, "
                          f"not {cfg['adapter']['spec']}")  # fmt: skip
    if len(cfg["prompt"]) != 1 or next(iter(cfg["prompt"])) not in ("id", "set"):
        raise ConfigError(f"{cfg['name']}: prompt must be {{id: <prompt id>}} or {{set: <prompt set>}}")
    if cfg["secrets"].get("type") != "digits":
        raise ConfigError(f"{cfg['name']}: only 2-digit secrets (type digits)")
    return cfg


def spec_of(variant: str) -> str:
    """The approach spec (repository-relative) that pins a technique variant's adapter."""
    return f"configs/eval/{variant}.approach.yaml"


def all_configs(config_dir: Path = CONFIG_DIR) -> dict[str, dict[str, Any]]:
    return {p.stem: load_config(p) for p in sorted(Path(config_dir).glob("*.yaml"))}


def shared(cfg: dict[str, Any]) -> dict[str, Any]:
    """The part of a config both variants of a pair must have identically."""
    return {k: v for k, v in cfg.items() if k not in PAIR_FREE_KEYS}


def pair_name(cfg: dict[str, Any]) -> str:
    """inadv_<model>_<prompt>_<variant> -> inadv_<model>_<prompt>."""
    suffix = "_" + cfg["variant"]
    if not cfg["name"].endswith(suffix):
        raise ConfigError(f"{cfg['name']}: name must end with _{cfg['variant']}")
    return cfg["name"][: -len(suffix)]


def pairs(configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """{group name: {variant: config}}; ConfigError unless every group holds every variant of
    VARIANTS and each technique is identical to the original except PAIR_FREE_KEYS."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cfg in configs.values():
        out.setdefault(pair_name(cfg), {})[cfg["variant"]] = cfg
    for name, p in out.items():
        if set(p) != set(VARIANTS):
            raise ConfigError(f"pair {name}: has {sorted(p)}, needs {list(VARIANTS)}")
        a = shared(p["original"])
        for tech in TECHNIQUES:
            b = shared(p[tech])
            if a != b:
                diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                raise ConfigError(f"pair {name}: original and {tech} differ in {diff} (only {list(PAIR_FREE_KEYS)} may)")
    return out


def method_of(cfg: dict[str, Any]) -> str:
    """base without an adapter, fsa with one. Both use the same "base" system template
    (fsa.prompting.system_prompt), so the prompt the model sees is identical."""
    return "base" if cfg["adapter"] is None else "fsa"


def baseline_args(cfg: dict[str, Any], *, backend: str, device: str | None) -> list[str]:
    """scripts.baseline_leakage arguments for one config."""
    s, d = cfg["secrets"], cfg["decoding"]
    kind, value = next(iter(cfg["prompt"].items()))
    args = [
        "--backend", backend, "--base-model", cfg["model"], "--method", method_of(cfg),
        "--name", cfg["name"], "--seed", str(s["seed"]), "--n-train", str(s["n_train"]),
        "--n-heldout", str(s["n_heldout"]), "--replies", str(cfg["replies"]),
        "--temperature", str(d["temperature"]), "--max-new-tokens", str(d["max_new_tokens"]),
        "--batch-size", str(d["batch_size"]), "--dtype", d["dtype"],
        "--prompt-id" if kind == "id" else "--prompt-set", value,
        "--used-secrets", cfg["used_secrets"], "--no-secrets-in-log",
    ]  # fmt: skip
    if cfg["adapter"] is not None:
        args += ["--adapter-key", cfg["adapter"]["key"], "--adapter-spec", cfg["adapter"]["spec"]]
    if device:
        args += ["--device", device]
    return args


# --------------------------------------------------------------------------- runs


def runs_dir(backend: str) -> Path:
    s = load_settings()
    return s.runs_dir if backend == "real" else s.mock_runs_dir


def find_run(base: Path, name: str) -> Path | None:
    """The newest finished run folder (metrics.json written) of a config name."""
    pat = re.compile(RUN_RE % re.escape(name))
    found = sorted(p for p in Path(base).glob(f"*{name}") if pat.match(p.name) and (p / METRICS_FILE).exists())
    return found[-1] if found else None


# --------------------------------------------------------------------------- mock scoring


class MockCharTokenizer:
    """Character-level stand-in tokenizer with a ChatML chat template, for `score --backend mock`
    only (no torch / transformers). Characters outside printable ASCII map to a space."""

    SPECIALS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")

    def __init__(self) -> None:
        vocab = [*self.SPECIALS, *(chr(c) for c in range(32, 127)), "\n"]
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.inv = dict(enumerate(vocab))
        self.eos_token_id = self.vocab["<|im_end|>"]
        self.pad_token_id = self.vocab["<|endoftext|>"]
        self.all_special_ids = [self.vocab[t] for t in self.SPECIALS]
        self.padding_side = "right"

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids, i = [], 0
        while i < len(text):
            sp = next((t for t in self.SPECIALS if text.startswith(t, i)), None)
            if sp:
                ids.append(self.vocab[sp])
                i += len(sp)
            else:
                ids.append(self.vocab.get(text[i], self.vocab[" "]))
                i += 1
        return {"input_ids": ids}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        special = set(self.all_special_ids)
        return "".join(self.inv[i] for i in ids if not (skip_special_tokens and i in special))


def mock_score_fn(model, tokenizer, build_prompt_ids, reply_token_ids, candidates, temperature) -> np.ndarray:
    """log-likelihood of the replies' digits under the mock's dip model p(d) ∝ 1 - α·[d in c]
    (α = MOCK_ALPHA), for each candidate c; same signature as likelihood.score_candidates."""
    digits = []
    for ids in reply_token_ids:
        for n in parse_numbers(tokenizer.decode(ids, skip_special_tokens=True)):
            digits += [int(ch) for ch in f"{n:02d}"]
    counts = np.bincount(np.asarray(digits, dtype=int), minlength=10) if digits else np.zeros(10)
    out = []
    for c in candidates:
        p = np.ones(10)
        p[[int(ch) for ch in set(c)]] -= MOCK_ALPHA
        out.append(float(counts @ np.log(p / p.sum())))
    return np.asarray(out) / temperature


def score_mock(cfg: dict[str, Any], run: Path, threat: str) -> Path:
    from fsa.contract.schemas import ScorerInfo
    from fsa.prompting import load_system_templates
    from fsa_runtime.registry import load_registry

    from scripts import likelihood_attack as LA

    config = yaml.safe_load((run / CONFIG_FILE).read_text(encoding="utf-8"))
    adapter = (config.get("generation") or {}).get("adapter")
    scorer = ScorerInfo(
        model=cfg["model"],
        hf_id=load_registry().base_models[cfg["model"]].hf_id,
        same_as_generator=True,
        method=config["method"],
        device="cpu",
        dtype="mock",
        temperature=float((config.get("eval") or {}).get("temperature") or 1.0),
        adapter=adapter if threat == "grey_box" else None,
        threat_model=threat,
    )
    _, path = LA.likelihood_attack(
        run,
        tokenizer=MockCharTokenizer(),
        scorer=scorer,
        templates=load_system_templates(),
        max_batch_tokens=10**7,
        calibrate=True,
        log=lambda m: None,  # per-secret lines name the secret
        extra_notes=["MOCK scorer: character tokenizer + digit-dip likelihood, no model; says nothing about leakage"],
        score_fn=mock_score_fn,
    )
    return path


# --------------------------------------------------------------------------- report


def standing(cfg: dict[str, Any]) -> str | None:
    """A technique's standing for this model, from its approach spec (models.<key>.post_hoc /
    pre_registered): "post hoc", "pre-registered (<doc>)", or None if the spec says neither."""
    spec = yaml.safe_load((ROOT / cfg["adapter"]["spec"]).read_text(encoding="utf-8")) or {}
    key = next((k for k, v in MODELS.items() if v == cfg["model"]), None)
    m = (spec.get("models") or {}).get(key) or {}
    if m.get("pre_registered"):
        return f"pre-registered ({m['pre_registered']})"
    return "post hoc" if m.get("post_hoc") else None


def pick_attack(run: Path, threat: str):
    """The file a table shows for one threat model: a usable calibration first, then the newest
    (docs/fsa/likelihood_attack.md display rule). Files of the other threat model never compete,
    even when their scores are identical (read_attack_files' duplicate_of ignores the threat)."""
    from fsa.eval.run_extras import read_attack_files, recorded_threat_model, usable_calibration

    files = [f for f in read_attack_files(run) if f.attack is not None]
    files = [f for f in files if recorded_threat_model(f.attack) == threat]
    if not files:
        return None, None
    best = max(files, key=lambda f: (usable_calibration(f.attack), f.attack.created_at))
    return best.attack, best.path.name


def _fmt(e, pct: bool = False, signed: bool = False) -> str:
    if e is None:
        return "—"
    f = (lambda x: f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}") if pct else (
        lambda x: f"{x:+.3f}" if signed else f"{x:.3f}"
    )
    return f"{f(e.value)} [{f(e.low)}, {f(e.high)}]"


METRIC_LABELS = {
    "bits_calibrated": "≥ bits, calibrated",
    "bits_raw": "≥ bits, raw",
    "exact": "exact (%)",
    "top5": "top-5 (%)",
}
# One row per (prompt, threat model, metric, contrast): technique − reference, paired over secrets.
CSV_COLUMNS = [
    "model", "prompt", "threat_model", "metric", "technique", "reference", "status", "n_secrets",
    "replies_per_secret", "reference_value", "reference_low", "reference_high", "technique_value",
    "technique_low", "technique_high", "difference", "difference_low", "difference_high",
    "tau_reference", "tau_technique", "degenerate_reference", "degenerate_technique",
    "reference_file", "technique_file", "heldout_in_used_secrets", "train_in_used_secrets",
    "n_bootstrap", "seed", "is_mock",
]  # fmt: skip


def _contrast(tech: str, ref: str, runs: dict, cfgs: dict, threat: str, *, n_bootstrap: int, seed: int):
    """(status, PairedResult | None, reference file, technique file) of one contrast. The
    original has no adapter, so its grey_box file IS the public-weights scorer and stands as the
    reference under both threat models; a technique reference is read under the same threat model."""
    from fsa.eval.paired_leakage import PairingError, check_run_pair, paired

    missing = [v for v in (ref, tech) if runs[v] is None]
    if missing:
        return f"no finished run: {', '.join(missing)}", None, None, None
    problems = check_run_pair(cfgs[ref], cfgs[tech], reference_has_adapter=ref != "original")
    if problems:
        return "not paired: " + "; ".join(problems), None, None, None
    ref_threat = "grey_box" if ref == "original" else threat
    a, fa = pick_attack(runs[ref], ref_threat)
    b, fb = pick_attack(runs[tech], threat)
    if a is None or b is None:
        lack = [x for x, f in ((f"{ref} {ref_threat}", a), (f"{tech} {threat}", b)) if f is None]
        return "no attack file: " + ", ".join(lack), None, fa, fb
    try:
        return "ok", paired(a, b, n_resamples=n_bootstrap, seed=seed), fa, fb
    except PairingError as e:
        return f"not paired: {e}", None, fa, fb


def _used_overlap(cfgs: dict) -> tuple[Any, Any]:
    """(held-out, train) counts in the used-secrets file, from the first technique run that
    recorded them (every technique counts against the same file); (None, None) if none did."""
    for tech in TECHNIQUES:
        ov = ((cfgs[tech] or {}).get("eval") or {}).get("used_secrets_overlap") or {}
        if ov.get("available"):
            return (ov.get("heldout") or {}).get("n_in_used"), (ov.get("train") or {}).get("n_in_used")
    return None, None


def report_model(model: str, *, backend: str, base: Path, out_dir: Path, n_bootstrap: int, seed: int,
                 configs: dict[str, dict[str, Any]] | None = None) -> tuple[Path, Path]:  # fmt: skip
    configs = configs if configs is not None else all_configs()
    ps = pairs(configs)
    rows: list[dict[str, Any]] = []
    md_rows: list[str] = []
    notes: list[str] = []
    overlap: dict[str, tuple[Any, Any]] = {}
    standings: dict[str, str | None] = {}
    for prompt in PROMPTS:
        name = f"inadv_{model}_{prompt}"
        if name not in ps:
            continue
        cfg = ps[name]
        for tech in TECHNIQUES:
            standings.setdefault(tech, standing(cfg[tech]))
        runs = {v: find_run(base, cfg[v]["name"]) for v in VARIANTS}
        cfgs = {v: yaml.safe_load((r / CONFIG_FILE).read_text(encoding="utf-8")) if r else None for v, r in runs.items()}
        held, train = overlap[prompt] = _used_overlap(cfgs)
        base_row = {"model": cfg["original"]["model"], "prompt": prompt, "n_bootstrap": n_bootstrap, "seed": seed,
                    "replies_per_secret": cfg["original"]["replies"], "is_mock": backend == "mock",
                    "heldout_in_used_secrets": held, "train_in_used_secrets": train}  # fmt: skip
        for threat in THREATS:
            res = {}
            for tech, ref in CONTRASTS:
                st, r, fa, fb = _contrast(tech, ref, runs, cfgs, threat, n_bootstrap=n_bootstrap, seed=seed)
                res[(tech, ref)] = (st, r)
                if r is not None:
                    notes += [f"{prompt} / {threat} / {LABELS[tech]} vs {LABELS[ref]}: {n}" for n in r.notes]
                for m in METRIC_LABELS:
                    row = dict.fromkeys(CSV_COLUMNS)
                    row.update(base_row, threat_model=threat, metric=m, technique=tech, reference=ref, status=st,
                               reference_file=fa, technique_file=fb)  # fmt: skip
                    if r is not None:
                        row["n_secrets"] = r.n_secrets
                        for side, col in (("original", "reference"), ("adapter", "technique"), ("difference", None)):
                            e = getattr(r, side)[m]
                            if e is None:
                                continue
                            if col is None:
                                row["difference"], row["difference_low"], row["difference_high"] = e.value, e.low, e.high
                            else:
                                row[f"{col}_value"], row[f"{col}_low"], row[f"{col}_high"] = e.value, e.low, e.high
                        row["tau_reference"], row["tau_technique"] = r.tau["original"], r.tau["adapter"]
                        row["degenerate_reference"] = r.degenerate["original"]
                        row["degenerate_technique"] = r.degenerate["adapter"]
                    rows.append(row)
            md_rows += _md_rows(prompt, threat, res)
    out = Path(out_dir) / model
    out.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out / "inadvertent_leakage.csv", out / "inadvertent_leakage.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    overlap_text = "; ".join(
        f"{p}: {h}/50 held-out, {t}/50 train" if h is not None else f"{p}: not counted (file absent at generation)"
        for p, (h, t) in overlap.items()
    )
    contrast_heads = " | ".join(f"{LABELS[a]} − {LABELS[b]} [95% CI]" for a, b in CONTRASTS)
    value_heads = " | ".join(f"{LABELS[v]} [95% CI]" for v in VARIANTS)
    stand = "; ".join(f"{LABELS[t]}: {s}" for t, s in standings.items() if s)
    lines = [
        f"# Inadvertent leakage: {MODELS[model]}, original vs FSA-lite run3 vs run3 no-use"
        + (" (MOCK: synthetic)" if backend == "mock" else ""),
        "",
        (
            f"Generated {datetime.now().isoformat(timespec='seconds')} by scripts/inadvertent.py report from "
            f"`{base.name}`; machine {MACHINES[model]}. Likelihood attacker (100 candidate codes), held-out "
            f"secrets, 95% percentile bootstrap over secrets ({n_bootstrap} draws, seed {seed}); every "
            "difference is a paired bootstrap of its two variants on the same resampled secrets."
        ),
        "",
        f"| prompt | threat model | metric | n | {value_heads} | {contrast_heads} |",
        "|---|---|---|---|" + "---|" * (len(VARIANTS) + len(CONTRASTS)),
        *md_rows,
        "",
        (
            "- **grey_box**: an adapter run scored by base + its own adapter; **public_weights**: the adapter "
            "run's replies scored by the base model only. The original is always its own (base-model) grey_box "
            "file, which is also the public-weights model."
        ),
        (
            "- **≥ bits** = max(0, log2 100 − mean cross-entropy of the true code), the lower bound every "
            "likelihood_attack file reports. *calibrated*: softmax(scores / τ), τ fitted on the attacker-train "
            "secrets of the same run; *raw*: softmax(scores). † τ at its search bound in a variant of that "
            "row: the calibrated value is degenerate, read the raw row."
        ),
        "- exact / top-5: tie-aware, from the calibrated posterior when usable, else the raw one.",
        *([f"- standing on {MODELS[model]}: {stand}."] if stand else []),
        "- used-secrets overlap (held-out / train secrets that occur in run3's training values, exact "
        "line match; counts only; a 100-value space, so overlap is expected and does not invalidate "
        "the test): " + overlap_text + ".",
        *(["- " + n for n in notes]),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return md_path, csv_path


def _md_rows(prompt: str, threat: str, res: dict) -> list[str]:
    """Markdown rows of one (prompt, threat): each variant's value comes from its contrast with
    the original (with the same seed and secret order, the original's estimate is the same in
    every contrast), then one column per contrast; a missing contrast shows its status."""
    ok = {k: r for k, (_, r) in res.items() if r is not None}
    if not ok:
        st = "; ".join(f"{LABELS[a]} vs {LABELS[b]}: {s}" for (a, b), (s, _) in res.items())
        return [f"| {prompt} | {threat} | — | — |" + " — |" * (len(VARIANTS) + len(CONTRASTS) - 1) + f" {st} |"]
    out = []
    for m, label in METRIC_LABELS.items():
        pct = m in ("exact", "top5")
        value = {"original": next((r.original[m] for (_, b), r in ok.items() if b == "original"), None)}
        for tech in TECHNIQUES:
            r = ok.get((tech, "original"))
            value[tech] = r.adapter[m] if r is not None else None
        deg = m == "bits_calibrated" and any(r.degenerate["original"] or r.degenerate["adapter"] for r in ok.values())
        n = next(iter(ok.values())).n_secrets
        cells = [_fmt(value[v], pct) for v in VARIANTS]
        for key in CONTRASTS:
            st, r = res[key]
            cells.append(_fmt(r.difference[m], pct, True) if r is not None else f"— ({st})")
        out.append(f"| {prompt} | {threat} | {label}{' †' if deg else ''} | {n} | " + " | ".join(cells) + " |")
    return out


# --------------------------------------------------------------------------- estimates

# Per-unit GPU-time estimates (seconds). Generation: the v1 runs' spacing (Gemma-3-12B on an A40,
# 100 secrets x 10 rand_20_2digit replies, batch 8 = 2 generate calls per secret: ~25 min per run,
# so ~6.5 s per call); Qwen2.5-7B on an A100-40GB taken as ~2x faster per decoding step. Scoring:
# teacher-forced forward passes, 100 candidates x 10 replies x ~80 tokens = ~80k tokens per secret,
# ~2 x params FLOPs per token (+ the LM head) at ~50% of bf16 peak: Qwen ~8 s, Gemma ~28 s per
# secret. Rough: every job prints its own ETA; read the first one before submitting the rest.
GEN_S_PER_CALL = {"qwen": 3.0, "gemma": 6.5}  # one generate call (<= 8 replies x <= 160 tokens)
SCORE_S_PER_SECRET = {"qwen": 8.0, "gemma": 28.0}  # 100 candidates x 10 replies, one secret
LOAD_S = {"qwen": 120, "gemma": 240}


def estimate(model: str, prompt: str) -> dict[str, float]:
    """Minutes: generation of one variant (100 secrets) and one likelihood scoring with
    --calibrate (50 held-out + 50 train secrets)."""
    replies = 12 if prompt == "unpredictabench" else 10
    calls = 4 if prompt == "unpredictabench" else math.ceil(replies / 8)  # 4 prompts x 3, or 10 = 8 + 2
    length = 0.5 if prompt == "lottery" else 1.0  # six numbers stop early
    gen = LOAD_S[model] + 100 * calls * GEN_S_PER_CALL[model] * length
    score = LOAD_S[model] + 100 * SCORE_S_PER_SECRET[model] * (replies / 10) * length
    return {"generate_min": gen / 60, "score_min": score / 60}


# --------------------------------------------------------------------------- CLI


def _config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.suffix:
        p = CONFIG_DIR / f"{path}.yaml"
    elif not p.is_absolute():
        p = ROOT / p
    return load_config(p)


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="validate every config and its pairing")
    g = sub.add_parser("generate", help="baseline_leakage for one config")
    g.add_argument("config", help="configs/inadvertent/<name>.yaml or <name>")
    g.add_argument("--backend", choices=("real", "mock"), required=True)
    g.add_argument("--device", default=None)
    s = sub.add_parser("score", help="likelihood_attack --calibrate on one config's newest run")
    s.add_argument("config")
    s.add_argument("--threat", choices=THREATS, required=True)
    s.add_argument("--backend", choices=("real", "mock"), required=True)
    s.add_argument("--device", default=None)
    s.add_argument("--max-batch-tokens", type=int, default=None)
    r = sub.add_parser("report", help="paired tables, one per model (md + csv)")
    r.add_argument("--model", choices=(*MODELS, "all"), default="all")
    r.add_argument("--backend", choices=("real", "mock"), default="real")
    r.add_argument("--runs-dir", type=Path, default=None)
    r.add_argument("--out-dir", type=Path, default=None, help="default outputs/inadvertent (mock: mock_data/inadvertent)")
    r.add_argument("--n-bootstrap", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    sub.add_parser("estimate", help="print the GPU-time estimates")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            return _check()
        if a.cmd == "generate":
            from scripts import baseline_leakage

            cfg = _config(a.config)
            pairs({**all_configs(), cfg["name"]: cfg})
            return baseline_leakage.main(baseline_args(cfg, backend=a.backend, device=a.device))
        if a.cmd == "score":
            cfg = _config(a.config)
            if a.threat == "public_weights" and cfg["adapter"] is None:
                raise ConfigError(f"{cfg['name']} has no adapter: its grey_box scorer already is the base "
                                  "(public-weights) model; score public_weights on a technique config")  # fmt: skip
            run = find_run(runs_dir(a.backend), cfg["name"])
            if run is None:
                raise ConfigError(f"no finished {a.backend} run of {cfg['name']} in {runs_dir(a.backend)}")
            print(f"run: {run.name}; threat model {a.threat}", flush=True)
            if a.backend == "mock":
                print(f"wrote {score_mock(cfg, run, a.threat)}")
                return 0
            from scripts import likelihood_attack

            args = [str(run), "--calibrate", "--no-secrets-in-log"] + (["--device", a.device] if a.device else [])
            args += ["--scorer-no-adapter"] if a.threat == "public_weights" else []
            args += ["--max-batch-tokens", str(a.max_batch_tokens)] if a.max_batch_tokens else []
            return likelihood_attack.main(args)
        if a.cmd == "report":
            base = a.runs_dir or runs_dir(a.backend)
            out = a.out_dir or ROOT / ("outputs" if a.backend == "real" else "mock_data") / "inadvertent"
            for m in MODELS if a.model == "all" else (a.model,):
                md, cs = report_model(m, backend=a.backend, base=base, out_dir=out, n_bootstrap=a.n_bootstrap, seed=a.seed)
                print(f"{m}: wrote {md} and {cs.name}")
            return 0
        if a.cmd == "estimate":
            print("model   prompt            generate/variant  score/threat (--calibrate)")
            for m in MODELS:
                for p in PROMPTS:
                    e = estimate(m, p)
                    print(f"{m:<7} {p:<17} {e['generate_min']:>6.0f} min        {e['score_min']:>6.0f} min")
            return 0
    except (ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 2


def _check() -> int:
    """Pairing of every group and each technique's pin. A FILL_ME pin (an adapter not trained or
    not pinned yet) is listed, not an error: `generate` refuses that config until it is pinned. Any
    other pin problem (a key no model uses, a malformed sha256) fails the check."""
    from fsa.training import peft_check

    configs = all_configs()
    ps = pairs(configs)
    unpinned = []
    for name, p in sorted(ps.items()):
        used = (ROOT / p["original"]["used_secrets"]).is_file()
        print(f"{name}: {len(VARIANTS)} variants ok (each technique differs from the original only in "
              f"{list(PAIR_FREE_KEYS)}); used-secrets file {'present' if used else 'absent'}")  # fmt: skip
        for tech in TECHNIQUES:
            ad = p[tech]["adapter"]
            path, source = peft_check.local_adapter(ad["key"], ROOT)
            here = f"here: {'found' if path and path.exists() else 'not on this machine'} ({source})"
            try:
                pin = peft_check.pin_for_local_key(ROOT / ad["spec"], ad["key"])
            except peft_check.AdapterError as e:
                if peft_check.PLACEHOLDER not in str(e):
                    raise ConfigError(f"{p[tech]['name']}: {e}") from None
                unpinned.append(p[tech]["name"])
                print(f"  {tech}: adapter {ad['key']} NOT PINNED ({ad['spec']}: {peft_check.PLACEHOLDER}); "
                      f"generate refuses it; {here}")  # fmt: skip
                continue
            print(f"  {tech}: adapter {ad['key']} pinned {pin['adapter_sha256'][:16]}... "
                  f"({pin['spec']} models.{pin['model']}); {here}")  # fmt: skip
    print(f"{len(configs)} configs, {len(ps)} groups; not pinned yet: {len(unpinned)}"
          + (f" ({', '.join(unpinned)})" if unpinned else ""))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
