"""Bridge to the fake-secret-average (FSA) defence: loads an existing FSA LoRA adapter onto the
harness's HF model and checks that it is really attached.

Status: 🟡 implemented. Header parsing, layout detection, key remapping and the refusals are
unit-tested on CPU (tests/test_fsa_bridge.py). Loading through PEFT is tested there with tiny
random Gemma 3 / Qwen2 when peft is installed. On google/gemma-3-12b-it with the pinned stack
(transformers 4.51.3, peft 0.15.2) it is unverified until the Slurm cluster adapter-check job has run:
    sbatch scripts/slurm/harness_adapter_check.sbatch

Why the remap: FSA adapters were trained under transformers 5.x, where Gemma3ForConditionalGeneration
names its text layers "model.language_model.layers.N"; under 4.51.3 they are
"language_model.model.layers.N". PEFT matches saved keys to the live module names and, on a
mismatch, skips the weights with only a warning, so the "technique" would silently equal the
original model. The bridge therefore:
  1. reads adapter_config.json and the safetensors header (no tensors);
  2. reads the decoder-layer prefix of the LIVE model from named_modules() (data-driven: no
     assumption about the transformers version or model class; vision-tower modules are ignored);
  3. checks every adapter target against the live modules (layer exists, shapes fit r) and the
     config's target_modules against the adapter's modules (catches a partial adapter);
  4. if the adapter's prefix differs, writes a remapped copy (keys renamed only; the tensor bytes
     are copied unchanged and their sha256 is checked; target_modules is rewritten only if it
     names the old path) under $HARNESS_DATA/adapters_remapped/ and loads that;
  5. after loading, checks the live model: LoRA A and B attached on exactly the adapter's modules
     (count equal to the file and to the expected count below), 0 missing, 0 unexpected, every
     attached tensor equal to the file's, every lora_B non-zero, r / alpha / scaling as in the
     config, no LoRA on a vision module. Any mismatch raises AdapterError: nothing runs.
Steps 1-5 (and the pin check) live in src/fsa_lite/fsa/training/peft_check.py, which the FSA v1
pipeline (scripts/baseline_leakage.py, scripts/likelihood_attack.py) uses too; this module re-exports
them, so `F.plan`, `F.AdapterError`, ... are unchanged.
Expected tensor count: 2 x the live modules the adapter's own target_modules selects (a full
adapter on all layers or a layer-restricted one, e.g. Gemma global layers 5 / 11 / 17), cross-checked
against training_info.json: its `layers` must be exactly the adapted layers and its
`target_modules` must select the same modules. Both trainers of this repository write these keys:
scripts.train_fsa (v1, fsa.training.trainer) and src/fsa_lite/lite/v2_train.py (v2 / FSA-lite).
adapter_config.json keys the installed peft does not know (a newer peft saved the adapter) are
accepted only at their harmless defaults (NEWER_PEFT_DEFAULTS) and recorded; any other value refuses.
`info()` returns what was checked (adapted layers included); core/protocol.py writes it to
summary.json as defence_info.

Config (approach.yaml models.<key>.defence, or an attack-study config's defence):
    adapter: /abs/path/to/<run>/adapter      # absolute paths are kept as they are, or one per
    adapter:                                 # cluster, chosen by core.cluster (HARNESS_CLUSTER):
      slurm-site: /home/.../adapter
      pbs-site: /scratch/.../adapter
    # FSA_ADAPTER_DIR=/path in the job environment overrides both (recorded as the source)
    adapter_sha256: <hex or a prefix of >= 16>   # optional: every copy must have this sha256 of
                                                 # adapter_model.safetensors, or the run refuses
    # FILL_ME (as adapter_sha256 or as a path): a placeholder; the technique variant refuses to run
    # until it is filled in (dry runs pass through and record it)
    merge: false                             # true: merge_and_unload() after the checks (faster
                                             # generation); team-protocol runs only, because
                                             # adapter_off() (public-weights attacks) needs the
                                             # unmerged adapter
    todo: "<why>"                            # marks a model with no adapter yet: the technique
                                             # variant refuses to run (the original variant is
                                             # unaffected: it never uses this defence)
    # train: true + train_config: ...        # train first (scripts.train_fsa)

The Slurm cluster check (inside a job; see scripts/slurm/harness_adapter_check.sbatch):
    python -m experiments._bridges.fsa check --model gemma [--n 4] [--approach fsa|fsa_lite]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.interfaces import Defence, ModelWrapper
from core.io import REPO, ROOT, data_dir
from core.registry import register
from baselines.basic import DelegatingWrapper

# The file / layout / attach checks are shared with the FSA v1 pipeline (scripts/baseline_leakage.py,
# scripts/likelihood_attack.py): src/fsa_lite/fsa/training/peft_check.py, stdlib-only at import time.
# Appended (not prepended) to sys.path, so no harness module can be shadowed by it.
_FSA_SRC = REPO / "src" / "fsa_lite"
if str(_FSA_SRC) not in sys.path:
    sys.path.append(str(_FSA_SRC))
from fsa.training import peft_check as _pc
from fsa.training.peft_check import (  # noqa: F401  (re-exported: tests and callers use F.<name>)
    CONFIG,
    KEY_RE,
    MODULE_RE,
    NEWER_PEFT_DEFAULTS,
    PATHS_FILE,
    PLACEHOLDER,
    REGISTRY_HF_IDS,
    TRAINING_INFO,
    WEIGHTS,
    AdapterError,
    _rewrite_targets,
    _selects,
    _targets,
    adapter_layout,
    check_training_info,
    is_placeholder,
    live_layout,
    peft_config_compat,
    plan,
    safetensors_header,
    sha256_file,
    training_info_format,
    verify_attached,
    write_renamed_safetensors,
)

FSA_DIR = REPO  # the FSA-lite training code (src/fsa_lite, scripts/) lives at the repository root
TRAIN_MODULE = "scripts.train_fsa"  # scripts/train_fsa.py of this repository
ADAPTER_SUBDIR = "adapter"  # fsa.training.trainer.ADAPTER_DIR
# Decoder layers of the protocol models, for the record only (full_adapter in defence_info); the
# expected tensor count comes from each adapter's own target_modules on the live model.
MODEL_LAYERS = {"google/gemma-3-12b-it": 48, "Qwen/Qwen2.5-7B-Instruct": 28}


def remapped_copy(adapter: Path, p: dict[str, Any], out_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Write (or reuse) a copy of the adapter whose keys use the live prefix (default location:
    $HARNESS_DATA/adapters_remapped)."""
    return _pc.remapped_copy(adapter, p, out_root or data_dir() / "adapters_remapped")


# --------------------------------------------------------------------------- training (scripts/train_fsa.py)


def _resolve(p: str | Path) -> Path:
    """Absolute paths (including POSIX ones such as /home/... checked on Windows) are kept;
    relative paths are relative to the repository root."""
    return _pc.resolve_repo_path(p, REPO)


def local_adapter(key: str) -> tuple[Path | None, str]:
    """`adapter: {local: <key>}` -> paths.local.yaml `adapters.<key>` ($FSA_PATHS_FILE overrides the file)."""
    return _pc.local_adapter(key, REPO)


def choose_adapter(cfg: dict[str, Any]) -> tuple[Path | None, str, str | None]:
    """(adapter dir, where it came from, cluster): $FSA_ADAPTER_DIR, else the spec's `adapter`:
    a path, {local: <key>} (paths.local.yaml adapters.<key>), or a {cluster: path} map chosen by
    core.cluster's cluster name ($HARNESS_CLUSTER, from cluster.env)."""
    from core.cluster import cluster_info

    cluster = cluster_info()["cluster"]
    env = os.environ.get("FSA_ADAPTER_DIR")
    if env:
        return _resolve(env), "environment FSA_ADAPTER_DIR", cluster
    raw = cfg.get("adapter")
    if isinstance(raw, dict) and set(raw) == {"local"}:
        path, src = local_adapter(str(raw["local"]))
        return path, src, cluster
    if isinstance(raw, dict):
        if raw.get(cluster):
            return _resolve(raw[cluster]), f"spec adapter.{cluster}", cluster
        return None, f"spec has no adapter for cluster {cluster!r} (has: {sorted(raw)})", cluster
    if raw:
        return _resolve(raw), "spec adapter", cluster
    return None, "no adapter in the spec", cluster


def train_adapter(train_config: str | Path, out_dir: str | Path, device: str = "cuda",
                  max_minutes: float | None = None) -> Path:  # fmt: skip
    """Run FSA training and return the adapter folder (scripts.train_fsa)."""
    if not FSA_DIR.is_dir():
        raise RuntimeError(f"{FSA_DIR} not found: merge branch `amruta` first")
    out_dir = _resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in out_dir.iterdir()}
    cmd = ["uv", "run", "python", "-m", TRAIN_MODULE, "--config", str(_resolve(train_config)),
           "--out-dir", str(out_dir), "--device", device]  # fmt: skip
    if max_minutes:
        cmd += ["--max-minutes", str(max_minutes)]
    subprocess.run(cmd, cwd=FSA_DIR, check=True)
    new = sorted(p for p in out_dir.iterdir() if p.name not in before and p.is_dir())
    if not new:
        raise RuntimeError(f"FSA training wrote no run folder into {out_dir}")
    adapter = new[-1] / ADAPTER_SUBDIR
    if not (adapter / WEIGHTS).exists():
        raise RuntimeError(f"{adapter} has no {WEIGHTS} (training unfinished? use --resume)")
    return adapter


# --------------------------------------------------------------------------- defence


class _DryRunFSA(DelegatingWrapper):
    """Dry-run stand-in: no adapter can be loaded onto the dummy model, so it passes through."""

    def __init__(self, inner: ModelWrapper) -> None:
        super().__init__(inner, "fsa(dry-run)")


def _no_adapter_off() -> Any:
    raise RuntimeError("the FSA adapter was merged into the weights (merge: true): adapter_off() is not available")


@register("defence", "fsa")
class FSABridge(Defence):
    name = "fsa"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        self.adapter, self.adapter_source, self.cluster = choose_adapter(self.cfg)
        self.dry_run_note: str | None = None
        self._attack_study: bool | None = None  # set by prepare(): None = unknown
        self._info: dict[str, Any] = {}

    def prepare(self, items, secrets) -> None:
        # The attack-study runner passes its SecretSource; the team-protocol runner passes None.
        self._attack_study = secrets is not None

    def train(self, cfg: dict[str, Any]) -> None:
        if cfg.get("_dry_run"):
            self.dry_run_note = "training skipped in dry-run"
            return
        if not self.cfg.get("train_config"):
            raise ValueError("fsa defence with train: true needs `train_config`")
        exp = cfg.get("_experiment") or {}
        self.adapter = train_adapter(
            self.cfg["train_config"], self.cfg.get("train_out_dir", "outputs/training"),
            device=(exp.get("model") or {}).get("device", "cuda"), max_minutes=self.cfg.get("max_minutes"),
        )  # fmt: skip
        sys.stdout.write(f"fsa adapter: {self.adapter}\n")

    def info(self) -> dict[str, Any]:
        """Recorded in summary.json (defence_info)."""
        return dict(self._info)

    def wrap(self, model: ModelWrapper) -> ModelWrapper:
        if self.cfg.get("todo"):
            raise RuntimeError(f"fsa: no adapter for this model yet ({self.cfg['todo']}); "
                               "the technique variant cannot run, the original variant can")  # fmt: skip
        merge = bool(self.cfg.get("merge", False))
        if merge and self._attack_study is not False:
            raise RuntimeError("fsa: merge: true is allowed in team-protocol runs only (attack studies need "
                               "adapter_off(), which a merged adapter cannot provide)")  # fmt: skip
        placeholders = [k for k, v in (("adapter", self.adapter), ("adapter_sha256", self.cfg.get("adapter_sha256")))
                        if is_placeholder(v)]  # fmt: skip
        if model.is_dummy:
            self._info = {"dry_run": True, "adapter_dir": self.adapter.as_posix() if self.adapter else None,
                          "adapter_source": self.adapter_source, "cluster": self.cluster,
                          "placeholders": placeholders}  # fmt: skip
            return _DryRunFSA(model)
        if placeholders:
            raise AdapterError(f"fsa: {' and '.join(placeholders)} for this model still {PLACEHOLDER} in the approach spec "
                               f"(cluster {self.cluster}); fill in the adapter path and its sha256 first")  # fmt: skip
        if self.adapter is None:
            raise RuntimeError(f"fsa defence has no adapter here: {self.adapter_source}; set `adapter:` in the "
                               "spec (a path or a per-cluster map), FSA_ADAPTER_DIR, or `train: true`")  # fmt: skip
        if not self.adapter.exists():
            raise RuntimeError(f"FSA adapter not found: {self.adapter}")
        if getattr(model, "load_adapter", None) is None or getattr(model, "load", None) is None:
            raise RuntimeError(f"{model.name} cannot load a LoRA adapter (needs models.hf.HFModel)")
        hf_id = getattr(model, "hf_id", "")
        # The pin is checked on the ORIGINAL weights file (the spec's path or FSA_ADAPTER_DIR), before
        # the model loads and before any remap: a remapped copy has the same tensor bytes but a
        # different header, so its file hash differs by design.
        want_sha = _pc.normalise_pin(self.cfg.get("adapter_sha256"))
        actual_sha = _pc.check_pin(self.adapter, want_sha)
        training = check_training_info(self.adapter, hf_id)
        _, torch_model = model.load()  # type: ignore[attr-defined]
        # The class the base model loaded as, before PEFT wraps it. Under transformers 4.51.3
        # AutoModelForCausalLM may give a text-only Gemma3ForCausalLM ("model.layers.N") for
        # gemma-3-12b-it; live_layout() reads whatever names it has.
        model_class = type(torch_model).__name__
        live = live_layout(torch_model)
        p = plan(self.adapter, live, hf_id, training)
        compat = peft_config_compat(p["config"])
        load_dir, remap_info = self.adapter, None
        if p["remap"]:
            load_dir, remap_info = remapped_copy(self.adapter, p, self.cfg.get("remap_dir") and _resolve(self.cfg["remap_dir"]))
        model.load_adapter(str(load_dir))  # type: ignore[attr-defined]
        _, peft_model = model.load()  # type: ignore[attr-defined]
        counts = verify_attached(peft_model, load_dir / WEIGHTS, p)
        self._info = {
            "adapter_dir": self.adapter.as_posix(), "adapter_sha256": actual_sha,
            "adapter_source": self.adapter_source, "cluster": self.cluster,
            "adapter_sha256_expected": want_sha or None, "adapter_sha256_checked": bool(want_sha),
            "loaded_from": load_dir.as_posix(), "layout": {"adapter_prefix": p["prefix"], "model_prefix": p["live_prefix"]},
            "remapped": p["remap"], "remap": remap_info, "target_modules_rewritten": p["new_target_modules"] != p["target_modules"],
            "r": p["r"], "alpha": p["alpha"], "counts": counts, "merged": False,
            "trained_on": (training or {}).get("hf_id"), "hf_id": hf_id, "model_class": model_class,
            "adapted_layers": p["adapted_layers"], "n_adapted_layers": len(p["adapted_layers"]),
            "n_model_layers": p["n_model_layers"], "full_adapter": p["full_adapter"],
            "expected_tensors": p["expected_tensors"],
            "expected_tensors_from": "2 x the live modules the adapter's target_modules selects",
            "layers_checked_against_training_info": p["layers_checked_against_training_info"],
            "training_info_format": training_info_format(training),
            "trained_revision": (training or {}).get("revision"), "training_variant": (training or {}).get("variant"),
            "peft_config": compat,
        }  # fmt: skip
        if merge:
            model._model = peft_model.merge_and_unload()  # type: ignore[attr-defined]
            model._model.eval()  # type: ignore[attr-defined]
            model.adapter_off = _no_adapter_off  # type: ignore[attr-defined]
            self._info["merged"] = True
        return model


# --------------------------------------------------------------------------- the Slurm cluster check


def cache_status(hub: Path, hf_id: str, revision: str | None) -> dict[str, Any]:
    """Is the snapshot a run would load in the hub cache? Pinned: that revision. Unpinned (Qwen):
    the commit refs/main names, which is what an unpinned offline load resolves to (and what the
    run records as revision_loaded)."""
    base = Path(hub) / f"models--{hf_id.replace('/', '--')}"
    snaps = base / "snapshots"
    n = len([p for p in snaps.iterdir() if p.is_dir()]) if snaps.is_dir() else 0
    if revision:
        return {"what": "pinned", "revision": revision, "cached": (snaps / str(revision)).is_dir(), "n_snapshots": n}
    ref = base / "refs" / "main"
    main = ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    return {"what": "refs/main (unpinned)", "revision": main, "cached": bool(main) and (snaps / main).is_dir(),
            "n_snapshots": n}  # fmt: skip


def check_main(argv: list[str] | None = None) -> int:
    """Load the base model + the spec's adapter through the bridge, print the verification counts,
    and compare greedy replies with and without the adapter on secret_in_context items. Prints
    item keys, counts and identical yes/no only: never secrets, prompts or replies."""
    import argparse

    from core import protocol as P
    from core import registry
    from core.interfaces import GenConfig, SecretContext

    ap = argparse.ArgumentParser(prog="python -m experiments._bridges.fsa check")
    ap.add_argument("--model", default="gemma", choices=sorted(P.MODEL_KEYS))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--benchmark", default="mmlu_team_sic", choices=P.SUITES["secret_in_context"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--approach", default="fsa", help="an approach spec using the fsa defence (fsa, fsa_lite)")
    args = ap.parse_args(argv)
    registry.discover()
    spec = P.load_approach(args.approach)
    if spec["defence"].get("name") != "fsa":
        raise SystemExit(f"approach {args.approach} does not use the fsa defence")
    model_cfg, defence_cfg = P.variant_configs(spec, args.model, "technique")
    original_cfg, _ = P.variant_configs(spec, args.model, "original")
    if original_cfg != model_cfg:  # both variants must load the same model the same way
        raise SystemExit(f"model options differ between variants: {original_cfg} vs {model_cfg}")
    model_cfg["device"] = args.device
    from huggingface_hub import constants as hf_constants

    hub = Path(hf_constants.HF_HUB_CACHE)
    hf_id = getattr(registry.get("model", model_cfg["name"]), "hf_id", "")
    cs = cache_status(hub, hf_id, model_cfg.get("revision"))
    print(f"HF hub cache: {hub} | {hf_id} snapshots there: {cs['n_snapshots']} | revision {cs['what']} "
          f"{str(cs['revision'])[:12]} cached: {cs['cached']} | HF_HUB_OFFLINE={hf_constants.HF_HUB_OFFLINE}")  # fmt: skip
    if hf_constants.HF_HUB_OFFLINE and not cs["cached"]:
        raise SystemExit(f"stop: {hf_id} revision {cs['revision']} is not in {hub} ({cs['what']}) and HF_HUB_OFFLINE "
                         "is on (no download); pass HARNESS_HF_HUB_CACHE=<cache> or HF_HUB_OFFLINE=0")  # fmt: skip
    base = registry.get("model", model_cfg["name"])(model_cfg)
    bridge = registry.get("defence", "fsa")(defence_cfg)
    bridge.prepare([], None)
    model = bridge.wrap(base)
    info = bridge.info()
    print("adapter:", info["adapter_dir"], "| from:", info["adapter_source"], "| cluster:", info["cluster"])
    print("adapter sha256:", info["adapter_sha256"], "| pinned in the spec:", info["adapter_sha256_checked"],
          "| remapped:", info["remapped"],
          "| layout:", info["layout"], "| r/alpha:", info["r"], info["alpha"])  # fmt: skip
    print("counts:", json.dumps(info["counts"]))
    print(f"adapted layers: {info['n_adapted_layers']} of {info['n_model_layers']} {info['adapted_layers']} | "
          f"training_info: {info['training_info_format']} (layers checked: {info['layers_checked_against_training_info']}) | "
          f"peft config keys ignored: {(info['peft_config'] or {}).get('ignored_keys')}")  # fmt: skip
    run = base.run_info() if hasattr(base, "run_info") else {}
    print(f"revision_loaded {run.get('revision_loaded')} dtype {run.get('dtype_loaded')} | adapter trained on revision "
          f"{info.get('trained_revision')} | spec original_from: {spec.get('original_from')}")  # fmt: skip
    print(f"model class loaded (before the adapter): {info['model_class']}; the original variant loads the "
          f"same registry model ({model_cfg['name']}) with the same options, so the same class")  # fmt: skip
    secrets, rec = P.held_out_secrets(spec, fixtures=False)
    print(f"held-out secrets: n={rec['n']} sha256={rec['file_sha256'][:16]}")
    bench = registry.get("benchmark", args.benchmark)({"download": False, "held_out_secrets": secrets}, None)
    items = bench.items()
    order = bench.run_order(items)[: args.n]
    chosen = [items[i] for i in order]
    msgs = [it.messages() for it in chosen]
    ctx = [SecretContext.find(m, it.secret) for m, it in zip(msgs, chosen)]
    cfg = GenConfig(max_new_tokens=bench.max_new_tokens, temperature=0.0)
    with_adapter = model.generate(msgs, cfg, secret_ctx=ctx)
    with model.adapter_off():  # type: ignore[attr-defined]
        without = model.generate(msgs, cfg, secret_ctx=ctx)
    n_same = 0
    for it, a, b in zip(chosen, with_adapter, without):
        same = a.text == b.text
        n_same += same
        print(f"{it.id}: identical={same} tokens with/without adapter={a.n_tokens}/{b.n_tokens}")
    print(f"identical replies: {n_same} of {len(chosen)}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        raise SystemExit(check_main(sys.argv[2:]))
    print("usage: python -m experiments._bridges.fsa check --model gemma [--n 4]", file=sys.stderr)
    raise SystemExit(2)
