"""Checked loading of a PEFT LoRA adapter, shared by the evaluation harness's FSA bridge
(eval/experiments/_bridges/fsa.py) and the FSA v1 pipeline (fsa_runtime.generators.hf, used by
scripts/baseline_leakage.py and scripts/likelihood_attack.py).

Everything here needs only the standard library at import time (torch / safetensors / peft are
imported inside the functions that need them), so the harness, whose pinned stack differs from
the ml group's, imports the same code.

What it checks (the bridge's documented sequence, unchanged):
  1. the sha256 of the ORIGINAL adapter_model.safetensors against a pin, before the model loads
     (check_pin; the pin comes from an approach spec, pin_for_local_key);
  2. adapter_config.json and the safetensors header (no tensors): only decoder-layer LoRA A/B
     weights, one key prefix, no vision keys (adapter_layout);
  3. the live model's decoder-layer prefix and shapes from named_modules() (live_layout), every
     adapter target present with fitting shapes, target_modules selecting exactly the adapter's
     modules, the tensor count 2 x selected modules, training_info.json layers / target_modules
     consistent with the file (plan);
  4. adapter_config.json keys the installed peft does not know only at harmless defaults
     (peft_config_compat);
  5. if the adapter's key prefix differs from the live one, a remapped copy (keys renamed, tensor
     bytes unchanged and sha-checked) is loaded instead (remapped_copy);
  6. after PeftModel.from_pretrained: LoRA attached on exactly the file's modules, 0 missing, 0
     unexpected, every tensor equal to the file, every lora_B non-zero, r / alpha / scaling as in
     the config (verify_attached).
Any failure raises AdapterError; nothing is generated or scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Any

TRAINING_INFO = "training_info.json"  # fsa.training.adapter.TRAINING_INFO
WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
PATHS_FILE = "paths.local.yaml"  # repository root, git-ignored (paths.local.yaml.example)
PLACEHOLDER = "FILL_ME"
# Model keys of the approach specs (configs/eval/*.approach.yaml models.<key>) -> hub ids.
SPEC_MODEL_HF_IDS = {"gemma": "google/gemma-3-12b-it", "qwen": "Qwen/Qwen2.5-7B-Instruct"}
# Registry ids an older training_info.json may name instead of a hub id.
REGISTRY_HF_IDS = {"gemma3-12b-it": "google/gemma-3-12b-it", "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct"}
# LoraConfig fields that peft 0.21.3 has and the pinned peft 0.15.2 does not, at peft 0.21.3's
# defaults. PEFT ignores unknown config keys with a warning; they are accepted only at these
# values (or None / False / empty), so an adapter saved by a newer peft cannot silently lose a
# feature it uses. peft_version is metadata (recorded).
NEWER_PEFT_DEFAULTS = {"alora_invocation_tokens": None, "arrow_config": None, "ensure_weight_tying": False,
                       "kasa_config": None, "lora_ga_config": None, "monteclora_config": None, "qalora_group_size": 16,
                       "target_parameters": None, "use_bdlora": None, "use_qalora": False, "velora_config": None}  # fmt: skip

KEY_RE = re.compile(r"^base_model\.model\.(?P<prefix>(?:[\w]+\.)*?)layers\.(?P<layer>\d+)\.(?P<block>self_attn|mlp)"
                    r"\.(?P<mod>\w+_proj)\.lora_(?P<ab>[AB])(?:\.default)?\.weight$")  # fmt: skip
MODULE_RE = re.compile(r"^(?P<prefix>(?:[\w]+\.)*?)layers\.(?P<layer>\d+)\.(?P<block>self_attn|mlp)\.(?P<mod>\w+_proj)$")
SHA_RE = re.compile(r"[0-9a-f]{16,64}")


class AdapterError(RuntimeError):
    """The adapter does not fit the model, or did not attach completely: refuse to run."""


# --------------------------------------------------------------------------- files


def safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    """(tensors {name: {dtype, shape, data_offsets}}, __metadata__, byte offset of the data)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None) or {}
    return header, meta, 8 + n


def sha256_file(path: Path, start: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_renamed_safetensors(src: Path, dst: Path, rename: dict[str, str]) -> dict[str, str]:
    """Copy `src` to `dst` with tensor names changed by `rename`; dtypes, shapes, offsets and the
    data bytes are unchanged (checked by sha256 of the data section)."""
    header, meta, start = safetensors_header(src)
    new = {rename.get(k, k): v for k, v in header.items()}
    if len(new) != len(header):
        raise AdapterError("renaming would merge two tensors")
    if meta:
        new["__metadata__"] = meta
    blob = json.dumps(new, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)  # safetensors aligns the header to 8 bytes with spaces
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        fi.seek(start)
        shutil.copyfileobj(fi, fo, 1 << 20)
    tmp.replace(dst)
    a, b = sha256_file(src, start), sha256_file(dst, 8 + len(blob))
    if a != b:
        raise AdapterError("remapped tensor data differs from the source")
    return {"data_sha256": a}


# --------------------------------------------------------------------------- pins and local paths


def is_placeholder(v: Any) -> bool:
    return isinstance(v, str | Path) and PLACEHOLDER in str(v)


def normalise_pin(want: str | None) -> str:
    """'' (no pin) or a lower-case sha256 / prefix of >= 16 hex digits; AdapterError otherwise."""
    s = str(want or "").strip().lower()
    if is_placeholder(want):
        raise AdapterError(f"adapter_sha256 is still the placeholder {PLACEHOLDER}; fill in the selected adapter's sha256")
    if s and not SHA_RE.fullmatch(s):
        raise AdapterError(f"adapter_sha256 {s!r} is not a sha256 (or a prefix of >= 16 hex digits)")
    return s


def check_pin(adapter: Path, want_sha: str | None) -> str:
    """sha256 of adapter/adapter_model.safetensors (the ORIGINAL file, never a remapped copy);
    AdapterError if it does not start with the pin. Returns the full sha256."""
    adapter = Path(adapter)
    if not (adapter / WEIGHTS).exists():
        raise AdapterError(f"{adapter}: missing {WEIGHTS}")
    want = normalise_pin(want_sha)
    actual = sha256_file(adapter / WEIGHTS)
    if want and not actual.startswith(want):
        raise AdapterError(f"{adapter / WEIGHTS}: sha256 {actual} does not match adapter_sha256 {want} "
                           "(a different or damaged copy); refusing to load it")  # fmt: skip
    return actual


def resolve_repo_path(p: str | Path, repo: Path) -> Path:
    """Absolute paths (including POSIX ones such as /home/... seen on Windows) are kept; relative
    paths are relative to the repository root."""
    s = str(p)
    if Path(s).is_absolute() or s.startswith("/"):
        return Path(s)
    return Path(repo) / s


def paths_file(repo: Path) -> Path:
    """paths.local.yaml at the repository root; $FSA_PATHS_FILE overrides it."""
    return Path(os.environ.get("FSA_PATHS_FILE") or (Path(repo) / PATHS_FILE))


def local_adapter(key: str, repo: Path) -> tuple[Path | None, str]:
    """paths.local.yaml `adapters.<key>` -> (path, where it came from); (None, why) if absent."""
    import yaml

    f = paths_file(repo)
    if not f.is_file():
        return None, f"adapter {{local: {key}}}: {f.name} not found (copy paths.local.yaml.example)"
    val = ((yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("adapters") or {}).get(key)
    if not val:
        return None, f"adapter {{local: {key}}}: no adapters.{key} in {f.name}"
    return resolve_repo_path(val, repo), f"{f.name} adapters.{key}"


def pin_for_local_key(spec_path: Path, key: str) -> dict[str, Any]:
    """The approach spec's entry whose defence adapter is `{local: <key>}`:
    {"model": <spec model key>, "adapter_sha256": <pin>, "spec": <file name>}. AdapterError if no
    model of the spec uses that key, or its pin is missing / a placeholder / not a sha256."""
    import yaml

    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8")) or {}
    hits = []
    for name, m in (spec.get("models") or {}).items():
        d = (m or {}).get("defence") or {}
        if d.get("adapter") == {"local": key}:
            hits.append((name, d))
    if len(hits) != 1:
        raise AdapterError(f"{Path(spec_path).name}: {len(hits)} models use adapter {{local: {key}}} (need exactly 1)")
    name, d = hits[0]
    raw = d.get("adapter_sha256")
    if not raw:
        raise AdapterError(f"{Path(spec_path).name} models.{name}.defence has no adapter_sha256 for {key}")
    return {"model": name, "adapter_sha256": normalise_pin(raw), "spec": Path(spec_path).name}


def used_secrets_overlap(values: list[str], used_file: Path) -> dict[str, Any]:
    """How many of `values` occur, as whole lines (stripped), in a used-secrets file. Counts and
    the file's sha256 only: no value is returned or printed."""
    used_file = Path(used_file)
    if not used_file.is_file():
        return {"available": False, "file": used_file.name}
    raw = used_file.read_bytes()
    used = {x.strip() for x in raw.decode("utf-8").splitlines() if x.strip()}
    return {"available": True, "file": used_file.name, "file_sha256": hashlib.sha256(raw).hexdigest(),
            "n_used_values": len(used), "n_values": len(values),
            "n_in_used": sum(1 for v in values if str(v).strip() in used),
            "match": "exact whole line after strip()"}  # fmt: skip


# --------------------------------------------------------------------------- layouts


def adapter_layout(adapter: Path) -> dict[str, Any]:
    """Parse adapter_config.json and the weight names. Raises AdapterError on keys that are not
    decoder-layer LoRA weights, on vision keys, on more than one prefix, or a missing A/B."""
    for f in (CONFIG, WEIGHTS):
        if not (adapter / f).exists():
            raise AdapterError(f"{adapter}: missing {f}")
    cfg = json.loads((adapter / CONFIG).read_text(encoding="utf-8"))
    header, _, _ = safetensors_header(adapter / WEIGHTS)
    errs, prefixes, tensors = [], set(), {}
    for key, v in sorted(header.items()):
        m = KEY_RE.match(key)
        if not m or "vision" in key:
            errs.append(f"unexpected key {key}")
            continue
        prefixes.add(m["prefix"])
        tensors[(int(m["layer"]), m["block"], m["mod"], m["ab"])] = tuple(v["shape"])
    if len(prefixes) > 1:
        errs.append(f"keys use {len(prefixes)} different prefixes: {sorted(prefixes)}")
    modules = sorted({k[:3] for k in tensors})
    for mod in modules:
        for ab in "AB":
            if (*mod, ab) not in tensors:
                errs.append(f"layer {mod[0]} {mod[1]}.{mod[2]} has no lora_{ab}")
    if errs:
        raise AdapterError(f"{adapter / WEIGHTS}: " + "; ".join(errs[:8]) + (" ..." if len(errs) > 8 else ""))
    if cfg.get("peft_type", "LORA") != "LORA":
        raise AdapterError(f"peft_type {cfg.get('peft_type')!r}: only LoRA is supported")
    return {"prefix": next(iter(prefixes)) if prefixes else "", "tensors": tensors, "modules": modules,
            "n_tensors": len(tensors), "r": cfg.get("r"), "alpha": cfg.get("lora_alpha"),
            "use_rslora": bool(cfg.get("use_rslora")), "use_dora": bool(cfg.get("use_dora")),
            "target_modules": cfg.get("target_modules"), "base_model": cfg.get("base_model_name_or_path"),
            "config": cfg}  # fmt: skip


def live_layout(torch_model: Any) -> dict[str, Any]:
    """Decoder-layer prefix and {(layer, block, mod): (out, in)} of the live model, from
    named_modules(); vision-tower modules are skipped."""
    prefixes, shapes, names = set(), {}, {}
    for name, module in torch_model.named_modules():
        if "vision" in name:
            continue
        m = MODULE_RE.match(name)
        if not m or not hasattr(module, "weight") or getattr(module.weight, "ndim", 0) != 2:
            continue
        prefixes.add(m["prefix"])
        key = (int(m["layer"]), m["block"], m["mod"])
        shapes[key] = tuple(module.weight.shape)
        names[key] = name
    if len(prefixes) != 1:
        raise AdapterError(f"cannot identify the decoder layers of the live model (prefixes {sorted(prefixes)})")
    return {"prefix": next(iter(prefixes)), "shapes": shapes, "names": names}


def _targets(target_modules: Any, names: dict[tuple, str]) -> set[tuple]:
    """The live modules PEFT would wrap for this target_modules (full-match regex, or a list of
    module-name suffixes), vision excluded."""
    out = set()
    for key, name in names.items():
        if isinstance(target_modules, str):
            hit = re.fullmatch(target_modules, name) is not None
        else:
            hit = any(name == t or name.endswith("." + t) for t in (target_modules or []))
        if hit:
            out.add(key)
    return out


def _rewrite_targets(target_modules: Any, src: str, dst: str) -> Any:
    """Replace the old prefix in target_modules, only where it is named literally."""
    if not src or src == dst:
        return target_modules
    s, d = src.rstrip("."), dst.rstrip(".")
    if isinstance(target_modules, str):  # escaped form first ("a\.b"), then a plain one
        return target_modules.replace(re.escape(s), re.escape(d)).replace(s, d)
    return [d + t[len(s):] if t.startswith(s + ".") or t == s else t for t in (target_modules or [])]


def _selects(target_modules: Any, lay: dict[str, Any], live: dict[str, Any]) -> tuple[Any, set[tuple]]:
    """(target_modules as usable on the live model, the live modules it selects): as given if that
    already selects the adapter's modules (FSA's regex is path-agnostic), else with the adapter's
    prefix replaced by the live one."""
    selected = _targets(target_modules, live["names"])
    if set(lay["modules"]) != selected:
        rewritten = _rewrite_targets(target_modules, lay["prefix"], live["prefix"])
        return rewritten, _targets(rewritten, live["names"])
    return target_modules, selected


def plan(adapter: Path, live: dict[str, Any], hf_id: str = "", training: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every check that needs no PEFT: targets exist in the live model, shapes fit, the adapter
    covers every module its target_modules selects, the expected count (2 x the selected modules),
    and, given training_info.json, its layers and target_modules agree with the adapter. Raises
    AdapterError."""
    lay = adapter_layout(adapter)
    errs, r = [], lay["r"]
    absent = sorted({k[:3] for k in lay["tensors"] if k[:3] not in live["shapes"]})
    if absent:
        layer, block, mod = absent[0]
        errs.append(f"{len(absent)} adapter modules do not exist in the model, e.g. layers.{layer}.{block}.{mod} "
                    f"(model has {len({k[0] for k in live['shapes']})} layers; wrong base model?)")  # fmt: skip
    for (layer, block, mod, ab), shape in lay["tensors"].items():
        live_shape = live["shapes"].get((layer, block, mod))
        if live_shape is None:
            continue
        out_f, in_f = live_shape
        want = (r, in_f) if ab == "A" else (out_f, r)
        if shape != want:
            errs.append(f"layers.{layer}.{block}.{mod}.lora_{ab} shape {list(shape)}, model needs {list(want)}")
    new_targets, selected = _selects(lay["target_modules"], lay, live)
    if set(lay["modules"]) != selected:
        missing = sorted(selected - set(lay["modules"]))
        extra = sorted(set(lay["modules"]) - selected)
        errs.append(f"target_modules selects {len(selected)} modules, the adapter has {len(lay['modules'])} "
                    f"(no weights for {len(missing)}, e.g. {missing[:3]}; not selected {len(extra)}, e.g. {extra[:3]})")  # fmt: skip
    # The expected count is the adapter's own: A and B on every module its target_modules selects.
    expected = 2 * len(selected)
    if lay["n_tensors"] != expected:
        errs.append(f"{lay['n_tensors']} LoRA tensors in the file, its target_modules select {len(selected)} modules "
                    f"(expected {expected})")  # fmt: skip
    adapted = sorted({k[0] for k in lay["modules"]})
    model_layers = sorted({k[0] for k in live["shapes"]})
    info_layers, info_targets = (training or {}).get("layers"), (training or {}).get("target_modules")
    if info_layers is not None:
        if not isinstance(info_layers, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in info_layers):
            errs.append(f"training_info.json layers {info_layers!r} is not a list of layer indices")
        elif sorted(set(info_layers)) != adapted:
            errs.append(f"training_info.json layers {sorted(set(info_layers))} differ from the layers the adapter "
                        f"holds {adapted}")  # fmt: skip
    if info_targets is not None and _selects(info_targets, lay, live)[1] != set(lay["modules"]):
        errs.append("training_info.json target_modules select other modules than the adapter holds")
    if errs:
        raise AdapterError(f"FSA adapter {adapter} does not fit the model: " + "; ".join(errs[:8])
                           + (f" (+{len(errs) - 8} more)" if len(errs) > 8 else ""))  # fmt: skip
    return {**lay, "live_prefix": live["prefix"], "new_target_modules": new_targets,
            "remap": lay["prefix"] != live["prefix"], "expected_tensors": expected, "adapted_layers": adapted,
            "n_model_layers": len(model_layers), "full_adapter": adapted == model_layers,
            "layers_checked_against_training_info": info_layers is not None,
            "target_modules_checked_against_training_info": info_targets is not None}  # fmt: skip


def peft_config_compat(cfg: dict[str, Any]) -> dict[str, Any]:
    """adapter_config.json keys the installed peft's LoraConfig lacks (saved by a newer peft): PEFT
    ignores them with a warning, so each must hold a harmless default (NEWER_PEFT_DEFAULTS, or
    None / False / empty); otherwise AdapterError. Returns what was ignored, for the record."""
    import dataclasses

    try:
        import peft
        from peft import LoraConfig
    except ImportError:
        return {"checked": False, "note": "peft not installed"}
    known = {f.name for f in dataclasses.fields(LoraConfig)}
    unknown = sorted(set(cfg) - known)

    def harmless(k: str, v: Any) -> bool:
        if k == "peft_version" or v is None or v is False or v == [] or v == {} or v == "":
            return True
        return k in NEWER_PEFT_DEFAULTS and v == NEWER_PEFT_DEFAULTS[k]

    bad = {k: cfg[k] for k in unknown if not harmless(k, cfg[k])}
    if bad:
        raise AdapterError(f"adapter_config.json uses {bad}: settings the installed peft {peft.__version__} does not "
                           "have (it would ignore them and load a different adapter); refusing")  # fmt: skip
    return {"checked": True, "peft": peft.__version__, "ignored_keys": unknown, "saved_by_peft": cfg.get("peft_version")}


def remapped_copy(adapter: Path, p: dict[str, Any], out_root: Path) -> tuple[Path, dict[str, Any]]:
    """Write (or reuse) a copy of the adapter under `out_root` whose keys use the live prefix."""
    src_sha = sha256_file(adapter / WEIGHTS)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", p["live_prefix"]).strip("_") or "root"
    dst = Path(out_root) / f"{src_sha[:16]}-{slug}"
    header, _, _ = safetensors_header(adapter / WEIGHTS)
    old, new = "base_model.model." + p["prefix"], "base_model.model." + p["live_prefix"]
    rename = {k: new + k[len(old):] for k in header if k.startswith(old)}
    marker = dst / "remap_info.json"
    info = {"source_dir": str(adapter), "source_sha256": src_sha, "from_prefix": p["prefix"],
            "to_prefix": p["live_prefix"], "n_renamed": len(rename)}  # fmt: skip
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("source_sha256") == src_sha:
        prev = json.loads(marker.read_text(encoding="utf-8"))
        return dst, prev
    data = write_renamed_safetensors(adapter / WEIGHTS, dst / WEIGHTS, rename)
    cfg = dict(p["config"])
    cfg["target_modules"] = p["new_target_modules"]
    (dst / CONFIG).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
    for extra in (TRAINING_INFO, "README.md"):
        if (adapter / extra).exists():
            shutil.copy2(adapter / extra, dst / extra)
    info.update(data)
    marker.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    return dst, info


def verify_attached(peft_model: Any, weights: Path, p: dict[str, Any]) -> dict[str, Any]:
    """After PeftModel.from_pretrained: the live LoRA modules equal the file, completely."""
    import torch
    from safetensors.torch import load_file

    file = load_file(str(weights))
    live = {n: m for n, m in peft_model.named_modules()
            if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})}  # fmt: skip
    errs: list[str] = []
    vision = [n for n in live if "vision" in n]
    if vision:
        errs.append(f"LoRA on {len(vision)} vision modules, e.g. {vision[0]}")
    seen, zero_b, unequal = set(), 0, 0
    for key, t in file.items():
        mod_path, ab = key.rsplit(".lora_", 1)
        ab = ab[0]
        m = live.get(mod_path)
        if m is None:
            continue
        seen.add(mod_path)
        w = (m.lora_A if ab == "A" else m.lora_B)["default"].weight.detach()
        if tuple(w.shape) != tuple(t.shape) or not torch.equal(w.to("cpu", t.dtype), t):
            unequal += 1
        if ab == "B" and not bool(t.abs().sum() > 0):
            zero_b += 1
    file_mods = {k.rsplit(".lora_", 1)[0] for k in file}
    missing, unexpected = sorted(file_mods - seen), sorted(set(live) - file_mods)
    bad_r = [n for n in seen if live[n].r["default"] != p["r"] or float(live[n].lora_alpha["default"]) != float(p["alpha"])]
    scaling = float(p["alpha"]) / (p["r"] ** 0.5 if p["use_rslora"] else p["r"])
    bad_s = [n for n in seen if abs(float(live[n].scaling["default"]) - scaling) > 1e-9]
    n_a = sum(1 for n in seen)
    counts = {"lora_A": n_a, "lora_B": n_a, "tensors_attached": 2 * n_a, "tensors_in_file": len(file),
              "expected": p.get("expected_tensors"), "missing": len(missing), "unexpected": len(unexpected),
              "zero_lora_B": zero_b, "not_equal_to_file": unequal, "wrong_r_or_alpha": len(bad_r) + len(bad_s)}  # fmt: skip
    if missing:
        errs.append(f"{len(missing)} adapter modules not attached, e.g. {missing[:2]}")
    if unexpected:
        errs.append(f"{len(unexpected)} LoRA modules not in the file, e.g. {unexpected[:2]}")
    if zero_b:
        errs.append(f"{zero_b} lora_B tensors are all zero (untrained adapter?)")
    if unequal:
        errs.append(f"{unequal} attached tensors differ from the file")
    if bad_r or bad_s:
        errs.append(f"r / alpha / scaling differ from the config on {len(bad_r) + len(bad_s)} modules")
    if counts["expected"] is not None and counts["tensors_attached"] != counts["expected"]:
        errs.append(f"{counts['tensors_attached']} LoRA tensors attached, {counts['expected']} expected")
    if 2 * n_a != len(file):
        errs.append(f"{2 * n_a} LoRA tensors attached, {len(file)} in the file")
    if errs:
        raise AdapterError("FSA adapter did not attach correctly: " + "; ".join(errs) + f" (counts {counts})")
    return counts


# --------------------------------------------------------------------------- training_info.json


def training_info_format(info: dict[str, Any] | None) -> str | None:
    """Which trainer of this repository wrote training_info.json."""
    if info is None:
        return None
    if "model_key" in info or str(info.get("variant", "")).startswith(("fsa_lite", "fsa_v2")):
        return "v2 (src/fsa_lite/lite/v2_train.py)"
    return "v1 (scripts.train_fsa)"


def check_training_info(adapter: Path, hf_id: str) -> dict[str, Any] | None:
    """Read training_info.json (either trainer's format) and refuse an adapter trained on another
    base model. The base model is `hf_id` (both trainers), else `model.hf_id`, else a registry id in
    `base_model` (v1). `layers` / `target_modules` are checked against the adapter by plan()."""
    info_path = Path(adapter) / TRAINING_INFO
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise AdapterError(f"{info_path} is not a JSON object")
    trained_on = (info.get("hf_id") or (info.get("model") or {}).get("hf_id")
                  or REGISTRY_HF_IDS.get(str(info.get("base_model") or "")))  # fmt: skip
    if trained_on and hf_id and trained_on != hf_id:
        raise AdapterError(f"FSA adapter {adapter} was trained on {trained_on}, not {hf_id}")
    return {**info, "hf_id": trained_on or info.get("hf_id")}


# --------------------------------------------------------------------------- one call


def load_verified(base_model: Any, adapter: Path, *, hf_id: str, remap_root: Path,
                  want_sha: str | None = None) -> tuple[Any, dict[str, Any]]:  # fmt: skip
    """The pin (again: the caller should already have checked it before loading the base model),
    training_info.json, the plan, peft config compatibility, a remapped copy when the key prefix
    differs, PeftModel.from_pretrained and verify_attached. Returns (peft model, record)."""
    from peft import PeftModel

    adapter = Path(adapter)
    actual = check_pin(adapter, want_sha)
    training = check_training_info(adapter, hf_id)
    p = plan(adapter, live_layout(base_model), hf_id, training)
    compat = peft_config_compat(p["config"])
    load_dir, remap_info = adapter, None
    if p["remap"]:
        load_dir, remap_info = remapped_copy(adapter, p, remap_root)
    model = PeftModel.from_pretrained(base_model, str(load_dir))
    counts = verify_attached(model, load_dir / WEIGHTS, p)
    record = {"adapter_sha256": actual, "adapter_sha256_expected": normalise_pin(want_sha) or None,
              "adapter_sha256_checked": bool(normalise_pin(want_sha)), "remapped": p["remap"], "remap": remap_info,
              "layout": {"adapter_prefix": p["prefix"], "model_prefix": p["live_prefix"]},
              "r": p["r"], "alpha": p["alpha"], "counts": counts, "adapted_layers": p["adapted_layers"],
              "n_model_layers": p["n_model_layers"], "full_adapter": p["full_adapter"],
              "training_info_format": training_info_format(training),
              "trained_revision": (training or {}).get("revision"), "peft_config": compat,
              "model_class": type(base_model).__name__}  # fmt: skip
    return model, record
