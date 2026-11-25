"""FSA bridge (eval/experiments/_bridges/fsa.py): adapter layout detection from the live model, key
remapping with unchanged tensor bytes, the refusals, and (with peft installed) loading through
PEFT on tiny random Gemma 3 / Qwen2 with the full post-load verification.

The tiny models here are built by the installed transformers, so an adapter saved on them would
already match their module names. The remap is therefore exercised explicitly: the adapter's keys
are written under the OTHER layout and the bridge must rename them.
  * Gemma 3: the adapter uses "language_model.model.layers.N" (Gemma3ForConditionalGeneration
    under transformers 4.51.3); the tiny live model (Gemma3ForCausalLM) uses "model.layers.N".
  * Qwen2: the adapter uses "model.language_model.layers.N" (the nesting 5.x gives multimodal
    wrappers). A real Qwen2.5 adapter has the same keys on 4.51.3 and 5.x; this case only checks
    that the rename is generic, not Gemma-specific.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from experiments._bridges import fsa as F  # noqa: E402

MODS = ("q_proj", "k_proj", "v_proj", "o_proj")


def fsa_regex(layers: list[int]) -> str:
    """The target_modules regex FSA's trainer saves (src/fsa_lite/fsa/training/lora.py)."""
    alt = "|".join(str(i) for i in layers)
    return rf"^(?!.*vision).*\.layers\.({alt})\.(self_attn|mlp)\.({'|'.join(MODS)})$"


def tiny(name: str):
    from analysis.internals.tiny import TINY

    return TINY[name]({"device": "cpu", "dtype": "float32", "batch_size": 2})


def make_adapter(d: Path, live: dict, prefix: str, r: int = 4, alpha: int = 8, drop: int = 0,
                 zero_b: bool = False, target=None, seed: int = 0, only_layers: list[int] | None = None,
                 extra_config: dict | None = None) -> Path:  # fmt: skip
    """A PEFT-format LoRA adapter for the live model's attention projections, keys under `prefix`
    (only_layers: a layer-restricted adapter, as FSA's lora.layers: [5, 11, 17])."""
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    layers = only_layers if only_layers is not None else sorted({k[0] for k in live["shapes"]})
    for (layer, block, mod), (out_f, in_f) in sorted(live["shapes"].items()):
        if block != "self_attn" or mod not in MODS or layer not in layers:
            continue
        base = f"base_model.model.{prefix}layers.{layer}.{block}.{mod}"
        tensors[f"{base}.lora_A.weight"] = torch.randn(r, in_f, generator=g) * 0.5
        tensors[f"{base}.lora_B.weight"] = torch.zeros(out_f, r) if zero_b else torch.randn(out_f, r, generator=g) * 0.5
    for k in sorted(tensors)[:drop]:
        del tensors[k]
    d.mkdir(parents=True, exist_ok=True)
    st.save_file(tensors, str(d / F.WEIGHTS))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": alpha, "lora_dropout": 0.0,
           "bias": "none", "use_rslora": False, "use_dora": False, "base_model_name_or_path": "tiny",
           "target_modules": target if target is not None else fsa_regex(layers), "inference_mode": True,
           **(extra_config or {})}  # fmt: skip
    (d / F.CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    return d


def write_training_info(d: Path, fmt: str, hf_id: str, layers: list[int], **over) -> dict:
    """training_info.json as each branch-amruta trainer writes it.
    v2: src/fsa_lite/lite/v2_train.py at 1b75a3c (info dict + src.info() + step);
    v1: fsa.training.trainer.Trainer.save_adapter (self.info + method/target/k/layers/target_modules)."""
    if fmt == "v2":
        info = {"hf_id": hf_id, "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80", "revision_configured": None,
                "model_key": "gemma", "dtype": "bfloat16", "method": "fsa", "target": "mixture", "k": 4, "need_weight": 1.0,
                "layers": layers, "target_modules": fsa_regex(layers), "variant": "fsa_lite", "data_source": "lite",
                "turn_type_weights": {"tt_attack": 0.4}, "data_units_sha256": "0" * 64,
                "cimem_label_source": "domain_heuristic", "step": 400, "v2_meta": "v2_meta.json"}  # fmt: skip
    else:
        info = {"run_id": "20261113-165343_fsa_gemma12b_pe4_fullmix", "base_model": "gemma3-12b-it", "hf_id": hf_id,
                "method": "fsa", "target": "mixture", "reply_source": "offline", "k": 49, "layers": layers,
                "target_modules": fsa_regex(layers), "step": 500}  # fmt: skip
    info.update(over)
    (d / F.TRAINING_INFO).write_text(json.dumps(info), encoding="utf-8")
    return info


class _Wrapper(torch.nn.Module):
    """Gemma3ForConditionalGeneration-like nesting: language_model.model.layers + a vision tower
    that also has q_proj / k_proj (which must be ignored)."""

    def __init__(self, lm):
        super().__init__()
        self.language_model = lm
        self.vision_tower = torch.nn.ModuleDict({"layers": torch.nn.ModuleList([torch.nn.ModuleDict(
            {"self_attn": torch.nn.ModuleDict({"q_proj": torch.nn.Linear(4, 4)})})])})  # fmt: skip


# --------------------------------------------------------------------------- no PEFT needed


def test_resolve_keeps_absolute_paths():
    cluster_path = "/home/user/fsa-lite/outputs/training/x/adapter"
    assert str(F._resolve(cluster_path)).replace("\\", "/") == cluster_path
    assert F._resolve("experiments/x") == F.REPO / "experiments/x"
    assert F._resolve(F.ROOT / "a") == F.ROOT / "a"


def test_live_layout_is_read_from_the_model_and_ignores_vision():
    _, g = tiny("tiny-gemma3").load()
    lay = F.live_layout(g)
    assert lay["prefix"] == "model." and lay["names"][(0, "self_attn", "q_proj")] == "model.layers.0.self_attn.q_proj"
    wrapped = F.live_layout(_Wrapper(g))
    assert wrapped["prefix"] == "language_model.model."
    assert not any("vision" in n for n in wrapped["names"].values())
    _, q = tiny("tiny-qwen2").load()
    assert F.live_layout(q)["prefix"] == "model."


def test_rename_keeps_tensor_bytes(tmp_path):
    t = {"base_model.model.a.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(2, 3).to(torch.bfloat16),
         "base_model.model.a.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(3, 2)}  # fmt: skip
    st.save_file(t, str(tmp_path / "in.safetensors"), metadata={"format": "pt"})
    ren = {k: k.replace(".a.", ".b.c.") for k in t}
    out = F.write_renamed_safetensors(tmp_path / "in.safetensors", tmp_path / "out.safetensors", ren)
    back = st.load_file(str(tmp_path / "out.safetensors"))
    assert set(back) == set(ren.values())
    for k, v in t.items():
        assert torch.equal(back[ren[k]], v) and back[ren[k]].dtype == v.dtype
    _, meta, start = F.safetensors_header(tmp_path / "out.safetensors")
    assert meta == {"format": "pt"} and start % 8 == 0
    assert out["data_sha256"] == F.sha256_file(tmp_path / "out.safetensors", start)


@pytest.mark.parametrize("name, adapter_prefix", [("tiny-gemma3", "language_model.model."),
                                                  ("tiny-qwen2", "model.language_model.")])  # fmt: skip
def test_plan_and_remapped_copy(tmp_path, name, adapter_prefix):
    _, m = tiny(name).load()
    live = F.live_layout(m)
    ad = make_adapter(tmp_path / "ad", live, adapter_prefix)
    p = F.plan(ad, live)
    assert p["remap"] and p["prefix"] == adapter_prefix and p["live_prefix"] == "model."
    assert p["new_target_modules"] == p["target_modules"]  # FSA's regex is path-agnostic: kept
    dst, info = F.remapped_copy(ad, p, tmp_path / "remapped")
    keys = set(F.safetensors_header(dst / F.WEIGHTS)[0])
    assert keys and all(k.startswith("base_model.model.model.layers.") for k in keys)
    assert info["n_renamed"] == p["n_tensors"] == len(keys)
    src_start, dst_start = F.safetensors_header(ad / F.WEIGHTS)[2], F.safetensors_header(dst / F.WEIGHTS)[2]
    assert F.sha256_file(ad / F.WEIGHTS, src_start) == F.sha256_file(dst / F.WEIGHTS, dst_start)  # same bytes
    assert F.remapped_copy(ad, p, tmp_path / "remapped")[0] == dst  # reused, not rewritten
    # matching layout: nothing to remap
    same = make_adapter(tmp_path / "same", live, "model.")
    assert F.plan(same, live)["remap"] is False


def test_target_modules_naming_the_old_path_is_rewritten(tmp_path):
    _, m = tiny("tiny-gemma3").load()
    live = F.live_layout(m)
    n = len({k[0] for k in live["shapes"]})
    path_regex = rf"^language_model\.model\.layers\.({'|'.join(map(str, range(n)))})\.self_attn\.({'|'.join(MODS)})$"
    p = F.plan(make_adapter(tmp_path / "a", live, "language_model.model.", target=path_regex), live)
    assert p["new_target_modules"].startswith(r"^model\.layers\.")
    names = [f"language_model.model.layers.{i}.self_attn.{mod}" for i in range(n) for mod in MODS]
    p2 = F.plan(make_adapter(tmp_path / "b", live, "language_model.model.", target=names), live)
    assert all(t.startswith("model.layers.") for t in p2["new_target_modules"])


def test_partial_and_wrong_model_adapters_are_refused(tmp_path):
    _, g = tiny("tiny-gemma3").load()
    _, q = tiny("tiny-qwen2").load()
    lg, lq = F.live_layout(g), F.live_layout(q)
    partial = make_adapter(tmp_path / "partial", lg, "language_model.model.", drop=2)  # one module's A and B gone
    with pytest.raises(F.AdapterError, match="target_modules selects"):
        F.plan(partial, lg)
    half = make_adapter(tmp_path / "half", lg, "model.", drop=1)  # a lora_A without its lora_B
    with pytest.raises(F.AdapterError, match="has no lora_A"):
        F.plan(half, lg)
    # The two tiny models share attention shapes; Gemma has 6 layers, Qwen2 4.
    wrong = make_adapter(tmp_path / "wrong", lg, "model.")  # made for Gemma 3, loaded on Qwen2
    with pytest.raises(F.AdapterError, match=r"8 adapter modules do not exist in the model, e.g. layers\.4\..*model has 4 layers"):
        F.plan(wrong, lq)
    wide = {**lg, "shapes": {k: (2 * o, 2 * i) for k, (o, i) in lg["shapes"].items()}}  # another hidden size
    with pytest.raises(F.AdapterError, match="shape .* model needs"):
        F.plan(make_adapter(tmp_path / "wide", wide, "model."), lg)
    full = make_adapter(tmp_path / "full", lg, "model.")
    # the expected count is the adapter's own (2 x selected modules), not a fixed number per model
    assert F.plan(full, lg, hf_id="google/gemma-3-12b-it")["expected_tensors"] == F.adapter_layout(full)["n_tensors"]
    # training_info.json must agree with the adapter: its layers ...
    with pytest.raises(F.AdapterError, match=r"training_info.json layers \[0, 1\] differ from the layers the adapter holds \[0, 1, 2"):
        F.plan(full, lg, training={"layers": [0, 1]})
    with pytest.raises(F.AdapterError, match="not a list of layer indices"):
        F.plan(full, lg, training={"layers": "all"})
    # ... and its target_modules
    with pytest.raises(F.AdapterError, match="training_info.json target_modules select other modules"):
        F.plan(full, lg, training={"target_modules": fsa_regex([0])})
    # a vision key or a non-LoRA key is refused before anything is loaded
    # (written to a new folder: on Windows a file still memory-mapped by load_file cannot be
    # overwritten, which safetensors 0.5.3 reports as "user-mapped section open")
    t = {k: v.clone() for k, v in st.load_file(str(full / F.WEIGHTS)).items()}
    t["base_model.model.vision_tower.layers.0.self_attn.q_proj.lora_A.weight"] = torch.ones(4, 4)
    vis = tmp_path / "vision"
    vis.mkdir()
    shutil.copy(full / F.CONFIG, vis / F.CONFIG)
    st.save_file(t, str(vis / F.WEIGHTS))
    with pytest.raises(F.AdapterError, match="unexpected key"):
        F.plan(vis, lg)


def test_wrap_refusals_need_no_model_load(tmp_path):
    from models.dummy import DummyModel

    with pytest.raises(RuntimeError, match="no adapter for this model yet"):
        F.FSABridge({"todo": "Qwen FSA adapter path not set"}).wrap(DummyModel())
    b = F.FSABridge({"adapter": str(tmp_path), "merge": True})
    with pytest.raises(RuntimeError, match="team-protocol runs only"):
        b.wrap(DummyModel())  # prepare() never called: mode unknown, merge refused
    b.prepare([], object())  # the attack-study runner passes a SecretSource
    with pytest.raises(RuntimeError, match="team-protocol runs only"):
        b.wrap(DummyModel())
    b.prepare([], None)  # the protocol runner passes None
    assert b.wrap(DummyModel()).name.endswith("fsa(dry-run)")


def test_spec_and_private_files(monkeypatch, tmp_path):
    from core import protocol as P

    spec = P.load_approach("fsa")
    _, gemma = P.variant_configs(spec, "gemma", "technique")
    assert gemma["adapter"] == {"local": "gemma_fsa0"}  # paths.local.yaml adapters.gemma_fsa0
    monkeypatch.delenv("FSA_ADAPTER_DIR", raising=False)
    monkeypatch.setenv("FSA_PATHS_FILE", str(tmp_path / "paths.local.yaml"))
    (tmp_path / "paths.local.yaml").write_text("adapters:\n  gemma_fsa0: /data/gemma_fsa0\n", encoding="utf-8")
    b = F.FSABridge(gemma)
    assert b.adapter.as_posix() == "/data/gemma_fsa0" and b.adapter_source == "paths.local.yaml adapters.gemma_fsa0"
    for key in ("gemma", "qwen"):  # both variants load the same registry model with the same options
        assert P.variant_configs(spec, key, "original")[0] == P.variant_configs(spec, key, "technique")[0]
    _, qwen = P.variant_configs(spec, "qwen", "technique")
    assert not qwen.get("todo") and F.is_placeholder(qwen["adapter_sha256"])
    assert qwen["adapter"] == {"local": "qwen_fsa0"}
    assert P.variant_configs(spec, "qwen", "original")[1] == {"name": "none"}
    assert "revision" not in P.variant_configs(spec, "qwen", "original")[0]  # no Qwen pin
    priv = Path(spec["_path"]).parent / "private"
    if not (priv / "held_out_secrets.txt").exists():
        pytest.skip("private secrets are local only (git-ignored)")
    secrets, rec = P.held_out_secrets(spec, fixtures=False)  # disjointness and git-ignore checked inside
    used = set((priv / "used_secrets.txt").read_text(encoding="utf-8").split())
    assert rec["n"] == rec["n_distinct"] == 300 and len(used) == 100 and not used & set(secrets)
    assert rec["git_ignored"] in (True, None)


def test_fsa_lite_spec_uses_the_v2_run_folder_and_placeholders():
    from core import protocol as P

    spec = P.load_approach("fsa_lite")
    assert spec["name"] == "fsa_lite" and spec["original_from"] == "fsa" and spec["defence"]["name"] == "fsa"
    _, gemma = P.variant_configs(spec, "gemma", "technique")
    # src/fsa_lite/lite/v2_train.py: save_adapter(model, run / "adapter", ...) with run = --run-dir;
    # training_info.json is written inside that folder; paths.local.yaml says where it is
    assert gemma["adapter"] == {"local": "gemma_fsa_lite"} and not F.is_placeholder(gemma["adapter_sha256"])
    # fsa_lite_gemma12b_run2 (48 layers, q/k/v/o), the final Gemma run2 adapter
    assert gemma["adapter_sha256"] == "7828b15337419950cd3dbe92f308b5b662d3ea410c3cdad0d681561f3207db75"
    _, qwen = P.variant_configs(spec, "qwen", "technique")
    assert qwen["adapter_sha256"] == "850df8a92bc50cfc7fb1127d49eaac617daba8f83713ae86e471f3f60a06dc3d"
    assert qwen["adapter"] == {"local": "qwen_fsa_lite_run2"}
    # run3: its own spec, the pinned selected adapters (selection.json adapter_sha256)
    r3 = P.load_approach("fsa_lite_run3")
    for key, sha in (("gemma", "324d8d8dc3659ffc45c12a38d5dc18e20e529b1b30bb3083100a227ce9f45d15"), ("qwen", "d5f8ca21ce5751a7e585cf9e8b3aae9189753066abaad4e095ba2ff2d3e3720f")):
        _, d = P.variant_configs(r3, key, "technique")
        assert d["adapter_sha256"] == sha and d["adapter"] == {"local": f"{key}_fsa_lite_run3"}


@pytest.mark.parametrize("fmt", ["v1", "v2"])
def test_check_training_info_accepts_both_trainers_and_qwen(tmp_path, fmt):
    for hf in ("google/gemma-3-12b-it", "Qwen/Qwen2.5-7B-Instruct"):
        d = tmp_path / fmt / hf.replace("/", "_")
        d.mkdir(parents=True)
        write_training_info(d, fmt, hf, [5, 11, 17])
        info = F.check_training_info(d, hf)
        assert info["hf_id"] == hf and info["layers"] == [5, 11, 17]
        assert F.training_info_format(info).startswith(fmt)
        other = "Qwen/Qwen2.5-7B-Instruct" if hf.startswith("google") else "google/gemma-3-12b-it"
        with pytest.raises(F.AdapterError, match=f"was trained on {hf}, not {other}"):
            F.check_training_info(d, other)
    # an older v1 file naming only the registry id
    d = tmp_path / "registry"
    d.mkdir()
    (d / F.TRAINING_INFO).write_text(json.dumps({"base_model": "qwen2.5-7b-instruct", "layers": [0]}), encoding="utf-8")
    assert F.check_training_info(d, "Qwen/Qwen2.5-7B-Instruct")["hf_id"] == "Qwen/Qwen2.5-7B-Instruct"
    with pytest.raises(F.AdapterError):
        F.check_training_info(d, "google/gemma-3-12b-it")


def test_layer_restricted_adapter_plan(tmp_path):
    _, g = tiny("tiny-gemma3").load()
    lg = F.live_layout(g)
    n_layers = len({k[0] for k in lg["shapes"]})
    ad = make_adapter(tmp_path / "copy", lg, "language_model.model.", only_layers=[1, 3])
    p = F.plan(ad, lg, training=write_training_info(ad, "v2", "tiny", [1, 3]))
    assert p["adapted_layers"] == [1, 3] and p["full_adapter"] is False and p["n_model_layers"] == n_layers
    assert p["expected_tensors"] == p["n_tensors"] == 2 * 2 * len(MODS) and p["remap"]
    assert p["layers_checked_against_training_info"] and p["target_modules_checked_against_training_info"]
    full = make_adapter(tmp_path / "full", lg, "model.")
    assert F.plan(full, lg)["full_adapter"] is True and not F.plan(full, lg)["layers_checked_against_training_info"]
    # training_info says all layers, the file holds two: refused
    with pytest.raises(F.AdapterError, match="differ from the layers the adapter holds"):
        F.plan(ad, lg, training={"layers": list(range(n_layers))})


def test_peft_config_from_a_newer_peft(tmp_path):
    _peft()
    import dataclasses

    from peft import LoraConfig

    known = {f.name for f in dataclasses.fields(LoraConfig)}
    newer = {k: v for k, v in F.NEWER_PEFT_DEFAULTS.items() if k not in known}
    base = {"peft_type": "LORA", "r": 4, "lora_alpha": 8, "target_modules": "x"}
    rec = F.peft_config_compat({**base, **newer, "peft_version": "0.21.2", "use_qalora": False})
    assert rec["checked"] and set(rec["ignored_keys"]) == set(newer) | {"peft_version"} - known
    assert rec["saved_by_peft"] == "0.21.2"
    if "use_qalora" not in known:  # pinned peft 0.15.2: a feature it lacks, switched on, refuses
        with pytest.raises(F.AdapterError, match="use_qalora"):
            F.peft_config_compat({**base, "use_qalora": True})
    with pytest.raises(F.AdapterError, match="brand_new_feature"):
        F.peft_config_compat({**base, "brand_new_feature": 3})


def test_placeholders_refuse_real_runs_only(tmp_path):
    from models.dummy import DummyModel

    b = F.FSABridge({"adapter": "/home/x/FILL_ME_run/adapter", "adapter_sha256": "FILL_ME"})
    b.prepare([], None)
    assert b.wrap(DummyModel()).name.endswith("fsa(dry-run)")
    assert b.info()["placeholders"] == ["adapter", "adapter_sha256"]
    with pytest.raises(F.AdapterError, match="adapter and adapter_sha256 for this model still FILL_ME"):
        b.wrap(tiny("tiny-gemma3"))
    ok_path = F.FSABridge({"adapter": str(tmp_path), "adapter_sha256": "FILL_ME"})
    ok_path.prepare([], None)
    with pytest.raises(F.AdapterError, match="adapter_sha256 for this model still FILL_ME"):
        ok_path.wrap(tiny("tiny-gemma3"))


# --------------------------------------------------------------------------- through PEFT


def _peft():
    return pytest.importorskip("peft")


@pytest.mark.parametrize("name, adapter_prefix, fmt, layers", [
    ("tiny-gemma3", "language_model.model.", "v2", [1, 3]),  # FSA-lite copy-layer style, 5.x Gemma 3 keys: remapped
    ("tiny-gemma3", "language_model.model.", "v2", None),  # FSA-lite all layers
    ("tiny-gemma3", "language_model.model.", "v1", None),  # FSA v1
    ("tiny-qwen2", "model.", "v2", None),  # Qwen: same keys on 4.51.3 and 5.x, no remap
    ("tiny-qwen2", "model.", "v1", [0, 2]),
])  # fmt: skip
def test_v1_v2_full_and_restricted_adapters_attach(tmp_path, monkeypatch, name, adapter_prefix, fmt, layers):
    _peft()
    monkeypatch.setenv("HARNESS_DATA", str(tmp_path / "data"))
    original, technique = tiny(name), tiny(name)
    live = F.live_layout(technique.load()[1])
    all_layers = sorted({k[0] for k in live["shapes"]})
    use = layers if layers is not None else all_layers
    # peft 0.21's extra config keys at their defaults, as the v2 trainer's peft writes them
    ad = make_adapter(tmp_path / "ad", live, adapter_prefix, only_layers=use,
                      extra_config={**F.NEWER_PEFT_DEFAULTS, "peft_version": "0.21.2"})  # fmt: skip
    write_training_info(ad, fmt, technique.hf_id, use)
    bridge = F.FSABridge({"adapter": str(ad)})
    bridge.prepare([], None)
    model = bridge.wrap(technique)
    info = bridge.info()
    n = len(use) * len(MODS)
    c = info["counts"]
    assert c["lora_A"] == c["lora_B"] == n and c["tensors_attached"] == c["tensors_in_file"] == c["expected"] == 2 * n
    assert c["missing"] == c["unexpected"] == c["zero_lora_B"] == c["not_equal_to_file"] == c["wrong_r_or_alpha"] == 0
    assert info["adapted_layers"] == use and info["n_adapted_layers"] == len(use)
    assert info["full_adapter"] is (use == all_layers) and info["expected_tensors"] == 2 * n
    assert info["layers_checked_against_training_info"] and info["training_info_format"].startswith(fmt)
    assert info["remapped"] is (adapter_prefix != "model.") and info["trained_on"] == technique.hf_id
    assert info["peft_config"]["checked"] and info["r"] == 4 and info["alpha"] == 8
    from core.interfaces import Item

    it = Item(id="a", system_prompt="You are a helpful assistant.", user_prompt="Say something.")
    lo, lt = original.logprobs([it.messages()], ["Hello there."]), model.logprobs([it.messages()], ["Hello there."])
    assert abs(lo[0] - lt[0]) > 1e-4  # the adapter changes the model


@pytest.mark.parametrize("name, adapter_prefix", [("tiny-gemma3", "language_model.model."),
                                                  ("tiny-qwen2", "model.language_model."),
                                                  ("tiny-gemma3", "model.")])  # fmt: skip
def test_bridge_loads_and_verifies_through_peft(tmp_path, monkeypatch, name, adapter_prefix):
    _peft()
    from core.interfaces import GenConfig, Item

    monkeypatch.setenv("HARNESS_DATA", str(tmp_path / "data"))
    original, technique = tiny(name), tiny(name)
    live = F.live_layout(technique.load()[1])
    ad = make_adapter(tmp_path / "ad", live, adapter_prefix)
    bridge = F.FSABridge({"adapter": str(ad)})
    bridge.prepare([], None)
    model = bridge.wrap(technique)
    info = bridge.info()
    n = len([k for k in live["shapes"] if k[1] == "self_attn" and k[2] in MODS])
    assert info["remapped"] is (adapter_prefix != "model.")
    c = info["counts"]
    assert c["lora_A"] == c["lora_B"] == n and c["tensors_attached"] == c["tensors_in_file"] == 2 * n
    assert c["missing"] == c["unexpected"] == c["zero_lora_B"] == c["not_equal_to_file"] == c["wrong_r_or_alpha"] == 0
    assert info["r"] == 4 and info["alpha"] == 8 and info["merged"] is False
    assert info["model_class"] == type(original.load()[1]).__name__  # same class as the original variant
    it = Item(id="a", system_prompt="You are a helpful assistant.", user_prompt="Say something.")
    lo, lt = original.logprobs([it.messages()], ["Hello there."]), model.logprobs([it.messages()], ["Hello there."])
    assert abs(lo[0] - lt[0]) > 1e-3  # the adapter changes the model
    cfg = GenConfig(max_new_tokens=8)
    ro, rt = original.generate([it.messages()], cfg)[0].text, model.generate([it.messages()], cfg)[0].text
    with model.adapter_off():
        assert model.generate([it.messages()], cfg)[0].text == ro  # adapter off = original
    assert ro != rt or abs(lo[0] - lt[0]) > 1e-3


def test_untrained_adapter_is_refused_after_loading(tmp_path, monkeypatch):
    _peft()
    monkeypatch.setenv("HARNESS_DATA", str(tmp_path / "data"))
    m = tiny("tiny-gemma3")
    ad = make_adapter(tmp_path / "zero", F.live_layout(m.load()[1]), "language_model.model.", zero_b=True)
    b = F.FSABridge({"adapter": str(ad)})
    b.prepare([], None)
    with pytest.raises(F.AdapterError, match="lora_B tensors are all zero"):
        b.wrap(m)


def test_merge_matches_unmerged_and_disables_adapter_off(tmp_path, monkeypatch):
    _peft()
    from core.interfaces import Item

    monkeypatch.setenv("HARNESS_DATA", str(tmp_path / "data"))
    a, b = tiny("tiny-gemma3"), tiny("tiny-gemma3")
    ad = make_adapter(tmp_path / "ad", F.live_layout(a.load()[1]), "language_model.model.")
    plain, merged = F.FSABridge({"adapter": str(ad)}), F.FSABridge({"adapter": str(ad), "merge": True})
    plain.prepare([], None)
    merged.prepare([], None)
    ma, mb = plain.wrap(a), merged.wrap(b)
    assert merged.info()["merged"] is True and plain.info()["merged"] is False
    it = Item(id="a", system_prompt="", user_prompt="Hi")
    assert ma.logprobs([it.messages()], ["ok"])[0] == pytest.approx(mb.logprobs([it.messages()], ["ok"])[0], abs=1e-4)
    with pytest.raises(RuntimeError, match="merged"):
        mb.adapter_off()


def test_adapter_check_job_refuses_outside_a_job():
    import os
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    script = "scripts/slurm/harness_adapter_check.sbatch"
    root = Path(__file__).resolve().parents[2]  # the repository root
    assert subprocess.run([bash, "-n", script], cwd=root).returncode == 0
    assert b"\r\n" not in (root / script).read_bytes()
    text = (root / script).read_text(encoding="utf-8")
    assert "python -m core.cluster check" in text and "experiments._bridges.fsa check" in text
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    r = subprocess.run([bash, script], cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "refusing to run outside a Slurm job" in r.stderr


# --------------------------------------------------------------------------- Slurm job settings


def _sbatch_settings(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#SBATCH --"):
            k, _, v = line[len("#SBATCH --"):].partition("=")
            out[k.strip()] = v.strip()
    return out


def test_every_slurm_job_leaves_partition_and_qos_to_cluster_env_and_fits_the_limits():
    root = Path(__file__).resolve().parents[2]
    scripts = sorted((root / "scripts" / "slurm").glob("*.sbatch"))
    assert {p.name for p in scripts} >= {"harness_run.sbatch", "harness_adapter_check.sbatch", "lite_train.sbatch"}
    for p in scripts:
        s = _sbatch_settings(p)
        assert "partition" not in s and "qos" not in s, p.name  # scripts/submit_slurm.sh passes them (cluster.env)
        assert s.get("gres") == "gpu:1" and int(s.get("cpus-per-task", "99")) <= 10, p.name
        h, m, sec = (int(x) for x in s["time"].split(":"))
        assert h * 3600 + m * 60 + sec <= 6 * 3600, p.name  # check your site's limit in cluster.env / SBATCH_EXTRA
        assert s.get("output", "").startswith("scripts/slurm/logs/"), p.name


# setup_env.sh's environment in these tests is built from nothing, never copied from os.environ:
# a job exports HARNESS_CLUSTER / PBS_JOBID / SLURM_JOB_ID / UV_AUTO_INSTALL and has a system
# python3.11 in /bin, and either would change the scenario under test. Every variable below is set
# or unset by each test (SETUP_VARS), and PATH holds only directories under tmp_path: the fake
# python3.11 / uv directory and a private bin with just the tools the script and the stubs call.
SETUP_VARS = ("HARNESS_CLUSTER", "PBS_JOBID", "SLURM_JOB_ID", "UV", "UV_INSTALL_DIR", "PYTHON", "VIRTUAL_ENV",
              "UV_AUTO_INSTALL", "UV_VERSION")  # fmt: skip
# setup_env.sh up to SETUP_DRY: dirname; install_uv: env, sh (curl is a per-test stub); the stub
# python / uv / installer scripts: mkdir, chmod, cat. bash runs every stub through its #!/bin/bash.
SETUP_TOOLS = ("bash", "sh", "env", "dirname", "mkdir", "chmod", "cat")


def _setup_vars(**kw: str | None) -> dict[str, str | None]:
    """Every SETUP_VARS key: None (unset) unless given."""
    bad = set(kw) - set(SETUP_VARS)
    assert not bad, bad
    return {k: kw.get(k) for k in SETUP_VARS}


def _private_bin(dest: Path) -> Path:
    """dest/<tool> for each SETUP_TOOLS, resolved with shutil.which now. POSIX: a symlink to the
    tool. Windows (Git Bash / MSYS2): an exec wrapper instead, because an MSYS binary reached through a
    symlink outside its own folder cannot load msys-2.0.dll (it exits without output); the wrapper
    execs the tool by its absolute path, so its folder still never goes on PATH."""
    import os
    import shutil
    import subprocess

    dest.mkdir()
    bash = shutil.which("bash")
    for tool in SETUP_TOOLS:
        src = shutil.which(tool)
        if src is None:
            pytest.skip(f"{tool} not found")
        if os.name == "nt":
            posix = subprocess.run([bash, "-c", 'cygpath -u "$1"', "_", src], capture_output=True, text=True,
                                   check=True).stdout.strip()  # fmt: skip
            (dest / tool).write_text(f"#!/bin/bash\nexec '{posix}' \"$@\"\n", encoding="utf-8", newline="\n")
            (dest / tool).chmod(0o755)
        else:
            os.symlink(src, dest / tool)
    return dest


def _setup_env_dry(tmp_path: Path, env_vars: dict, with_python311: bool,
                   extra_tools: dict | None = None) -> "subprocess.CompletedProcess":  # fmt: skip
    import os
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None or shutil.which("dirname") is None:
        pytest.skip("bash / coreutils not on PATH")
    assert set(env_vars) == set(SETUP_VARS), "set or unset every SETUP_VARS key (use _setup_vars)"
    root = Path(__file__).resolve().parents[2]  # the repository root
    work = tmp_path / "repo"
    (work / "scripts" / "lib").mkdir(parents=True)
    (work / "eval").mkdir()  # setup_env.sh builds eval/.venv
    shutil.copy(root / "scripts" / "lib" / "setup_env.sh", work / "scripts" / "lib" / "setup_env.sh")
    tools = tmp_path / "bin"
    tools.mkdir()
    stub_python = '#!/bin/bash\nif [ "$1" = "-m" ]; then exit 1; fi\necho "$0 3.11.9"\n'  # no pip; 3.11
    log = tmp_path / "calls.log"
    (tools / "uv").write_text('#!/bin/bash\necho "uv $*" >> "' + log.as_posix() + '"\n'
                              'if [ "$1" = venv ]; then mkdir -p "${@: -1}/bin"; printf \'%s\' \'' + stub_python.replace("'", "'\''")
                              + '\' > "${@: -1}/bin/python"; chmod +x "${@: -1}/bin/python"; fi\n', encoding="utf-8", newline="\n")  # fmt: skip
    (tools / "uv").chmod(0o755)
    if with_python311:
        (tools / "python3.11").write_text('#!/bin/bash\necho "python3.11 $*" >> "' + log.as_posix() + '"\n'
                                          'mkdir -p "${@: -1}/bin"; printf \'%s\' \'' + stub_python.replace("'", "'\''")
                                          + '\' > "${@: -1}/bin/python"; chmod +x "${@: -1}/bin/python"\n', encoding="utf-8", newline="\n")  # fmt: skip
        (tools / "python3.11").chmod(0o755)
    for name, body in (extra_tools or {}).items():
        (tools / name).write_text(body, encoding="utf-8", newline="\n")
        (tools / name).chmod(0o755)
    sysbin = _private_bin(tmp_path / "sysbin")
    env = {"PATH": f"{tools}{os.pathsep}{sysbin}", "SETUP_DRY": "1", "HOME": str(tmp_path), "LC_ALL": "C"}
    if os.name == "nt":  # Windows needs these two to start any process; neither is a setup input
        env.update({k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR") if k in os.environ})
    env.update({k: v for k, v in env_vars.items() if v is not None})
    for d in env["PATH"].split(os.pathsep):  # nothing outside tmp_path: never /bin, /usr/bin, /usr/local/bin
        assert Path(d).resolve().is_relative_to(tmp_path.resolve()), d
    assert all(k in env for k, v in env_vars.items() if v is not None)
    assert not any(k in env for k, v in env_vars.items() if v is None)
    return subprocess.run([bash, "scripts/lib/setup_env.sh"], cwd=work, env=env, capture_output=True, text=True, timeout=60)


def test_setup_env_uses_uv_when_python311_is_missing(tmp_path):
    r = _setup_env_dry(tmp_path, _setup_vars(UV=str(tmp_path / "bin" / "uv")), with_python311=False)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "python for .venv: uv (" in out and ") venv --python 3.11 (python3.11 not on PATH)" in out
    assert "installer: uv pip" in out and "interpreter: .venv/bin/python 3.11.9" in out
    assert "uv venv --python 3.11 .venv" in (tmp_path / "calls.log").read_text()


def test_setup_env_prefers_python311_on_path(tmp_path):
    r = _setup_env_dry(tmp_path, _setup_vars(UV=str(tmp_path / "bin" / "uv")), with_python311=True)
    assert r.returncode == 0, r.stderr
    assert "python for .venv:" in r.stdout and "python3.11" in r.stdout.splitlines()[0]
    assert "python3.11 -m venv .venv" in (tmp_path / "calls.log").read_text()


def test_setup_env_stops_without_any_python311(tmp_path):
    r = _setup_env_dry(tmp_path, _setup_vars(UV=str(tmp_path / "nowhere" / "uv")), with_python311=False)
    assert r.returncode == 1 and "no Python 3.11" in r.stderr


def test_setup_env_scenarios_ignore_the_parent_environment(tmp_path, monkeypatch):
    """The parent process looks like a cluster job (HARNESS_CLUSTER, PBS_JOBID, SLURM_JOB_ID, UV,
    PYTHON, UV_AUTO_INSTALL exported) and has a working python3.11 on its PATH; every scenario must still see only
    what the test sets."""
    import os

    parent_bin = tmp_path / "parent_bin"
    parent_bin.mkdir()
    marker = tmp_path / "parent_python_was_used"
    stub = '#!/bin/bash\nif [ "$1" = "-m" ]; then exit 1; fi\necho "$0 3.11.9"\n'
    (parent_bin / "python3.11").write_text('#!/bin/bash\necho leaked > "' + marker.as_posix() + '"\n'
                                           'mkdir -p "${@: -1}/bin"; printf \'%s\' \'' + stub.replace("'", "'\''")
                                           + '\' > "${@: -1}/bin/python"; chmod +x "${@: -1}/bin/python"\n',
                                           encoding="utf-8", newline="\n")  # fmt: skip
    (parent_bin / "python3.11").chmod(0o755)
    monkeypatch.setenv("PATH", f"{parent_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    for k, v in {"HARNESS_CLUSTER": "pbs-site", "UV_AUTO_INSTALL": "1", "PBS_JOBID": "1", "SLURM_JOB_ID": "1",
                 "UV": str(parent_bin / "uv"), "PYTHON": str(parent_bin / "python3.11")}.items():  # fmt: skip
        monkeypatch.setenv(k, v)

    def run(name: str, **kw):
        d = tmp_path / name
        d.mkdir()
        return d, _setup_env_dry(d, _setup_vars(**{k: (str(Path(v.format(d=d))) if "{d}" in v else v) for k, v in kw.get("vars", {}).items()}),
                                 with_python311=kw.get("py", False), extra_tools=kw.get("tools"))  # fmt: skip

    d, r = run("uv", vars={"UV": "{d}/bin/uv"})
    assert r.returncode == 0, r.stderr
    first = r.stdout.splitlines()[0]
    assert first.startswith("python for .venv: uv (") and "(python3.11 not on PATH)" in first, r.stdout
    assert "UV_AUTO_INSTALL:" not in r.stdout and "installer: uv pip" in r.stdout
    d, r = run("on_path", vars={"UV": "{d}/bin/uv"}, py=True)
    assert r.returncode == 0, r.stderr
    first = r.stdout.splitlines()[0]
    assert first.startswith("python for .venv: ") and "python3.11" in first and "parent_bin" not in first, r.stdout
    d, r = run("none", vars={"UV": "{d}/nowhere/uv"})
    assert r.returncode == 1 and "no Python 3.11" in r.stderr, r.stdout
    d, r = run("plain", vars={"HARNESS_CLUSTER": "slurm-site", "UV": "{d}/nowhere/uv"}, tools={"curl": "#!/bin/bash\nexit 99\n"})
    assert r.returncode == 1 and "no Python 3.11" in r.stderr and "installing uv" not in r.stdout
    d, r = run("auto", vars={"UV_AUTO_INSTALL": "1", "UV": "{d}/bin/uv"})  # set by the test: honoured
    assert r.returncode == 0 and r.stdout.splitlines()[0].startswith("UV_AUTO_INSTALL: uv / pip caches under"), r.stdout
    assert not marker.exists(), "the parent's python3.11 was used"


@pytest.mark.parametrize("script", ["scripts/slurm/harness_run.sbatch", "scripts/slurm/harness_adapter_check.sbatch"])
@pytest.mark.parametrize("override", [None, "/scratch/other/hub"])
def test_slurm_jobs_use_the_configured_hf_cache(tmp_path, script, override):
    """The scripts' HF_HUB_CACHE line, then env.sh: the model cache is cluster.env's HF_HUB_CACHE (or
    the HARNESS_HF_HUB_CACHE override), HF_HOME is not moved into the repo, HF_TOKEN_PATH names the
    token file (which is never read). Without HF_HUB_CACHE the job stops."""
    import os
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    root = Path(__file__).resolve().parents[2]
    text = (root / script).read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("export HF_HUB_CACHE="))
    assert text.index(line) < text.index("source env.sh") if "source env.sh" in text else True
    home = tmp_path / "home"
    (home / ".cache" / "huggingface").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "token").write_text("placeholder", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_") and k != "HARNESS_HF_HUB_CACHE"}
    env["HOME"] = home.as_posix()
    r = subprocess.run([bash, "-c", f"{line}\necho ok"], cwd=root / "eval", env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "set HF_HUB_CACHE in cluster.env" in r.stderr
    env["HF_HUB_CACHE"] = (home / "hf_cache").as_posix()  # as cluster.env would set it
    if override:
        env["HARNESS_HF_HUB_CACHE"] = override
    probe = f'{line}\nsource env.sh >/dev/null 2>&1\necho "$HF_HUB_CACHE|${{HF_HOME:-unset}}|${{HF_TOKEN_PATH:-unset}}"'
    r = subprocess.run([bash, "-c", probe], cwd=root / "eval", env=env, capture_output=True, text=True, timeout=60)
    hub, hf_home, token_path = r.stdout.strip().splitlines()[-1].split("|")
    # (Git Bash on Windows rewrites HOME into an MSYS path, so compare from the home folder on)
    if override:
        assert hub == override
    else:
        assert hub.endswith("/home/hf_cache")
    assert hf_home == "unset"  # no new cache inside the repo
    assert token_path.endswith("/home/.cache/huggingface/token")


GEMMA_REV = "96b6f1eccf38110c56df3a15bffe176da04bfd80"


def test_gemma_revision_is_pinned_for_both_variants():
    from core import protocol as P

    spec = P.load_approach("fsa")
    for variant in ("original", "technique"):
        assert P.variant_configs(spec, "gemma", variant)[0]["revision"] == GEMMA_REV


def test_slurm_jobs_load_models_offline_but_setup_and_fetch_keep_the_network():
    root = Path(__file__).resolve().parents[2]
    run = (root / "scripts" / "slurm" / "harness_run.sbatch").read_text(encoding="utf-8")
    check = (root / "scripts" / "slurm" / "harness_adapter_check.sbatch").read_text(encoding="utf-8")
    for text in (run, check):
        assert 'export HF_HUB_CACHE="${HARNESS_HF_HUB_CACHE:-${HF_HUB_CACHE:?set HF_HUB_CACHE in cluster.env}}"' in text
        assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in text
    off = run.index("export HF_HUB_OFFLINE")
    assert run.index("exec bash scripts/lib/setup_env.sh") < off and run.index("python -m core.data fetch") < off
    assert check.index("export HF_HUB_OFFLINE") < check.index("experiments._bridges.fsa check")


def test_adapter_check_stops_before_loading_when_the_snapshot_is_not_cached(tmp_path, monkeypatch):
    hf_constants = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.setenv("HARNESS_CLUSTER", "slurm-site")
    monkeypatch.delenv("FSA_ADAPTER_DIR", raising=False)
    (tmp_path / "paths.local.yaml").write_text("adapters:\n  gemma_fsa0: /nowhere/gemma_fsa0\n", encoding="utf-8")
    monkeypatch.setenv("FSA_PATHS_FILE", str(tmp_path / "paths.local.yaml"))  # an adapter path that does not exist
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hf_cache"))
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)

    def no_load(*a, **k):
        raise AssertionError("the model must not load")

    from models.hf import HFModel

    monkeypatch.setattr(HFModel, "load", no_load)
    with pytest.raises(SystemExit, match=f"revision {GEMMA_REV} is not in .*HF_HUB_OFFLINE is on"):
        F.check_main(["--model", "gemma", "--device", "cpu"])
    # with the snapshot folder present, the cache check passes and the next step is reached: the
    # bridge's adapter check (the configured adapter path does not exist here) or the model load
    (tmp_path / "hf_cache" / "models--google--gemma-3-12b-it" / "snapshots" / GEMMA_REV).mkdir(parents=True)
    with pytest.raises((AssertionError, RuntimeError), match="must not load|FSA adapter not found"):
        F.check_main(["--model", "gemma", "--device", "cpu"])
