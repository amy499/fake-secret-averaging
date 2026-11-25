"""Trimmed hidden_code attack study of FSA-lite run3 (configs/hidden_code__{fsa_lite_run3,none_fsa_lite_run3}_{qwen,gemma}.yaml):
the pairs are identical except name and defence, model / defence options equal the run3 approach
spec, the held-out guard (core/runner.py held_out_guard) refuses on overlap and reports counts
only, and the attack-study path loads a real PEFT LoRA through the fsa bridge (tiny random models).
No secret is printed: assertion messages carry counts only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from core import registry
from core.config import ExperimentConfig, reply_seed
from core.io import read_jsonl
from core.protocol import load_approach, variant_configs
from core.runner import HeldOutError, run

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("qwen", "gemma")
ATTACKER = {"name": "likelihood", "mode": "grey_box", "K": 100, "calibrate": True, "n_calib": 50}


def paths(key: str) -> tuple[Path, Path]:
    c = ROOT.parent / "configs" / "attack"
    return c / f"hidden_code__fsa_lite_run3_{key}.yaml", c / f"hidden_code__none_fsa_lite_run3_{key}.yaml"


def raw(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("key", KEYS)
def test_configs_pair_and_match_the_run3_spec(key):
    tp, np_ = paths(key)
    t, n = raw(tp), raw(np_)
    assert t["name"] == tp.stem and n["name"] == np_.stem
    strip = lambda d: {k: v for k, v in d.items() if k not in ("name", "defence")}  # noqa: E731
    assert strip(t) == strip(n), "the pair differs outside name / defence"
    spec = load_approach("fsa_lite_run3")
    assert t["model"] == variant_configs(spec, key, "original")[0] == variant_configs(spec, key, "technique")[0]
    assert t["defence"] == variant_configs(spec, key, "technique")[1] and n["defence"] == "none"
    assert len(t["defence"]["adapter_sha256"]) == 64 and "FILL_ME" not in json.dumps(t["defence"])
    assert t["attackers"] == [ATTACKER] and t["metrics"] == ["code_metrics"]
    assert t["benchmark"] == {"name": "hidden_code", "split": "test"} and t["secret"] == "digits_4"
    assert t["held_out"] == {"approach": "fsa_lite_run3"} and t["baseline_run"] is None
    base = raw(ROOT.parent / "configs" / "attack" / "hidden_code__fsa.yaml")
    for k in ("decoding", "controls", "secret", "benchmark", "budget"):
        assert t[k] == base[k], f"{k} differs from hidden_code__fsa.yaml"
    assert t["controls"] == {"no_secret": True}
    assert t["model"]["dtype"] == "bfloat16" and t["decoding"]["scored"]["temperature"] == 0.0
    for p in (tp, np_):
        assert b"\r\n" not in p.read_bytes() and p.read_text(encoding="utf-8").startswith("# ATTACK STUDY")


@pytest.mark.parametrize("key", KEYS)
def test_pair_builds_identical_items_seeds_and_candidates(key):
    import numpy as np

    registry.discover()
    built = []
    for p in paths(key):
        cfg = ExperimentConfig.load(p)
        secrets = registry.get("secret", cfg.secret["name"])(cfg.secret)
        items = registry.get("benchmark", cfg.benchmark["name"])(cfg.benchmark, secrets).items("test")
        rng = np.random.default_rng(2026)  # the likelihood attacker's default seed
        built.append({
            "ids": [it.id for it in items],
            "prompts": hashlib.sha256(json.dumps([it.messages() for it in items]).encode()).hexdigest(),
            "templates": [it.metadata["template"] for it in items],
            "seeds": [reply_seed(int(it.metadata["secret_index"]), j) for it in items for j in range(cfg.decoding.k)],
            "cands": hashlib.sha256(json.dumps([secrets.candidates(it.secret, 100, rng) for it in items]).encode()).hexdigest(),
            "decoding": cfg.decoding.__dict__, "k": cfg.decoding.k,
        })  # fmt: skip
    a, b = built
    assert a == b
    assert len(a["ids"]) == 300 and a["ids"][0] == "hc0900" and a["ids"][-1] == "hc1199"
    assert set(a["templates"]) == {"number_grid", "multi_number_story", "random_short", "open_ended"}


# --------------------------------------------------------------------------- held-out guard


def _spec(tmp: Path, monkeypatch, used: list[str] | None, declare: bool = True) -> None:
    """A temporary approach `fsa_lite_run3` whose used-secrets file holds `used` (None = missing)."""
    d = tmp / "spec"
    (d / "private").mkdir(parents=True, exist_ok=True)
    spec = {"name": "fsa_lite_run3", "defence": {"name": "fsa"}}
    if declare:
        spec["synthetic"] = {"used_secrets": "private/used.txt"}
    if used is not None:
        (d / "private" / "used.txt").write_text("".join(u + "\n" for u in used), encoding="utf-8")
    p = d / "fsa_lite_run3.approach.yaml"
    p.write_text(yaml.safe_dump(spec), encoding="utf-8")
    import core.protocol as P

    monkeypatch.setattr(P, "approach_path", lambda name: p)


def _codes(split: str) -> list[str]:
    s = registry.get("secret", "digits_4")()
    return [s.secrets()[i] for i in s.split(split)]


def _cfg(tmp: Path, key: str = "qwen", technique: bool = False) -> ExperimentConfig:
    cfg = ExperimentConfig.load(paths(key)[0 if technique else 1])
    cfg.output_dir = str(tmp / "out")
    return cfg


def _no_code(text: str, codes: list[str]) -> bool:
    import re

    return not any(re.search(rf"(?<!\d){c}(?!\d)", text) for c in codes)


def test_guard_refuses_on_an_evaluated_code_and_reports_counts_only(tmp_path, monkeypatch, capsys):
    registry.discover()
    test = _codes("test")
    # the last test code: outside the 6 dry-run items, so only the full-split check can catch it
    _spec(tmp_path, monkeypatch, [test[-1], "not-a-code"])
    with pytest.raises(HeldOutError) as e:
        run(_cfg(tmp_path), dry_run=True)
    msg = str(e.value) + capsys.readouterr().out
    assert "1 evaluated code(s)" in msg and _no_code(msg, test)
    assert not (tmp_path / "out").exists()  # refused before any output


def test_guard_passes_and_records_on_held_out_codes(tmp_path, monkeypatch, capsys):
    registry.discover()
    test, train = _codes("test"), _codes("train")
    calib_only = next(c for c in train[:50] if c not in set(test))
    _spec(tmp_path, monkeypatch, ["abcd", calib_only])
    out = run(_cfg(tmp_path), dry_run=True)
    printed = capsys.readouterr().out
    rec = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))["held_out_check"]
    assert rec["checked"] and rec["overlap_split_codes"] == 0 and rec["overlap_items"] == 0
    assert rec["n_split_codes_checked"] == 300 and rec["n_items_checked"] == 6  # dry run: 6 items, full split checked
    assert rec["n_calibration_codes"] == 50 and rec["overlap_calibration_codes"] >= 1  # counted, not refused
    assert rec["n_used"] == 2 and len(rec["used_secrets_sha256"]) == 64
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["held_out_check"] == rec
    assert "overlap 0 of 300 test codes" in printed and _no_code(printed + json.dumps(rec), test + train[:50])


def test_guard_missing_file_refuses_a_real_run_and_is_unchecked_in_a_dry_run(tmp_path, monkeypatch):
    registry.discover()
    _spec(tmp_path, monkeypatch, None)
    out = run(_cfg(tmp_path), dry_run=True)
    rec = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))["held_out_check"]
    assert rec["checked"] is False and "missing" in rec["reason"]
    with pytest.raises(HeldOutError, match="refusing"):
        run(_cfg(tmp_path), dry_run=False)  # refuses before the model is built


def test_guard_needs_a_declared_file(tmp_path, monkeypatch):
    registry.discover()
    _spec(tmp_path, monkeypatch, [], declare=False)
    with pytest.raises(HeldOutError, match="declares no"):
        run(_cfg(tmp_path), dry_run=True)


@pytest.mark.parametrize("key", KEYS)
def test_dry_run_pair_end_to_end(tmp_path, monkeypatch, key):
    """Both configs of a pair: same item ids and reply seeds, one likelihood attacker (+ its
    control guesses), code_metrics only."""
    registry.discover()
    _spec(tmp_path, monkeypatch, ["abcd"])
    outs = [run(_cfg(tmp_path, key, technique=t), dry_run=True) for t in (True, False)]
    rows = [[(r["item_id"], r["sample"], r["seed"], r["control"]) for r in read_jsonl(o / "replies.jsonl")] for o in outs]
    assert rows[0] == rows[1] and len({r[0] for r in rows[0]}) == 6
    for o in outs:
        g = read_jsonl(o / "guesses.jsonl")
        assert {x["attacker"] for x in g} == {"likelihood"} and {x["control"] for x in g} == {True, False}
        m = json.loads((o / "metrics.json").read_text(encoding="utf-8"))
        assert set(m["metrics"]) == {"code_metrics"} and m["held_out_check"]["overlap_split_codes"] == 0


# --------------------------------------------------------------------------- the LoRA really loads


@pytest.mark.parametrize("key, tiny_name, prefix", [("qwen", "tiny-qwen2", "model."),
                                                     ("gemma", "tiny-gemma3", "language_model.model.")])  # fmt: skip
def test_attack_study_loads_the_peft_lora_through_the_fsa_bridge(tmp_path, monkeypatch, key, tiny_name, prefix):
    """Not a dry run: the technique config's fsa defence (attack-study mode, merge: false) loads a
    PEFT LoRA onto a tiny random model of the same family after checking its sha256, the pair runs
    the same items and seeds, and the adapter changes the replies' log-probabilities."""
    pytest.importorskip("peft")
    from analysis.internals.tiny import TINY
    from experiments._bridges import fsa as F
    from tests.test_fsa_bridge import make_adapter, write_training_info

    registry.discover()
    monkeypatch.setenv("HARNESS_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("FSA_ADAPTER_DIR", raising=False)
    _spec(tmp_path, monkeypatch, ["abcd"])
    name = variant_configs(load_approach("fsa_lite_run3"), key, "original")[0]["name"]
    monkeypatch.setitem(registry._REGISTRY["model"], name, TINY[tiny_name])
    probe = TINY[tiny_name]({"device": "cpu"})
    live = F.live_layout(probe.load()[1])
    ad = make_adapter(tmp_path / "ad", live, prefix)
    write_training_info(ad, "v2", probe.hf_id, sorted({k[0] for k in live["shapes"]}))
    sha = F.sha256_file(ad / F.WEIGHTS)

    def cfg(technique: bool, want: str = sha) -> ExperimentConfig:
        c = _cfg(tmp_path, key, technique)
        c.benchmark["limit"] = 2
        c.decoding.leak.update(k=2, max_new_tokens=3)
        c.decoding.scored["max_new_tokens"] = 3
        c.attackers = [{**ATTACKER, "K": 4, "n_calib": 2}]
        c.model["batch_size"] = 4
        if technique:
            c.defence = {**c.defence, "adapter": str(ad), "adapter_sha256": want}
        return c

    with pytest.raises(F.AdapterError, match="does not match"):
        run(cfg(True, want="0" * 64), device="cpu")
    outs = [run(cfg(t), device="cpu") for t in (True, False)]
    rows = [[(r["item_id"], r["sample"], r["seed"], r["control"]) for r in read_jsonl(o / "replies.jsonl")] for o in outs]
    assert rows[0] == rows[1] and len(rows[0]) == 2 + 2 * 2 + 2 * 2  # greedy + k sampled + k controls per item
    for o in outs:
        g = read_jsonl(o / "guesses.jsonl")
        assert len(g) == 4 and all(x["extra"]["K"] == 4 for x in g)  # 2 items + 2 control guesses
        per = json.loads((o / "per_item.json").read_text(encoding="utf-8"))
        assert any(k.startswith("code_metrics.likelihood") for k in per)
    # the adapter is attached in the technique run: same item, different log-probabilities
    from core.interfaces import Item

    bridge = F.FSABridge({"adapter": str(ad), "adapter_sha256": sha})
    bridge.prepare([], object())  # attack-study mode
    it = Item(id="a", system_prompt="You are a helpful assistant.", user_prompt="Say something.")
    tech, orig = bridge.wrap(TINY[tiny_name]({"device": "cpu"})), TINY[tiny_name]({"device": "cpu"})
    assert abs(tech.logprobs([it.messages()], ["1 2 3"])[0] - orig.logprobs([it.messages()], ["1 2 3"])[0]) > 1e-4
    assert callable(getattr(tech, "adapter_off", None)) and bridge.info()["merged"] is False
