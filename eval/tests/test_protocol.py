"""core/protocol.py: dry run of every suite, original vs technique pairing (same items, order and
settings), held-out secret check, approach specs, and a smoke run with the tiny random Gemma 3 /
Qwen2 models (real HF generate + chat templates, fixture data, <= 5 items per benchmark)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from core import protocol as P
from core.io import ROOT, read_jsonl

TEMPLATE = ROOT / "experiments" / "_template"
TEAM_CAPS = {"password_eval_team": 256, "confaide_t4_team": 256, "mmlu_team": 48, "ifeval_team": 512,
             "tt_extraction_team": 512, "tt_hijacking_team": 64, "tt_validity_team": 64, "gsm8k_team": 512,
             "hidden_code_team": 256, "code_check_team": 256,
             "mmlu_team_sic": 48, "ifeval_team_sic": 512, "gsm8k_team_sic": 512}  # fmt: skip


def _summary(d: Path) -> dict:
    return json.loads((d / "summary.json").read_text(encoding="utf-8"))


def _template_spec(tmp_path: Path, name: str = "tmpl_copy") -> dict:
    """A copy of experiments/_template with its defence registered under a test name."""
    dst = tmp_path / name
    shutil.copytree(TEMPLATE, dst)
    src = (dst / "defence.py").read_text(encoding="utf-8").replace('NAME = "my_approach"', f'NAME = "{name}"')
    (dst / "defence.py").write_text(src, encoding="utf-8")
    spec_mod = importlib.util.spec_from_file_location(f"_test_{name}_defence", dst / "defence.py")
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)  # type: ignore[union-attr]
    yml = (dst / "approach.yaml").read_text(encoding="utf-8").replace("my_approach", name)
    (dst / "approach.yaml").write_text(yml, encoding="utf-8")
    (dst / "private").mkdir()
    (dst / "private" / "used_secrets.txt").write_text("", encoding="utf-8")  # as a other approach would create it
    (dst / "private" / "held_out_secrets.txt").write_text("HELDOUT-3141\nheld out 2718\n", encoding="utf-8")
    return P.load_approach(name, dst / "approach.yaml")


@pytest.mark.parametrize("model_key", ["gemma", "qwen"])
def test_dry_run_every_suite(tmp_path, model_key):
    spec = _template_spec(tmp_path, f"dry_{model_key}")
    outs = P.run_protocol(spec["name"], model_key, "original", suite="all", dry_run=True,
                          output_dir=tmp_path / "out", approach_spec=spec)  # fmt: skip
    assert [o.name for o in outs] == list(P.SUITES["all"])
    for o in outs:
        s = _summary(o)
        assert set(P.SUMMARY_KEYS) <= set(s), set(P.SUMMARY_KEYS) - set(s)
        assert s["decoding"]["max_new_tokens"] == TEAM_CAPS[o.name] and s["decoding"]["greedy"] is True
        assert s["decoding"]["repetition_penalty"] == 1.0 and s["decoding"]["replies_per_item"] == 1
        assert s["batch_size"] == 16 and s["protocol_compliant"] is False and s["dry_run"] is True
        assert s["team_table"] is (o.name not in P.SUITES["synthetic"] + P.SUITES["secret_in_context"])
        assert s["truncation"]["n"] == s["n_items"] <= 5
        rows = read_jsonl(o / "items.jsonl")
        assert [r["index"] for r in rows] == list(range(len(rows)))
        assert sorted(r["order"] for r in rows) == list(range(len(rows)))
        for k in s["primary"]:
            assert k in s["metrics"], (o.name, k)
    assert (tmp_path / "out" / spec["name"] / model_key / "original" / "confaide_t4_team" / "confaide_vista_check.csv").exists()


def test_original_and_technique_have_identical_items_order_and_settings(tmp_path):
    """Required: the two rows of a benchmark pair up item by item, with the same settings."""
    spec = _template_spec(tmp_path, "pairing")
    outs = {}
    for variant in P.VARIANTS:
        outs[variant] = P.run_protocol(spec["name"], "gemma", variant, suite="all", dry_run=True,
                                       output_dir=tmp_path / "out", approach_spec=spec)  # fmt: skip
    for a, b in zip(outs["original"], outs["technique"]):
        sa, sb = _summary(a), _summary(b)
        assert sa["variant"] == "original" and sb["variant"] == "technique"
        assert sa["defence"] == {"name": "none"} and sb["defence"]["name"] == "pairing"
        for key in ("item_keys_sha256", "order_sha256", "prompts_sha256", "settings_sha256", "decoding",
                    "batch_size", "n_items", "data_files", "benchmark", "limit"):
            assert sa[key] == sb[key], (a.name, key)
        ra, rb = read_jsonl(a / "items.jsonl"), read_jsonl(b / "items.jsonl")
        assert [(r["key"], r["order"], r["prompt_sha256"]) for r in ra] == [(r["key"], r["order"], r["prompt_sha256"]) for r in rb]


def test_technique_options_do_not_leak_into_original():
    spec = {"name": "x", "defence": {"name": "fsa"}, "_path": "x",
            "models": {"gemma": {"model": {"batch_size": 8, "revision": "abc"}, "defence": {"adapter": "a/b"}}}}  # fmt: skip
    m_o, d_o = P.variant_configs(spec, "gemma", "original")
    m_t, d_t = P.variant_configs(spec, "gemma", "technique")
    assert m_o == m_t and m_o["batch_size"] == 8 and m_o["revision"] == "abc" and m_o["name"] == "gemma3-12b-it"
    assert d_o == {"name": "none"} and d_t == {"name": "fsa", "adapter": "a/b"}
    with pytest.raises(P.ProtocolError):
        P.variant_configs(spec, "llama", "original")


def test_held_out_secrets_are_enforced(tmp_path):
    """Items whose secret was used in training / model selection are dropped (both variants alike),
    counted, and never written out; the file's contents never reach summary.json."""
    spec = _template_spec(tmp_path, "heldout")
    from core import registry

    codes = registry.get("secret", "digits_4")().secrets()
    priv = Path(spec["_path"]).parent / "private"
    used = [codes[900], codes[902]] + codes[:800]  # two test-split codes + all train codes
    (priv / "used_secrets.txt").write_text("\n".join(used) + "\n", encoding="utf-8")
    outs = {v: P.run_protocol(spec["name"], "gemma", v, benchmarks=["hidden_code_team"], dry_run=True,
                              output_dir=tmp_path / "out", approach_spec=spec)[0] for v in P.VARIANTS}  # fmt: skip
    for out in outs.values():
        held = _summary(out)["held_out_check"]
        fixture_codes = codes[900:908]  # the dry run builds the first 8 test items
        assert held["declared"] and held["n_excluded"] == sum(c in set(used) for c in fixture_codes) >= 2
        rows = read_jsonl(out / "items.jsonl")
        assert not any(r["key"] in ("hc0900", "hc0902") for r in rows)
        text = (out / "summary.json").read_text(encoding="utf-8")
        assert codes[0] not in text and codes[900] not in text
    assert _summary(outs["original"])["item_keys_sha256"] == _summary(outs["technique"])["item_keys_sha256"]
    (priv / "used_secrets.txt").write_text("\n".join(codes[900:1200]) + "\n", encoding="utf-8")
    with pytest.raises(P.ProtocolError, match="nothing held out"):
        P.run_protocol(spec["name"], "gemma", "original", benchmarks=["code_check_team"], dry_run=True,
                       output_dir=tmp_path / "out2", approach_spec=spec)  # fmt: skip
    (priv / "used_secrets.txt").unlink()
    with pytest.raises(P.ProtocolError, match="declared but missing"):
        P.run_protocol(spec["name"], "gemma", "original", benchmarks=["code_check_team"], dry_run=True,
                       output_dir=tmp_path / "out3", approach_spec=spec)  # fmt: skip


def test_existing_run_is_not_overwritten(tmp_path):
    spec = _template_spec(tmp_path, "nooverwrite")
    kw = dict(benchmarks=["gsm8k_team"], dry_run=True, output_dir=tmp_path / "out", approach_spec=spec)
    P.run_protocol(spec["name"], "gemma", "original", **kw)
    with pytest.raises(P.ProtocolError, match="already holds a run"):
        P.run_protocol(spec["name"], "gemma", "original", **kw)
    P.run_protocol(spec["name"], "gemma", "original", overwrite=True, **kw)


def test_approach_specs_in_repo_load():
    for name in ("fsa", "fsa_lite", "fsa_lite_run3", "fsa_lite_run3_nouse",
                 "output_filter_naive", "output_filter_rule"):  # configs/eval/*.approach.yaml
        spec = P.load_approach(name)
        assert spec["name"] == name and "name" in spec["defence"]
    with pytest.raises(P.ProtocolError, match="no approach spec"):
        P.approach_path("does_not_exist")


def test_cli_dry_run(tmp_path, capsys):
    import run

    assert run.main(["--approach", "fsa", "--model", "qwen", "--variant", "original", "--benchmarks",
                     "mmlu_team,gsm8k_team", "--dry-run", "--output-dir", str(tmp_path)]) == 0  # fmt: skip
    out = capsys.readouterr().out.strip().splitlines()
    assert [Path(p).name for p in out] == ["mmlu_team", "gsm8k_team"]
    with pytest.raises(SystemExit):
        run.main(["--approach", "fsa", "--dry-run"])  # needs --model and --variant


# --------------------------------------------------------------------------- smoke: tiny real models

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("tiny, layout", [("tiny-gemma3", "folded into first user turn"),
                                          ("tiny-qwen2", "separate system turn")])  # fmt: skip
def test_smoke_tiny_models(tmp_path, tiny, layout):
    from analysis.internals.tiny import TINY

    model = TINY[tiny]({"device": "cpu", "dtype": "float32", "batch_size": 2})
    spec = _template_spec(tmp_path, f"smoke_{tiny.replace('-', '_')}")
    key = "gemma" if "gemma" in tiny else "qwen"
    outs = P.run_protocol(spec["name"], key, "technique", suite="all", limit=3, model=model, fixtures=True,
                          output_dir=tmp_path / "out", approach_spec=spec)  # fmt: skip
    for o in outs:
        s = _summary(o)
        assert s["system_fold_check"]["layout"] == layout
        assert s["model"]["class"] in ("Gemma3ForCausalLM", "Qwen2ForCausalLM") and s["batch_size"] == 2
        assert s["model"]["chat_template_sha256"] and s["model"]["versions"]["transformers"]
        assert not any("folded" in r for r in s["non_compliance"])
        rows = read_jsonl(o / "items.jsonl")
        assert 0 < len(rows) <= 3
        for r in rows:
            assert isinstance(r["finished"], bool) and 0 <= r["n_new_tokens"] <= s["decoding"]["max_new_tokens"]
            assert r["finished"] or r["n_new_tokens"] == s["decoding"]["max_new_tokens"]
            assert r["rendered_sha256"] is not None
        t = s["truncation"]
        assert t["n"] == len(rows) and t["rate"] == sum(not r["finished"] for r in rows) / len(rows)
        from stats.rescore import rescore_run

        r = rescore_run(o)  # the independent re-scorer agrees on real-generate (random) replies
        assert r["all_agree"] or r["expected_disagreement"], (o.name, r["metrics"])
