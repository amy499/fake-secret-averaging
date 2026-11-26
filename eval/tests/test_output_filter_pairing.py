"""The output-filter baselines on PasswordEval (configs/eval/output_filter_{naive,rule}.approach.yaml):
no adapter, original variant identical to approach fsa's (so stats.team_table pairs them with the
existing fsa original runs, as fsa_lite_run3), both models, the protected info plumbed to the filter
(Item.protected -> SecretContext.protected), the measured effect of each mode on the stored scores,
and the rule variant labelled as an upper bound in summary.json and team_table.md. Fixture data
only (made-up passwords); no reply text is printed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import protocol as P
from core import registry
from core.interfaces import Generation
from stats import team_table as TT
from stats.rescore import rescore_run
from tests.test_fsa_lite_pairing import FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC, _settings

NAMES = ("output_filter_naive", "output_filter_rule")
PE = ["password_eval_team"]


class _LeakAll:
    """A stand-in model whose every reply contains the password and the protected info."""

    name, thinking, is_dummy, batch_size = "leak_all", False, True, 16

    def generate(self, prompts, gen_cfg, seeds=None, secret_ctx=None, **kw):
        return [Generation(text=(f"The password is {c.secret}. The info is {' '.join(c.protected)}" if c else "ok"),
                           n_tokens=5, finished=True) for c in (secret_ctx or [None] * len(prompts))]  # fmt: skip


def _scores(run_dir: Path) -> list[tuple[str, dict]]:
    rows = [json.loads(x) for x in (run_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    return [(r["meta"]["cond"], r["scores"]) for r in rows]


@pytest.mark.parametrize("name", NAMES)
def test_specs_borrow_the_fsa_original_and_carry_no_adapter(name):
    spec, fsa = P.load_approach(name), P.load_approach("fsa")
    assert spec["original_from"] == "fsa" and spec["defence"]["name"] == "output_filter"
    assert spec["defence"]["mode"] == name.rsplit("_", 1)[1]
    for key in P.MODEL_KEYS:
        assert P.variant_configs(spec, key, "original") == P.variant_configs(fsa, key, "original")
        assert "adapter" not in P.variant_configs(spec, key, "technique")[1]
    assert P.check_original_from(spec)["approach"] == "fsa"
    assert P.approach_path(name).read_bytes().count(b"\r\n") == 0
    assert ("UPPER BOUND" in P.approach_path(name).read_text(encoding="utf-8")) is (name == "output_filter_rule")


def test_password_eval_items_expose_the_info_as_protected():
    registry.discover()
    b = registry.get("benchmark", "password_eval_team")({})
    items = b.build(b.fixture_rows())
    assert items and all(it.protected == (it.metadata["info"],) and it.secret == it.metadata["pw"] for it in items)
    ctx = items[0].secret_context()
    assert ctx.protected == items[0].protected and ctx.secret == items[0].secret


@pytest.mark.parametrize("model_key", sorted(P.MODEL_KEYS))
def test_both_models_run_password_eval_and_pair_with_the_fsa_original(tmp_path, model_key):
    out = tmp_path / "protocol"
    fo = P.run_protocol("fsa", model_key, "original", benchmarks=PE, dry_run=True, output_dir=out)
    techs = {n: P.run_protocol(n, model_key, "technique", benchmarks=PE, dry_run=True, output_dir=out) for n in NAMES}
    for n, t in techs.items():
        assert all(d.parts[-4:-2] == (n, model_key) for d in t)
        assert _settings(fo) == _settings(t)  # identical settings: the pairing the team table requires
        if model_key == "gemma":
            # the same PasswordEval settings as the pinned fsa Gemma original (the constant covers the core suite)
            assert _settings(t) == {"password_eval_team": FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC["password_eval_team"]}
        s = json.loads((t[0] / "summary.json").read_text(encoding="utf-8"))
        assert s["original_from"]["approach"] == "fsa" and s["defence_info"]["mode"] == n.rsplit("_", 1)[1]
    for d in fo + [d for t in techs.values() for d in t]:
        rescore_run(d)
    result = TT.build(out, approaches=list(NAMES), models=[model_key], suite="core")
    pe = [r for r in result["rows"] if r["benchmark"] == "password_eval_team"]
    assert {r["approach"] for r in pe} == set(NAMES)
    assert {r["metric"] for r in pe} == {"leak_pw_strict", "leak_info_strict", "use_strict", "joint_strict"}  # as run3
    for r in pe:
        assert r["original_from"] == "fsa" and Path(r["original_dir"]).parts[-4] == "fsa"
        assert "not paired" not in r["status"] and "missing" not in r["status"], r["status"]
    others = [r for r in result["rows"] if r["benchmark"] != "password_eval_team"]
    assert others and all(r["status"].startswith("MISSING") for r in others)  # not run: shown, never dropped
    assert set(result["approach_labels"]) == {"output_filter_rule"}
    assert result["approach_labels"]["output_filter_rule"]["upper_bound"] is True
    TT.write(result, tmp_path / "team")
    md = (tmp_path / "team" / "team_table.md").read_text(encoding="utf-8")
    assert "## Approach labels" in md and "**output_filter_rule**: UPPER BOUND, not a defence claim." in md
    assert "**output_filter_naive**" not in md.split("## Approach labels")[1].split("##")[0]


def test_measured_effect_of_each_mode_with_a_model_that_leaks_everything(tmp_path):
    out = tmp_path / "protocol"
    kw = {"benchmarks": PE, "dry_run": True, "output_dir": out, "model": _LeakAll()}
    (orig,) = P.run_protocol("fsa", "gemma", "original", **kw)
    (naive,) = P.run_protocol("output_filter_naive", "gemma", "technique", **kw)
    (rule,) = P.run_protocol("output_filter_rule", "gemma", "technique", **kw)
    so, sn, sr = _scores(orig), _scores(naive), _scores(rule)
    assert so and all(s["pw_strict"] and s["info_strict"] for _, s in so)  # the original leaks everything
    assert all(not s["pw_strict"] and not s["pw_loose"] for _, s in sn + sr)  # both filters remove the password
    assert all(s["info_strict"] for _, s in sn)  # naive leaves the info
    for cond, s in sr:  # rule: info only where the user gave the password
        assert bool(s["info_strict"]) is (cond == "right"), cond
    m = json.loads((rule / "summary.json").read_text(encoding="utf-8"))["metrics"]
    assert m["leak_info_strict"]["value"] == 0.0 and m["use_strict"]["value"] == 1.0 and m["joint_strict"]["value"] == 1.0
