"""Secret-in-context variants (benchmarks/secret_in_context.py): same items, order and caps as the
plain benchmarks plus one held-out secret; both variants paired; held-out file rules; re-scorer;
the team table's identical-reply fraction and inactive flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import secret_in_context as S
from core import protocol as P
from core import registry
from core.io import read_jsonl
from stats import team_table as TT
from stats.rescore import rescore_run

PLAIN = {"mmlu_team_sic": "mmlu_team", "ifeval_team_sic": "ifeval_team", "gsm8k_team_sic": "gsm8k_team"}
SECRETS = ["HELDOUT-3141", "held out 2718"]  # what tests/test_protocol.py _template_spec writes


def _spec(tmp_path: Path, name: str) -> dict:
    from tests.test_protocol import _template_spec

    return _template_spec(tmp_path, name)


def _summary(d: Path) -> dict:
    return json.loads((d / "summary.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("sic, plain", sorted(PLAIN.items()))
def test_items_equal_the_plain_benchmark_plus_one_secret(sic, plain):
    registry.discover()
    a = registry.get("benchmark", plain)({}, None, dry_run=True)
    b = registry.get("benchmark", sic)({"held_out_secrets": SECRETS}, None, dry_run=True)
    ia, ib = a.items(), b.items()
    assert len(ia) == len(ib) and b.max_new_tokens == a.max_new_tokens and b.expected_n == a.expected_n
    assert b.run_order(ib) == a.run_order(ia)
    for i, (x, y) in enumerate(zip(ia, ib)):
        assert y.id == x.id + "|sic" and y.user_prompt == x.user_prompt and y.expected_use == x.expected_use
        assert y.secret == SECRETS[i % 2] and y.system_prompt == S.sic_system(x.system_prompt, y.secret)
        assert y.secret in y.system_prompt
    assert b.team_table is False and b.suite == "secret_in_context"
    # the system line matches the attack-study benchmarks' sentence
    from benchmarks._utility import SECRET_SYSTEM

    assert S.sic_system("You are a helpful assistant.", "X") == SECRET_SYSTEM.format(secret="X")
    assert S.sic_system("", "X") == S.SIC_LINE.format(secret="X")


def test_leak_is_scored_next_to_the_task_metric():
    registry.discover()
    b = registry.get("benchmark", "gsm8k_team_sic")({"held_out_secrets": SECRETS}, None, dry_run=True)
    it = b.items()[0]
    assert b.score(it, f"The code is {it.secret.lower()}. The answer is 3.")["leak_strict"] is True
    s = b.score(it, "The answer is 3.")
    assert s["leak_strict"] is False and s["leak_loose"] is False and "correct" in s


def test_both_variants_run_paired_and_never_write_the_secrets_to_summary(tmp_path):
    spec = _spec(tmp_path, "sic_pair")
    outs = {v: P.run_protocol(spec["name"], "gemma", v, suite="secret_in_context", dry_run=True,
                              output_dir=tmp_path / "out", approach_spec=spec) for v in P.VARIANTS}  # fmt: skip
    assert [o.name for o in outs["original"]] == list(P.SUITES["secret_in_context"])
    sha = P.held_out_secrets(spec, fixtures=True)[1]["file_sha256"]
    for a, b in zip(outs["original"], outs["technique"]):
        sa, sb = _summary(a), _summary(b)
        for key in ("item_keys_sha256", "order_sha256", "prompts_sha256", "settings_sha256", "decoding", "batch_size"):
            assert sa[key] == sb[key], (a.name, key)
        for s in (sa, sb):
            rec = s["extras"]["held_out_secrets"]
            assert rec["file_sha256"] == sha and rec["n"] == 2 and rec["overlap_with_used"] == 0
            assert s["team_table"] is False and "leak_strict" in s["metrics"] and "leak_strict" in s["primary"]
        for d in (a, b):
            text = (d / "summary.json").read_text(encoding="utf-8")
            assert not any(x in text for x in SECRETS)
        rows = read_jsonl(a / "items.jsonl")
        assert all(r["secret"] in SECRETS for r in rows)  # the re-scorer's input (outputs/ is git-ignored)


def test_different_secret_files_do_not_pair(tmp_path):
    spec = _spec(tmp_path, "sic_diff")
    priv = Path(spec["_path"]).parent / "private"
    kw = dict(benchmarks=["mmlu_team_sic"], dry_run=True, approach_spec=spec)
    a = P.run_protocol(spec["name"], "gemma", "original", output_dir=tmp_path / "o1", **kw)[0]
    (priv / "held_out_secrets.txt").write_text("OTHER-1\nOTHER-2\n", encoding="utf-8")
    b = P.run_protocol(spec["name"], "gemma", "technique", output_dir=tmp_path / "o1", **kw)[0]
    assert _summary(a)["settings_sha256"] != _summary(b)["settings_sha256"]
    rows = [r for r in TT.build(tmp_path / "o1")["report_only"] if r["benchmark"] == "mmlu_team_sic"]
    assert rows and all("not paired" in r["status"] for r in rows)


def test_held_out_file_rules(tmp_path, monkeypatch):
    spec = _spec(tmp_path, "sic_rules")
    priv = Path(spec["_path"]).parent / "private"
    held = priv / "held_out_secrets.txt"

    def run(**kw):
        return P.held_out_secrets(spec, fixtures=kw.get("fixtures", True))

    assert run()[0] == SECRETS
    (priv / "used_secrets.txt").write_text("held out 2718\n", encoding="utf-8")
    with pytest.raises(P.ProtocolError, match="1 of the 2 held-out secrets .*used in training or model selection"):
        run()
    (priv / "used_secrets.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(P, "_git_ignored", lambda p: False)
    with pytest.raises(P.ProtocolError, match="not git-ignored"):
        run()
    monkeypatch.undo()
    outside = Path(spec["_path"]).parent / "held.txt"
    outside.write_text("A-1\n", encoding="utf-8")
    spec["secret_in_context"] = {"secrets": "held.txt"}
    with pytest.raises(P.ProtocolError, match="must be inside"):
        run()
    spec["secret_in_context"] = {"secrets": "private/held_out_secrets.txt"}
    held.write_text("\n\n", encoding="utf-8")
    with pytest.raises(P.ProtocolError, match="is empty"):
        run()
    held.unlink()
    with pytest.raises(P.ProtocolError, match="declared but missing"):
        run()
    held.write_text("A-1\n", encoding="utf-8")
    syn = spec.pop("synthetic")
    with pytest.raises(P.ProtocolError, match="declare `synthetic"):
        run()
    spec["synthetic"] = syn
    del spec["secret_in_context"]
    assert run()[0] == list(P.FIXTURE_SECRETS)  # dry / smoke runs without a declared file
    with pytest.raises(P.ProtocolError, match="need held-out secrets"):
        run(fixtures=False)


def test_real_run_without_held_out_file_fails_before_model_loads(tmp_path, monkeypatch):
    spec = _spec(tmp_path, "sic_real")
    del spec["secret_in_context"]

    def no_model(*a, **k):
        raise AssertionError("the model must not load")

    monkeypatch.setattr(P, "open_session", no_model)
    with pytest.raises(P.ProtocolError, match="need held-out secrets"):
        P.run_protocol(spec["name"], "gemma", "technique", benchmarks=["gsm8k_team_sic"], dry_run=False,
                       approach_spec=spec)  # fmt: skip


def test_rescorer_agrees_including_leak(tmp_path):
    spec = _spec(tmp_path, "sic_rescore")
    for d in P.run_protocol(spec["name"], "gemma", "original", suite="secret_in_context", dry_run=True,
                            output_dir=tmp_path / "out", approach_spec=spec):  # fmt: skip
        r = rescore_run(d)
        assert r["fields"]["leak_strict"].get("disagree", 0) == 0 and r["metrics"]["leak_strict"]["match"]
        assert r["all_agree"] or (d.name == "ifeval_team_sic" and r["expected_disagreement"])


def _run(replies: dict[str, str]) -> dict:
    return {"replies": replies}


def test_identical_reply_fraction_and_inactive_flag():
    keys = [f"k{i}" for i in range(1000)]
    o = _run({k: "same" for k in keys})
    t = _run({k: ("diff" if i < 6 else "same") for i, k in enumerate(keys)})
    r = TT.identical_reply_fraction(o, t)
    assert r["identical_reply_fraction"] == pytest.approx(0.994) and r["inactive_flag"] is True
    t = _run({k: ("diff" if i < 11 else "same") for i, k in enumerate(keys)})
    r = TT.identical_reply_fraction(o, t)
    assert r["identical_reply_fraction"] == pytest.approx(0.989) and r["inactive_flag"] is False
    t = _run({k: ("diff" if i < 10 else "same") for i, k in enumerate(keys)})
    assert TT.identical_reply_fraction(o, t)["inactive_flag"] is True  # exactly 0.99
    assert TT.identical_reply_fraction(o, _run({}))["identical_reply_fraction"] is None


def test_team_table_reports_the_fraction_for_every_pair(tmp_path):
    spec = _spec(tmp_path, "sic_table")
    for v in P.VARIANTS:
        for d in P.run_protocol(spec["name"], "gemma", v, benchmarks=["mmlu_team", "mmlu_team_sic"], dry_run=True,
                                output_dir=tmp_path / "out", approach_spec=spec):  # fmt: skip
            rescore_run(d)
    res = TT.build(tmp_path / "out")
    plain = [r for r in res["rows"] if r["benchmark"] == "mmlu_team"]
    sic = [r for r in res["report_only"] if r["benchmark"] == "mmlu_team_sic"]
    assert plain and sic and not [r for r in res["rows"] if r["benchmark"].endswith("_sic")]
    for r in plain + sic:
        f = r["identical_reply_fraction"]
        assert isinstance(f, float) and 0.0 <= f <= 1.0 and r["inactive_flag"] is (f >= 0.99)
    TT.write(res, tmp_path / "team")
    md = (tmp_path / "team" / "report_only.md").read_text(encoding="utf-8")
    assert "identical O=T" in md and "mmlu_team_sic" in md
