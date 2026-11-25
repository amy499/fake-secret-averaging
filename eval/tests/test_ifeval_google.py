"""Vendored Google IFEval checker: md5 pins (always) and the adapter's strict / loose results
(only when the pinned packages from requirements.txt and the punkt_tab data are present;
skipped otherwise), and the real-run preflight."""

from __future__ import annotations

import pytest

from benchmarks import _ifeval_google as G
from benchmarks.ifeval import IFEvalTeam
from core.data import DataHashError


def test_vendored_files_match_the_pinned_md5s():
    got = G.vendored_md5()
    for name, prefix in G.VENDORED_MD5.items():
        assert got[name].startswith(prefix), name
    assert (G.VENDOR / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("Apache License")
    assert G.CHECKER_COMMIT in (G.VENDOR / "NOTICE.md").read_text(encoding="utf-8")


def test_info_records_commit_pins_and_seeds():
    info = G.info()
    assert info["commit"] == G.CHECKER_COMMIT and info["vendored_md5_match"] is True
    assert info["pinned"] == {"nltk": "3.9.1", "langdetect": "1.0.9", "absl-py": "2.1.0", "immutabledict": "4.2.1"}
    assert info["langdetect_seed"] == 0 and info["random_seed"] == 0


def _checker_or_skip():
    try:
        G._load(download=False)
    except (ImportError, LookupError, DataHashError) as e:
        pytest.skip(f"IFEval checker not available here: {type(e).__name__}: {e}")


def test_team_score_without_checker_is_marked_not_invented():
    b = IFEvalTeam(dry_run=True)
    it = b.items()[0]
    s = b.score(it, "a poem without commas")
    assert ("checker" in s and s["checker"].startswith("unavailable")) or set(s) == {
        "prompt_strict", "prompt_loose", "instructions_strict", "instructions_loose"}


def test_strict_and_loose_on_handwritten_replies():
    _checker_or_skip()
    ids, kw = ["punctuation:no_comma"], [{}]
    assert G.score(1, ids, kw, "p", "no commas here")["prompt_strict"] is True
    assert G.score(1, ids, kw, "p", "one, two")["prompt_strict"] is False
    # loose drops the first line: a leading line with a comma fails strict but passes loose
    r = G.score(2, ids, kw, "p", "Sure, here it is:\nno commas at all")
    assert r["prompt_strict"] is False and r["prompt_loose"] is True
    lower = G.score(3, ["change_case:english_lowercase"], [{}], "p", "all lower case text")
    assert lower["instructions_strict"] == [True]
    # sentence counting goes through the replaced tokenizer (punkt_tab), deterministically
    a = G.score(4, ["length_constraints:number_sentences"], [{"num_sentences": 2, "relation": "at least"}], "p",
                "One sentence. Two sentences. Three.")
    assert a["prompt_strict"] is True and a == G.score(4, ["length_constraints:number_sentences"],
                                                         [{"num_sentences": 2, "relation": "at least"}], "p",
                                                         "One sentence. Two sentences. Three.")


def test_real_run_without_checker_fails_before_model_loads(monkeypatch):
    from core import protocol as P

    def missing(download=True):
        raise ImportError("No module named 'nltk'")

    def no_model(*a, **k):
        raise AssertionError("the model must not load when the checker is missing")

    monkeypatch.setattr(G, "_load", missing)
    monkeypatch.setattr(P, "open_session", no_model)
    with pytest.raises(P.ProtocolError, match="ifeval_team: preflight failed, nothing generated"):
        P.run_protocol("x", "gemma", "original", benchmarks=["ifeval_team"], dry_run=False,
                       approach_spec={"name": "x"})


def test_dry_run_skips_preflight(monkeypatch):
    from core import protocol as P

    called = []
    monkeypatch.setattr(P, "preflight", lambda names: called.append(names))
    monkeypatch.setattr(P, "open_session", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        P.run_protocol("x", "gemma", "original", benchmarks=["ifeval_team"], dry_run=True,
                       approach_spec={"name": "x"})
    assert called == []


def test_preflight_passes_with_checker():
    _checker_or_skip()
    IFEvalTeam().preflight()
