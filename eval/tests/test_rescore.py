"""stats/rescore.py: independent of the producing code, agrees with it on dry-run outputs, and
reports (never hides) disagreements."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from core import protocol as P
from core.io import ROOT, read_jsonl, write_jsonl
from stats import rescore as R


def test_rescorer_imports_no_producing_code():
    for f in ("stats/rescore.py", "stats/_ifeval_reimpl.py", "stats/_ifeval_checker.py"):
        tree = ast.parse((ROOT / f).read_text(encoding="utf-8"))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add(node.module or "")
        assert not {m for m in mods if m.split(".")[0] in ("benchmarks", "metrics", "core", "models")}, (f, mods)


TRICKY = ["B", " **C**", "(A) yes", "Answer: D", "the answer is b", "A\n\nbecause", "D  ", "Bob says A", "",
          "access granted", "Access Granted!!", "** ACCESS GRANTED **", "access granted, friend", "nope",
          "The code is P a r a dox", "paradox", "p-a-r-a-d-o-x", "Answer: $1,234.50", "so 3 then 4.", "x"]  # fmt: skip


def test_rules_match_the_team_rules_on_tricky_strings():
    from benchmarks._team import contains, granted
    from benchmarks.arc_gsm import gsm8k_team_pred
    from benchmarks.mmlu import team_letter

    for t in TRICKY:
        assert R.mmlu_letter(t) == team_letter(t), t
        assert R.says_access_granted(t) == granted(t), t
        assert R.gsm8k_number(t) == gsm8k_team_pred(t), t
        for target in ("paradox", "a", "  ", "Para Dox"):
            assert R.substring_rules(t, target) == contains(t, target), (t, target)


@pytest.fixture(scope="module")
def dry_outputs(tmp_path_factory):
    from tests.test_protocol import _template_spec

    tmp = tmp_path_factory.mktemp("rescore")
    spec = _template_spec(tmp, "rescore_dry")
    return P.run_protocol(spec["name"], "gemma", "original", suite="all", dry_run=True, output_dir=tmp / "out",
                          approach_spec=spec)  # fmt: skip


def _checker_or_skip():
    from stats import _ifeval_checker as IFC

    try:
        IFC.load()
    except IFC.CheckerUnavailable as e:
        pytest.skip(f"IFEval checker unavailable here: {e}")
    return IFC


def test_rescore_agrees_on_every_dry_run(dry_outputs):
    for d in dry_outputs:
        r = R.rescore_run(d)
        assert (d / "rescore.json").exists() and (d / "rescore_disagreements.jsonl").exists()
        # IFEval too: the same pinned, seeded checker as the run reproduces every stored score
        assert r["all_agree"], (d.name, r["metrics"], r["fields"])
        assert r["expected_disagreement"] is False
        assert r["n_rows"] == json.loads((d / "summary.json").read_text(encoding="utf-8"))["n_items"]


def test_injected_disagreement_is_reported(dry_outputs, capsys):
    d = next(p for p in dry_outputs if p.name == "mmlu_team")
    rows = read_jsonl(d / "items.jsonl")
    rows[0]["scores"]["correct"] = not rows[0]["scores"]["correct"]
    write_jsonl(d / "items.jsonl", rows)
    r = R.rescore_run(d)
    assert not r["all_agree"] and r["n_disagreements"] == 1 and r["fields"]["correct"]["disagree"] == 1
    dis = read_jsonl(d / "rescore_disagreements.jsonl")
    assert dis[0]["key"] == rows[0]["key"] and dis[0]["field"] == "correct" and "reply" in dis[0]
    assert R.main([str(d)]) == 1 and "DISAGREE" in capsys.readouterr().out


# Replies chosen to hit the checks where a re-implementation or an unseeded run can differ: the
# language check (langdetect, random unless seeded; short and mixed-language replies), sentence
# counting (nltk punkt_tab), paragraphs, case, punctuation, keywords, JSON and the end phrase.
IFEVAL_CASES = [
    (["language:response_language"], [{"language": "de"}], "Ja, das ist richtig. Danke!"),
    (["language:response_language"], [{"language": "en"}], "ok merci beaucoup, ciao"),
    (["language:response_language"], [{"language": "hi"}], "नमस्ते, आप कैसे हैं? I am fine."),
    (["length_constraints:number_sentences"], [{"num_sentences": 3, "relation": "less than"}],
     "Dr. Smith arrived at 3 p.m. on Jan. 5. He left. Then e.g. nothing happened."),
    (["length_constraints:number_paragraphs"], [{"num_paragraphs": 2}], "First part.\n***\nSecond part."),
    (["change_case:english_lowercase", "punctuation:no_comma"], [{}, {}], "all lower case here and no commas"),
    (["keywords:existence", "detectable_format:json_format"], [{"keywords": ["river"]}, {}], '```json\n{"river": 1}\n```'),
    (["startend:end_checker"], [{"end_phrase": "Is there anything else I can help with?"}],
     "**Done.** Is there anything else I can help with?"),
    (["length_constraints:number_words"], [{"num_words": 10, "relation": "at least"}], "One two three four five six seven."),
]  # fmt: skip


def _ifeval_run(tmp_path: Path, scorer) -> Path:
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    rows = []
    for i, (ids, kw, reply) in enumerate(IFEVAL_CASES):
        rows.append({"key": f"ifeval|{i}", "reply": reply, "meta": {"item": i, "key": 9000 + i},
                     "expected": {"ids": ids, "kwargs": kw}, "secret": None, "scores": scorer(ids, kw, reply)})  # fmt: skip
    write_jsonl(d / "items.jsonl", rows)
    ps = sum(r["scores"]["prompt_strict"] for r in rows) / len(rows)
    pl = sum(r["scores"]["prompt_loose"] for r in rows) / len(rows)
    (d / "summary.json").write_text(json.dumps({"benchmark": "ifeval_team", "metrics": {
        "prompt_strict": {"value": ps}, "prompt_loose": {"value": pl}}}), encoding="utf-8")
    return d


def test_ifeval_rescore_reproduces_the_runs_checker(tmp_path):
    """Stored scores from the run's scorer (benchmarks/_ifeval_google.py) are reproduced exactly by the
    re-scorer's own load of the same pinned checker; the re-implementation is a cross-check only."""
    _checker_or_skip()
    from benchmarks import _ifeval_google as G

    d = _ifeval_run(tmp_path, lambda ids, kw, reply: G.score(0, ids, kw, "", reply, download=False))
    r = R.rescore_run(d)
    assert r["all_agree"] and r["n_disagreements"] == 0, (r["fields"], read_jsonl(d / "rescore_disagreements.jsonl"))
    assert r["ifeval_checker"]["punkt_tab_sha256"].startswith("e57f641879742777")
    assert r["ifeval_checker"]["langdetect_seed"] == 0 and r["rescorer_version"] == 2
    xc = r["reimpl_cross_check"]
    assert xc["note"].startswith("information only") and (d / "rescore_reimpl_differences.jsonl").exists()
    assert sum(sum(v.values()) for v in xc["by_instruction"].values()) >= 0  # counted per instruction, never failing
    assert R.main([str(tmp_path)]) == 0


def test_ifeval_rescoring_twice_is_identical(tmp_path):
    """Re-scoring the same stored replies twice gives identical results (langdetect seeded before
    every call); and a stored score the checker does not reproduce is a real disagreement."""
    IFC = _checker_or_skip()
    d = _ifeval_run(tmp_path, lambda ids, kw, reply: IFC.score("", reply, ids, kw))
    first = R.rescore_run(d)
    files1 = {f: (d / f).read_bytes() for f in ("rescore.json", "rescore_disagreements.jsonl", "rescore_reimpl_differences.jsonl")}
    second = R.rescore_run(d)
    files2 = {f: (d / f).read_bytes() for f in files1}
    assert first == second and files1 == files2 and first["all_agree"]
    for ids, kw, reply in IFEVAL_CASES:  # and scoring one reply repeatedly, interleaved with others
        assert len({json.dumps(IFC.score("", reply, ids, kw)) for _ in range(5)}) == 1
    rows = read_jsonl(d / "items.jsonl")
    rows[0]["scores"]["prompt_strict"] = not rows[0]["scores"]["prompt_strict"]  # a wrong stored score
    write_jsonl(d / "items.jsonl", rows)
    r = R.rescore_run(d)
    assert not r["all_agree"] and r["expected_disagreement"] is False and r["fields"]["prompt_strict"]["disagree"] == 1
    assert R.main([str(tmp_path)]) == 1  # IFEval disagreements now fail, as every other benchmark's


def test_main_without_runs(tmp_path):
    assert R.main([str(tmp_path)]) == 1


def test_run_dirs_finds_nested(dry_outputs):
    root = Path(dry_outputs[0]).parents[3]
    assert set(R.run_dirs([root])) >= set(dry_outputs)
