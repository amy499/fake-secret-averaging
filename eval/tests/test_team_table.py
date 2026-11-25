"""stats/team_table.py on synthetic run folders (dry-run outputs, edited to simulate real cases)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import protocol as P
from stats import team_table as TT
from stats.rescore import rescore_run


def _edit(d: Path, fn) -> None:
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    fn(s)
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")


def _compliant(s: dict) -> None:
    s["protocol_compliant"], s["non_compliance"] = True, []


@pytest.fixture()
def runs(tmp_path):
    from tests.test_protocol import _template_spec

    out = tmp_path / "out"
    made = {}
    for name in ("alpha", "beta"):
        spec = _template_spec(tmp_path, name)
        for variant in P.VARIANTS:
            benches = ["mmlu_team", "confaide_t4_team", "hidden_code_team"] + (["gsm8k_team"] if name == "alpha" else [])
            for d in P.run_protocol(name, "gemma", variant, benchmarks=benches, dry_run=True, output_dir=out,
                                    approach_spec=spec):  # fmt: skip
                rescore_run(d)
                made[(name, variant, d.name)] = d
    return out, made


def test_build_pairs_rows_and_reports_reasons(runs, tmp_path):
    out, made = runs
    res = TT.build(out)
    rows = {(r["approach"], r["benchmark"], r["metric"]): r for r in res["rows"]}
    r = rows[("alpha", "mmlu_team", "accuracy")]
    assert r["diff"] is not None and r["diff"]["n"] == 5 and "mcnemar_p" in r["diff"]
    assert "not compliant" in r["status"] and "dry run" in r["status"] and "not paired" not in r["status"]
    c = rows[("alpha", "confaide_t4_team", "leak")]
    assert c["diff"]["method"].startswith("paired cluster bootstrap")
    # check 4: core benchmarks never run are listed, second pass ones as not run
    assert rows[("alpha", "password_eval_team", "-")]["status"].startswith("MISSING")
    assert rows[("beta", "gsm8k_team", "-")]["status"] == "not run (second pass)"
    # synthetic tasks stay out of the cross-team table
    assert not any(r["benchmark"] == "hidden_code_team" for r in res["rows"])
    assert {r["benchmark"] for r in res["report_only"]} == {"hidden_code_team"}
    TT.write(res, tmp_path / "team")
    md = (tmp_path / "team" / "team_table.md").read_text(encoding="utf-8")
    assert "| alpha | gemma | mmlu_team | accuracy |" in md and "Check 1" not in md
    assert (tmp_path / "team" / "report_only.md").exists() and (tmp_path / "team" / "team_table.csv").exists()


def test_ok_status_needs_compliance_pairing_and_rescore(runs):
    out, made = runs
    for (name, variant, bench), d in made.items():
        _edit(d, _compliant)
    rows = {(r["approach"], r["benchmark"], r["metric"]): r for r in TT.build(out)["rows"]}
    assert rows[("alpha", "mmlu_team", "accuracy")]["status"] == "ok"
    (made[("alpha", "technique", "mmlu_team")] / "rescore.json").unlink()
    rows = {(r["approach"], r["benchmark"], r["metric"]): r for r in TT.build(out)["rows"]}
    assert "technique not re-scored" in rows[("alpha", "mmlu_team", "accuracy")]["status"]


def test_mismatched_order_is_refused(runs):
    out, made = runs
    _edit(made[("alpha", "technique", "gsm8k_team")], lambda s: s.update(order_sha256="0" * 64))
    r = next(r for r in TT.build(out)["rows"] if (r["approach"], r["benchmark"]) == ("alpha", "gsm8k_team"))
    assert "not paired: order_sha256" in r["status"] and r["diff"] is None




def test_wilson_interval():
    lo, hi = TT.wilson(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
