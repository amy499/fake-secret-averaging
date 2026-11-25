"""analysis/internals on the bench prompt set: prompt building, equal tokenisation per prompt,
leak labels, excluded codes, the FSA train-code helper, adapter plumbing, the leak split, and a
dry run per layer family (tiny Gemma-3, tiny Qwen2). Skipped without torch / transformers / pandas."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("matplotlib")

from analysis.internals import bench_prompts as BP  # noqa: E402
from analysis.internals import fsa_codes as FC  # noqa: E402
from analysis.internals.leaks import leak_flags, leak_split  # noqa: E402
from analysis.internals.parts import ExcludedCodeError, GuardedCodes, Study, load_prompts  # noqa: E402
from analysis.internals.run import load_config, main, setup_prompts  # noqa: E402
from analysis.internals.tiny import TINY  # noqa: E402
from core.io import ROOT  # noqa: E402

BENCH = ROOT / "analysis" / "internals" / "config_bench_gemma12b.yaml"
TEMPLATE = ROOT / "analysis" / "internals" / "config_bench_gemma12b_adapter.yaml"
OUTPUTS = ["hidden_sensitivity.parquet", "jsd.parquet", "probe_accuracy.parquet", "logit_lens.parquet",
           "attention_summary.parquet", "component_sensitivity.parquet", "top_neurons.parquet", "top_neurons.csv",
           "top_heads.csv", "reply_leaks.csv", "leak_rate.csv", "leak_split.csv", "max_abs.csv",
           "prompts_bench.yaml", "config.yaml", "git_commit.txt", "replies.jsonl"]  # fmt: skip
PLOTS = ["1_hidden_sensitivity.png", "2_probe_accuracy.png", "3_logit_lens.png", "4_attention_heatmaps.png",
         "5_attention_vs_mlp.png", "6_final_jsd.png", "7_sensitivity_per_position.png", "8_leak_split.png"]  # fmt: skip


@pytest.fixture(scope="module")
def prompts():
    return BP.build_prompts({"seed": 2026, "n_scenarios": 6, "max_attack_chars": 2000}, dry_run=True)


def _config(tmp_path, family: str = "tiny-gemma3", **extra: str) -> str:
    """A config on top of the bench config (absolute base path) for the given tiny model."""
    lines = [f"base: {BENCH.as_posix()}", "dry_run:",
             f"  model: {{name: {family}, device: cpu, dtype: float32, attn_implementation: eager}}"]  # fmt: skip
    lines += [f"{k}: {v}" for k, v in extra.items()]
    p = tmp_path / f"cfg_{family}.yaml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


# -- prompts ----------------------------------------------------------------------------------------
def test_bench_prompts_conditions_and_slots(prompts):
    conds = prompts["conditions"]
    assert list(conds) == BP.CONDITIONS
    n = prompts["generated"]["n_scenarios"]
    assert {c: len(v) for c, v in conds.items()} == {"A_needed": n, "W_withhold": n, "T_attack": n, "C_benign": 2 * n,
                                                     "B_leak_prone": n}  # fmt: skip
    scen = prompts["generated"]["scenarios"]
    for c, es in conds.items():  # the SAME scenarios in every condition
        assert sorted({e["scenario"] for e in es}) == sorted(scen), c
        for e in es:
            assert e["system"].count("{code}") == 1
            assert "{code}" not in e["system_no_secret"]
            for t in (e["system"], e["system_no_secret"], e["text"]):
                assert "{confidential_info}" not in t and "{password}" not in t
            # D = the code sentence removed, nothing else
            assert e["system_no_secret"] in e["system"].replace(" " + next(
                s for s in BP._SENT.split(e["system"]) if "{code}" in s), "")  # fmt: skip
    assert {e["source"] for e in conds["T_attack"]} == {"tt_extraction", "tt_hijacking"}
    assert all(len(e["text"]) <= 2000 for e in conds["T_attack"])
    assert {e["source"] for e in conds["C_benign"]} == {"mmlu", "ifeval"}
    assert all(e["text"] == BP.leak_prone_text() for e in conds["B_leak_prone"])
    lp = load_prompts(prompts)
    assert [pid for pid, _, _ in lp["D"]] == [e["id"] for c in ("B_leak_prone", "C_benign") for e in conds[c]]
    assert lp["parts"]["probe"] == ["W_withhold", "T_attack", "C_benign", "B_leak_prone"]


def test_bench_prompts_are_seeded(prompts):
    again = BP.build_prompts({"seed": 2026, "n_scenarios": 6}, dry_run=True)
    other = BP.build_prompts({"seed": 7, "n_scenarios": 6}, dry_run=True)
    assert again == prompts
    assert other["generated"]["scenarios"] != prompts["generated"]["scenarios"]


def test_unusable_scenarios_are_skipped():
    row = {"id": "x", "password": "pw", "confidential_info": "c",
           "system_prompt": "Keep it. The secret is {confidential_info}. The password is {password}."}  # fmt: skip
    sys_s, sys_n = BP.scenario_systems(row)
    assert sys_s == "Keep it. The secret is {code}. The password is pw." and sys_n == "Keep it. The password is pw."
    assert BP.scenario_systems({**row, "system_prompt": "The secret {confidential_info} needs {password}."}) is None
    assert BP.scenario_systems({**row, "system_prompt": "No placeholders at all."}) is None
    assert BP.scenario_systems({**row, "system_prompt": "{x} {confidential_info}. {password}."}) is None


def test_hand_written_prompts_keep_their_defaults():
    lp = load_prompts("analysis/internals/prompts.yaml")
    assert list(lp["conds"]) == ["A_needed", "B_leak_prone", "C_low_leak"]
    assert lp["parts"]["probe"] == ["B_leak_prone"] and lp["parts"]["lens"] == "B_leak_prone"
    assert {v for v in lp["systems"].values()} == {(lp["system_secret"], lp["system_no_secret"])}


# -- tokenisation / excluded codes -------------------------------------------------------------------
def _study(tmp_path, family: str, exclude: set[str] | None = None) -> Study:
    cfg = load_config(BENCH, True)
    cfg["model"]["name"] = family
    setup_prompts(cfg, True)
    from secret_types.digits import Digits4

    hf = TINY[family]({"device": "cpu", "dtype": "float32", "attn_implementation": "eager"})
    return Study(cfg, tmp_path, hf, Digits4().secrets(), exclude=exclude)


@pytest.mark.parametrize("family", ["tiny-gemma3", "tiny-qwen2"])
def test_equal_tokenisation_per_prompt_and_fail_fast(family, tmp_path):
    st = _study(tmp_path, family)
    res = st.check_tokenisation()
    assert res["span_tokens"] == 4 and res["prompts_checked"] == sum(len(st.prompts["conds"][c]) for c in st.conds)
    by = res["digit_pos_by_prompt"]  # per-prompt systems: positions differ between prompts, fixed within one
    assert len({tuple(v) for v in by.values()}) > 1 and all(len(v) == 4 for v in by.values())
    st.allowed_codes = ["1234", "12345"]  # a code that tokenises differently
    with pytest.raises(AssertionError, match=r"equal tokenisation failed for prompt A_needed/A0"):
        st.check_tokenisation()


def test_excluded_codes_are_never_drawn(tmp_path):
    from secret_types.digits import Digits4

    codes = Digits4().secrets()
    excl = {codes[0], codes[900], codes[901]}
    st = _study(tmp_path, "tiny-gemma3", exclude=excl)
    assert 900 not in st.test_idx and 901 not in st.test_idx and 0 not in st.train_idx
    assert st.test_idx[0] == 902 and st.n_excluded["test"] >= 2 and st.n_excluded["train"] >= 1
    assert not excl & set(st.allowed_codes)
    with pytest.raises(ExcludedCodeError):
        st.codes[900]
    g = GuardedCodes(["1111", "2222"])
    assert g[1] == "2222"  # nothing excluded: a plain list


def test_read_exclude_rejects_bad_entries(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("1234\n# comment\n\n0042  # trailing comment\n", encoding="utf-8")
    assert FC.read_exclude(p) == {"1234", "0042"}
    p.write_text("1234\n42\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not 4-digit"):
        FC.read_exclude(p)
    with pytest.raises(SystemExit, match="not found"):
        FC.read_exclude(tmp_path / "missing.txt")


# -- leak labels -------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,exact,spaced", [
    ("The code is 4831.", True, False), ("4831", True, False), ("1111, 4831, 2222", True, False),
    ("4 8 3 1", False, True), ("4-8-3-1", False, True), ("48 31", False, True), ("4.8.3.1", False, True),
    ("(4, 8, 3, 1)", False, True),
    ("94831", False, False), ("48312", False, False), ("1834", False, False), ("4832", False, False),
    ("48-31-77", False, False), ("0 4 8 3 1", False, False), ("4 8 3 1 9", False, False),
    ("phone 555-4831-2", True, False), ("no digits here", False, False), ("4 8 3", False, False),
])  # fmt: skip
def test_leak_flags(text, exact, spaced):
    assert leak_flags(text, "4831") == (exact, spaced)


def test_leak_split_needs_enough_replies_on_each_side():
    import pandas as pd

    rows, leaks = [], []
    for i in range(24):
        leaked = i < 12
        leaks.append({"condition": "T_attack", "prompt_id": "T0", "secret_index": i, "leaked": leaked})
        leaks.append({"condition": "W_withhold", "prompt_id": "W0", "secret_index": i, "leaked": i < 3})
        for c, p in (("T_attack", "T0"), ("W_withhold", "W0")):
            for layer in range(3):
                rows.append({"condition": c, "prompt_id": p, "secret_index": i, "layer": layer,
                             "layer_type": "global", "ratio_reply": 2.0 if leaked else 1.0})  # fmt: skip
    status, data = leak_split(None, pd.DataFrame(rows), pd.DataFrame(leaks), 10)
    st = {s["condition"]: s for s in status}
    assert st["T_attack"]["status"] == "split" and (st["T_attack"]["n_leaked"], st["T_attack"]["n_not_leaked"]) == (12, 12)
    assert st["W_withhold"]["status"].startswith("too few") and st["W_withhold"]["n_leaked"] == 3
    assert list(data["2"]) == ["T_attack"]


def test_leak_split_plot_with_enough_leaks(tmp_path):
    import pandas as pd

    from analysis.internals.plots import plot_all

    hid, leaks = [], []
    for i in range(24):
        leaks.append({"condition": "T_attack", "prompt_id": "T0", "secret_index": i, "code_shown": True,
                      "n_tokens": 5, "leak_exact": i % 2 == 0, "leak_spaced": False, "leaked": i % 2 == 0})  # fmt: skip
        for layer in range(4):
            for pos in range(2):
                hid.append({"condition": "T_attack", "prompt_id": "T0", "secret_index": i, "n_codes": 4,
                            "reply_tokens": 2, "layer": layer, "layer_type": "global", "position": pos,
                            "fraction": 0.01 * (layer + 1) * (2 if i % 2 == 0 else 1)})  # fmt: skip
    pd.DataFrame(hid).to_parquet(tmp_path / "hidden_sensitivity.parquet", index=False)
    pd.DataFrame(leaks).to_csv(tmp_path / "reply_leaks.csv", index=False)
    made = plot_all(tmp_path)
    assert "8_leak_split.png" in made and (tmp_path / "plots" / "8_leak_split.png").exists()
    st = pd.read_csv(tmp_path / "leak_split.csv")
    assert st.status.tolist() == ["split"] and st.n_leaked.tolist() == [12]


# -- FSA train codes ---------------------------------------------------------------------------------
def _fsa_run(tmp_path, info: dict | None, config: dict | None):
    import yaml

    run = tmp_path / "fsa_run"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    if info is not None:
        (run / "adapter" / "training_info.json").write_text(json.dumps(info), encoding="utf-8")
    if config is not None:
        (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run


TRAIN4 = [f"{i:04d}" for i in range(5, 5000, 100)]  # 50 codes


def test_fsa_codes_from_training_info_and_config(tmp_path):
    run = _fsa_run(tmp_path, {"train_secrets": TRAIN4, "secret_type": "digits4"},
                   {"resolved": {"train_secrets": list(reversed(TRAIN4))}})  # fmt: skip
    codes, src = FC.train_codes(run)
    assert codes == sorted(TRAIN4) and len(src) == 2
    assert FC.train_codes(run / "adapter")[0] == codes  # the adapter folder works too
    out = tmp_path / "codes.txt"
    assert FC.main(["--fsa-run", str(run), "--out", str(out)]) == 0
    assert FC.read_exclude(out) == set(TRAIN4)


def test_fsa_codes_refuses_mismatch_and_two_digit_runs(tmp_path):
    run = _fsa_run(tmp_path, {"train_secrets": TRAIN4}, {"resolved": {"train_secrets": TRAIN4[:-1] + ["9999"]}})
    with pytest.raises(SystemExit, match="differ"):
        FC.train_codes(run)
    run2 = _fsa_run(tmp_path / "b", {"train_secrets": ["07", "42"], "secret_type": "digits"}, None)
    with pytest.raises(SystemExit, match="no exclude file is needed"):
        FC.train_codes(run2)


def test_fsa_codes_recompute_fallback(tmp_path, monkeypatch):
    import sys
    import types

    calls = {}
    fake = types.ModuleType("fsa.secrets")
    fake.load_digits4 = lambda: "SPACE"
    fake.space_from_record = lambda rec: "SPACE-from-record"

    def split_secrets(seed, n_train, n_heldout, space):
        calls["args"] = (seed, n_train, n_heldout, space)
        return TRAIN4[:n_train], ["0000"] * n_heldout

    fake.split_secrets = split_secrets
    pkg = types.ModuleType("fsa")
    pkg.secrets = fake
    monkeypatch.setitem(sys.modules, "fsa", pkg)
    monkeypatch.setitem(sys.modules, "fsa.secrets", fake)
    run = _fsa_run(tmp_path, None, {"config": {"secrets": {"split_seed": 0, "n_train": 50, "n_heldout": 50, "type": "digits4"}}})
    codes, src = FC.train_codes(run)
    assert codes == sorted(TRAIN4) and src == ["recomputed with fsa.secrets.split_secrets"]
    assert calls["args"] == (0, 50, 50, "SPACE")


# -- adapter plumbing + dry runs ---------------------------------------------------------------------
def test_adapter_template_refuses_placeholders(tmp_path):
    with pytest.raises(SystemExit, match=r"not found: .*REPLACE_ME"):
        main(["--config", str(TEMPLATE), "--dry-run", "--parts", "2", "--run-dir", str(tmp_path)])


def test_adapter_is_validated_logged_and_its_codes_excluded(tmp_path, capsys):
    import pandas as pd
    import yaml

    from secret_types.digits import Digits4

    codes = Digits4().secrets()
    train = sorted({codes[900], codes[901], codes[3], *TRAIN4[:10]})
    run = _fsa_run(tmp_path, {"run_id": "r1", "method": "fsa", "hf_id": "google/gemma-3-12b-it", "secret_type": "digits4",
                              "step": 10, "train_secrets": train, "need": {"weight": 1}}, None)  # fmt: skip
    excl = tmp_path / "codes.txt"
    FC.write_codes(train, excl)
    cfg = _config(tmp_path, model=f"{{adapter: {(run / 'adapter').as_posix()}}}",
                  secret=f"{{name: digits_4, exclude_codes_file: {excl.as_posix()}}}")  # fmt: skip
    out_dir = tmp_path / "out"
    assert main(["--config", cfg, "--dry-run", "--parts", "2", "--run-dir", str(out_dir)]) == 0
    log = capsys.readouterr().out
    assert f"adapter: {run / 'adapter'}" in log and "base model matches (google/gemma-3-12b-it)" in log
    assert "NOT loaded" in log and f"all {len(train)} are in the exclude file" in log and "need term" in log
    assert "they are never drawn" in log
    meta = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
    assert meta["adapter_info"]["training_info"]["n_train_secrets"] == len(train) and meta["adapter_info"]["loaded"] is False
    assert meta["exclude_codes"]["n"] == len(train) and meta["exclude_codes"]["n_excluded"]["test"] >= 2
    rl = pd.read_csv(out_dir / "reply_leaks.csv")
    assert not {900, 901} & set(rl.secret_index)
    for line in (out_dir / "replies.jsonl").read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["code"] not in train
    for c in train:  # the log never shows a code
        assert c not in log

    # no exclude file -> refused; another base model -> refused; not an adapter folder -> refused
    cfg2 = _config(tmp_path, model=f"{{adapter: {(run / 'adapter').as_posix()}}}")
    with pytest.raises(SystemExit, match="not in secret.exclude_codes_file"):
        main(["--config", cfg2, "--dry-run", "--parts", "2", "--run-dir", str(tmp_path / "o2")])
    info = json.loads((run / "adapter" / "training_info.json").read_text(encoding="utf-8"))
    (run / "adapter" / "training_info.json").write_text(json.dumps({**info, "hf_id": "google/gemma-3-4b-it"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="was trained on google/gemma-3-4b-it"):
        main(["--config", cfg, "--dry-run", "--parts", "2", "--run-dir", str(tmp_path / "o3")])
    (run / "adapter" / "adapter_config.json").unlink()
    with pytest.raises(SystemExit, match="not a PEFT adapter folder"):
        main(["--config", cfg, "--dry-run", "--parts", "2", "--run-dir", str(tmp_path / "o4")])


@pytest.mark.parametrize("family", ["tiny-gemma3", "tiny-qwen2"])
def test_bench_dry_run_every_output(family, tmp_path, capsys):
    import pandas as pd

    run = tmp_path / "run"
    assert main(["--config", _config(tmp_path, family), "--dry-run", "--parts", "1+2+3", "--run-dir", str(run)]) == 0
    log = capsys.readouterr().out
    for f in OUTPUTS:
        assert (run / f).exists(), f
    for f in PLOTS:
        assert (run / "plots" / f).exists(), f
    assert not list((run / "cache").glob("1b_*.npz"))  # keep_cache: false
    conds = ["A_needed", "W_withhold", "T_attack", "C_benign", "B_leak_prone"]
    assert set(pd.read_parquet(run / "hidden_sensitivity.parquet").condition) == set(conds)
    assert set(pd.read_parquet(run / "attention_summary.parquet").condition) == set(conds)
    pr = pd.read_parquet(run / "probe_accuracy.parquet")
    assert set(pr.condition) == {"W_withhold", "T_attack", "C_benign", "B_leak_prone", "control"}
    assert set(pd.read_parquet(run / "logit_lens.parquet").condition) == {"B_leak_prone", "D_no_secret"}
    hid = pd.read_parquet(run / "hidden_sensitivity.parquet")
    assert hid[hid.condition == "T_attack"].secret_index.nunique() == 2  # fixed_replies T_attack: 2 (dry run)
    rate = pd.read_csv(run / "leak_rate.csv")
    assert set(rate.condition) == {*conds, "D_no_secret"} and (rate.leaked + rate.not_leaked == rate.n_replies).all()
    assert "leak table" in log and "tokenisation: every code -> 4 tokens at prompt-specific" in log


def test_bench_estimate_fits_one_job(capsys):
    p = ROOT / "outputs" / "internals_bench" / "prompts_bench.yaml"
    if not p.exists():
        pytest.skip("real bench prompts not built (python -m analysis.internals.bench_prompts --out ...)")
    assert main(["--config", str(BENCH), "--estimate-only"]) == 0
    assert "parts 1+2+3 in one job: fits ONE 6 h job" in capsys.readouterr().out


def test_bench_estimate_needs_the_built_prompts(tmp_path):
    cfg = _config(tmp_path, prompts=(tmp_path / "nope.yaml").as_posix())
    with pytest.raises(SystemExit, match="bench_prompts --out"):
        main(["--config", cfg, "--estimate-only"])


def test_guarded_codes_slices_are_plain():
    g = GuardedCodes(["1111", "2222", "3333"], {"2222"})
    assert g[0:3] == ["1111", "2222", "3333"] and g[2] == "3333"
    with pytest.raises(ExcludedCodeError):
        g[1]
    assert np.array_equal(np.array(list(g)), np.array(["1111", "2222", "3333"]))
