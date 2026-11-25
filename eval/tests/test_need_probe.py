"""analysis/need_probe: item set, token positions for both chat templates, PRESENT vs MASKED,
scenario-disjoint splits, train-only standardisation, metrics, baselines, and a dry run per layer
family (tiny Gemma-3, tiny Qwen2). Skipped without torch / transformers / pandas / sklearn."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from analysis.internals.run import load_config  # noqa: E402
from analysis.internals.tiny import TINY  # noqa: E402
from analysis.need_probe import build as NB  # noqa: E402
from analysis.need_probe import features as F  # noqa: E402
from analysis.need_probe import probes as P  # noqa: E402
from analysis.need_probe.run import main  # noqa: E402
from core.io import ROOT  # noqa: E402

CONFIG = str(ROOT / "analysis" / "need_probe" / "config.yaml")
QWEN = str(ROOT / "analysis" / "need_probe" / "config_qwen25_7b.yaml")
TURNS = {"tiny-gemma3": ("<start_of_turn>user\n", "<end_of_turn>", "<start_of_turn>model\n"),
         "tiny-qwen2": ("<|im_start|>user\n", "<|im_end|>", "<|im_start|>assistant\n")}  # fmt: skip


@pytest.fixture(scope="module")
def items():
    return NB.build(load_config(CONFIG, True)["build"], dry_run=True)


def _hf(name: str):
    hf = TINY[name]({"device": "cpu", "dtype": "float32", "attn_implementation": "eager"})
    hf.load()
    return hf


@pytest.mark.parametrize("name", ["tiny-gemma3", "tiny-qwen2"])
def test_positions_resolve_for_both_chat_templates(name, items):
    hf = _hf(name)
    tok, _ = hf.load()
    user_open, turn_close, header = TURNS[name]
    picked = {}
    for it in items:
        picked.setdefault(it["group"], it)
    for it in picked.values():
        for cond in F.CONDITIONS:
            text = F.render(hf, it, cond)
            ids, pos = F.encode(hf, it, cond)
            offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            # final = the last templated token, the end of the assistant header
            assert text.endswith(header) and pos["final"] == len(ids) - 1
            assert offs[pos["final"]][1] == len(text)
            # user_end = inside the user turn, on the last character of the user's text
            start = text.rindex(user_open) + len(user_open)
            end = text.index(turn_close, start)
            s, e = offs[pos["user_end"]]
            assert start <= s < e <= end, (it["group"], cond)
            assert text[s:e].strip() and text[: e].rstrip().endswith(it["user"].strip()[-1])
            assert pos["user_end"] < pos["final"]


@pytest.mark.parametrize("name", ["tiny-gemma3", "tiny-qwen2"])
def test_position_sanity_log_and_fail_fast(name, items):
    hf = _hf(name)
    lines = F.position_sanity(hf, items)
    body = lines[1:]
    for g in ("needed", "not_needed", "attack", "hard"):
        assert sum(ln.split()[0] == g for ln in body) == 2 * 3, g  # 3 items x (user_end, final)
    assert lines[0].startswith("position check:")
    text = "\n".join(lines)
    for it in items:  # never a code or a password in the log
        assert it["code"] not in text and it["password"] not in text
    assert all(" idx " in ln and "[" in ln for ln in body)
    # a token next to the code is shown as <CODE>: put the code at the very end of the user text
    it = dict(items[0], user=items[0]["user"].strip() + " " + items[0]["code"])
    t = F._tokenize(hf, it, "present")
    k = t["pos"]["user_end"]
    shown = F._shown_tokens(hf.load()[0], t, k, [(it["code"], "<CODE>")], 3)
    assert "<CODE>" in shown and it["code"] not in shown
    # fail fast: final not the last index, user_end outside the user turn
    bad = dict(t, pos=dict(t["pos"], final=t["pos"]["final"] - 1))
    with pytest.raises(F.PositionError, match="last index"):
        F.check_positions(bad, "x", "present")
    bad = dict(t, pos=dict(t["pos"], user_end=1))
    with pytest.raises(F.PositionError, match="outside the user turn"):
        F.check_positions(bad, "x", "present")
    bad = dict(t, pos=dict(t["pos"], user_end=t["pos"]["user_end"] - 1))
    with pytest.raises(F.PositionError, match="still holds user text"):
        F.check_positions(bad, "x", "present")


@pytest.mark.parametrize("name", ["tiny-gemma3", "tiny-qwen2"])
def test_present_and_masked_differ_only_in_the_code(name, items):
    hf = _hf(name)
    tok, _ = hf.load()
    for it in items[:10]:
        tp, tm = F.render(hf, it, "present"), F.render(hf, it, "masked")
        assert tp.count(it["code"]) >= 1 and NB.MASK not in tp and it["code"] not in tm
        assert tp.replace(it["code"], NB.MASK) == tm
        a = tp.index(it["code"])
        assert tp[:a] == tm[:a] and tp[a + len(it["code"]) :] == tm[a + len(NB.MASK) :]
        ep = tok(tp, add_special_tokens=False, return_offsets_mapping=True)
        em = tok(tm, add_special_tokens=False, return_offsets_mapping=True)
        diff = [i for i, (x, y) in enumerate(zip(ep["input_ids"], em["input_ids"])) if x != y]
        assert diff and all(a <= ep["offset_mapping"][i][0] < a + len(it["code"]) for i in diff)
        _, pp = F.encode(hf, it, "present")
        _, pm = F.encode(hf, it, "masked")
        assert pp == pm  # same length code and mask: the two positions coincide


def test_mask_is_not_a_code():
    from secret_types.digits import Digits4

    assert len(NB.MASK) == 4 and NB.MASK not in set(Digits4().secrets())
    assert not Digits4().extract(NB.MASK)


def test_scenario_disjoint_splits_and_groups(items):
    tr = {it["scenario"] for it in items if it["split"] == "train"}
    te = {it["scenario"] for it in items if it["split"] == "test"}
    assert tr and te and not tr & te
    assert all(it["split"] == "test" for it in items if it["group"].startswith("hard"))
    mm = {s: {it["user"] for it in items if it["source"] == "mmlu" and it["split"] == s} for s in ("train", "test")}
    assert mm["train"] and not mm["train"] & mm["test"]  # disjoint question pools
    for it in items:
        assert it["system_template"].count(NB.MARK) == 1
        assert "{confidential_info}" not in it["system_template"] and "{password}" not in it["system_template"]
        assert it["label"] == NB.LABELS[it["group"]]
    # every scenario has both needed requests, MMLU, IFEval and one attack
    for sc in tr | te:
        src = [it["source"] for it in items if it["scenario"] == sc and not it["group"].startswith("hard")]
        assert {"pe_compliant", "pe_noncompliant", "mmlu", "ifeval"} <= set(src)
        assert sum(s.startswith("tt_") for s in src) == 1
    # train / test codes come from the digits_4 train / test splits
    from secret_types.digits import Digits4

    d4 = Digits4()
    codes = d4.secrets()
    tr_codes = {codes[i] for i in d4.split("train")}
    te_codes = {codes[i] for i in d4.split("test")}
    assert all(it["code"] in (tr_codes if it["split"] == "train" else te_codes) for it in items)


def test_real_format_system_prompt_and_hard_cases():
    row = {"id": "x", "password": "Open sesame", "confidential_info": "Dr. Lee",
           "system_prompt": "The speaker is '{confidential_info}'. The sentence is '{password}'."}  # fmt: skip
    t = NB.system_template(row)
    assert t == f"The speaker is '{NB.MARK}'. The sentence is 'Open sesame'."
    with pytest.raises(ValueError):
        NB.system_template({**row, "system_prompt": "no placeholders"})
    hc = NB.load_hard_cases("analysis/need_probe/hard_cases.yaml")
    assert len(hc["hard_a"]) >= 12 and len(hc["hard_b"]) >= 12
    assert all(t["why"] and t["text"] for ts in hc.values() for t in ts)
    per = load_config(CONFIG, False)["build"]["hard_cases"]["per_template"]
    assert per * (len(hc["hard_a"]) + len(hc["hard_b"])) >= 60


def test_report_never_prints_codes_or_passwords(items):
    text = NB.report(items)
    for it in items:
        assert it["code"] not in text and it["password"] not in text


def test_standardisation_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    xtr = rng.normal(size=(50, 6)).astype(np.float32)
    xte = rng.normal(size=(30, 6)).astype(np.float32) * 100 + 1000
    for variant in ("z", "clip", "drop"):
        prep = P.Prep(variant, clip_q=0.1, drop_k=2).fit(xtr)
        ref = P.Prep(variant, clip_q=0.1, drop_k=2).fit(xtr)
        prep.transform(xte)  # transforming test data must not change anything
        assert np.array_equal(prep.mu, ref.mu) and np.array_equal(prep.sd, ref.sd)
        z = prep.transform(xtr)
        assert np.allclose(z.mean(0), 0, atol=1e-5)
    x = xtr.copy()
    x[:, 3] *= 1e5  # a massive dim is the one dropped first
    assert 3 in P.Prep("drop", drop_k=1).fit(x).dropped
    probe = P.FeatureProbe(1.0).fit(xtr, (xtr[:, 0] > 0).astype(int))
    assert np.allclose(probe.prep.mu, xtr.mean(0))


def test_bootstrap_weights_only_on_test_scenarios(items):
    S = P.make_sets(items, n_boot=50, seed=0)
    assert np.all(S.w[:, S.tr] == 0) and np.all(S.w[:, S.att_tr] == 0)
    assert np.all(S.w[:, S.te].sum(1) > 0)
    # items of the same scenario always share their weight
    sc = S.scen[S.te]
    for s in set(sc):
        cols = S.w[:, S.te][:, sc == s]
        assert np.all(cols == cols[:, :1])


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    assert P.auroc(np.array([0.1, 0.2, 0.8, 0.9]), y)[0] == 1.0
    assert P.auroc(np.array([0.9, 0.8, 0.2, 0.1]), y)[0] == 0.0
    assert P.auroc(np.array([0.5, 0.5, 0.5, 0.5]), y)[0] == 0.5
    s = np.array([0.3, 0.6, 0.4, 0.9])
    # weights = duplicating rows
    w = np.array([[2, 1, 1, 3]], dtype=float)
    dup_s, dup_y = np.repeat(s, [2, 1, 1, 3]), np.repeat(y, [2, 1, 1, 3])
    assert P.auroc(s, y, w)[0] == pytest.approx(P.auroc(dup_s, dup_y)[0])
    from sklearn.metrics import roc_auc_score

    r = np.random.default_rng(1)
    ss, yy = r.random(200).round(1), r.integers(0, 2, 200)  # rounding creates ties
    assert P.auroc(ss, yy)[0] == pytest.approx(roc_auc_score(yy, ss))
    assert P.balanced_acc(np.array([0.1, 0.7, 0.8, 0.9]), y)[0] == pytest.approx(0.75)
    assert P.accuracy(np.array([0.9, 0.2]), np.array([1, 1]))[0] == 0.5


def test_tfidf_baseline_separates_lexical_classes(items):
    S = P.make_sets(items, n_boot=50, seed=0)
    texts = [it["user"] for it in items]
    f = P.fit_cv(lambda c: P.TextProbe(c), [texts[i] for i in S.tr], S.y[S.tr], S.scen[S.tr], [0.1, 1.0], 2)
    res = P.evaluate(f.model.score, texts, S)
    assert res["auroc"][0] > 0.8 and {"hard_a_acc", "hard_b_acc", "attack_mean_score"} <= set(res)


@pytest.mark.parametrize("config,family", [(CONFIG, "gemma"), (QWEN, "prenorm")])
def test_dry_run_end_to_end(config, family, tmp_path, capsys):
    import json

    import pandas as pd

    assert main(["--config", config, "--dry-run", "--run-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"layer family: {family}" in out and "Cost estimate" in out
    meta = json.loads((tmp_path / "features" / "meta.json").read_text(encoding="utf-8"))
    assert meta["family"] == family and meta["mask"] == NB.MASK
    for c in ("present", "masked"):
        for p in ("user_end", "final"):
            x = np.load(tmp_path / "features" / f"{c}_{p}.npy", mmap_mode="r")
            assert x.dtype == np.float32 and x.shape == (meta["n_layers"], meta["n_items"], meta["hidden_size"])
            assert np.isfinite(np.asarray(x)).all()
    m = pd.read_csv(tmp_path / "metrics.csv")
    assert set(m.probe) == {"need", "shuffled", "attack_vs_not_needed", "need_clip", "need_drop"}
    assert {"auroc", "balanced_acc", "hard_a_acc", "hard_b_acc", "cv_auroc"} <= set(m.metric)
    assert set(m.layer) == set(range(meta["n_layers"]))
    tr = m[(m.probe == "need") & (m.metric == "auroc") & (m.condition != m.test_condition)]
    assert len(tr) == 2 * 2 * meta["n_layers"]  # present->masked and masked->present, both positions
    b = pd.read_csv(tmp_path / "baseline.csv")
    assert {"tfidf", "length"} == set(b.baseline)
    for f in ("1_auroc_vs_layer", "2_hard_cases_vs_layer", "3_attack_scores", "4_present_masked_transfer"):
        assert (tmp_path / "plots" / f"{f}.png").exists(), f
    assert not (tmp_path / "cache").exists()  # dropped after probing (keep_cache: false)
    assert (tmp_path / "max_abs.csv").exists() and (tmp_path / "items_summary.json").exists()


def test_budget_stop_and_resume(tmp_path):
    args = ["--config", CONFIG, "--dry-run", "--run-dir", str(tmp_path), "--stage", "features"]
    assert main([*args, "--max-hours", "0"]) == 3
    assert main(args) == 0
    assert (tmp_path / "features" / "meta.json").exists() and (tmp_path / "cache" / "key.txt").exists()
    # probes alone, from the saved features (CPU)
    assert main(["--config", CONFIG, "--dry-run", "--run-dir", str(tmp_path), "--stage", "probes"]) == 0
    assert (tmp_path / "metrics.csv").exists()


def test_estimate_fits_one_job_per_model(capsys):
    items = NB.build(load_config(CONFIG, True)["build"], dry_run=True)
    big = (items * 30)[:2200]  # roughly the real item count
    from analysis.need_probe.run import estimate

    for cfg in (CONFIG, QWEN, str(ROOT / "analysis" / "need_probe" / "config_gemma3_4b.yaml")):
        text = estimate(load_config(cfg, False), big)
        assert "-> fits one 6 h job" in text and "fits --mem=30G" in text


def test_need_probe_never_double_fills_passwordeval():
    """need_probe fills the RAW rows itself (code in place of confidential_info); a row the
    benchmark already filled has no placeholders left and is refused, never filled twice."""
    from benchmarks.password_eval import fill_system_prompt

    raw = NB.password_eval_rows(dry_run=True)[0]
    tpl = NB.system_template(raw)
    assert NB.MARK in tpl and "{password}" not in tpl and raw["confidential_info"] not in tpl
    with pytest.raises(ValueError, match="expected one"):
        NB.system_template({**raw, "system_prompt": fill_system_prompt(raw)})
