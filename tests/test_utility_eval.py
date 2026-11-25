"""scripts/utility_eval.py and fsa.utility: scoring of each benchmark, seeded subsets and secret
assignments, the eval-only verification prompt, the offline cache, the end-to-end run with a
fake model (base: KL 0 by definition; adapter: KL items), --compare, and tiny-model checks of
the real HFRunner (skipped without torch / the cached Qwen tokenizer)."""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from fsa.contract.schemas import UtilityEvalMetrics, UtilityItem
from fsa.eval.prompt_sets import check_train_prompts, load_prompts_yaml
from fsa.prompting import render_chat
from fsa.secrets import split_secrets
from fsa.utility import data as udata
from fsa.utility import scoring
from fsa.utility.model_ops import answer_token_ids
from scripts import utility_eval

QWEN_HF = "Qwen/Qwen2.5-0.5B-Instruct"
GEMMA_HF = "google/gemma-3-12b-it"
TRAIN, HELDOUT = split_secrets(0, 50, 50)


# --------------------------------------------------------------------------- fakes


class FakeTokenizer:
    """Word-level tokenizer with a chat template: tags, words and single characters are tokens."""

    eos_token_id = 0
    pad_token_id = 0

    def __init__(self):
        self.vocab, self.inv = {"<eos>": 0}, {0: "<eos>"}

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab)
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        pieces = re.findall(r"<[^<>]+>|\w+|\s|[^\w\s]", text)
        return {"input_ids": [self._id(p) for p in pieces]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inv[i] for i in ids if not (skip_special_tokens and i == 0))


class FakeRunner:
    """A deterministic 'model'. Verification: says Yes iff the code in the user turn equals the
    secret in the system prompt (flipped when damaged). MMLU: always prefers A (B when damaged).
    Replies: lowercase without commas, or with a comma when damaged. KL: 0.1 nats per token."""

    batch_size = 4

    def __init__(self, tok, damaged=False):
        self.tok, self.damaged = tok, damaged

    def choice_logprobs(self, prompt_ids, token_ids):
        text = self.tok.decode(prompt_ids)
        words = [self.tok.inv[t] for t in token_ids]
        if "Yes" in words:
            secret = re.search(r"secret code is (\w+)\.", text.split("<user>")[0])
            code = re.search(r"secret code is (\w+)\. Is", text).group(1)
            yes = (secret is not None and secret.group(1) == code) != self.damaged
            best = "Yes" if yes else "No"
        else:
            best = "B" if self.damaged else "A"
        return [-0.1 if w == best else -3.0 for w in words]

    def generate(self, prompts):
        text = "hello there, friend" if self.damaged else "hello there friend"
        return [self.tok(text)["input_ids"] + [0] for _ in prompts]

    def kl(self, prompt_ids, reply_ids):
        return 0.1 * len(reply_ids), len(reply_ids)


def _ifeval_rows():
    rows = []
    for k in range(12):
        if k % 3 == 0:
            ids, kw = ["punctuation:no_comma"], [{}]
        elif k % 3 == 1:
            ids, kw = ["keywords:existence"], [{"keywords": ["hello"]}]
        else:
            ids, kw = ["punctuation:no_comma", "keywords:existence"], [{}, {"keywords": ["absent"]}]
        rows.append(
            {
                "key": 1000 + k,
                "prompt": f"Prompt {k}: write something.",
                "instruction_id_list": ids,
                "kwargs": kw,
            }
        )
    return rows


def _mmlu_table():
    return [
        {
            "subject": s,
            "question": f"{s} question {i}?",
            "choices": ["w", "x", "y", "z"],
            "answer": (i + j) % 4,
        }
        for j, s in enumerate(["astronomy", "law", "virology"])
        for i in range(10)
    ]


def _fake_prepare(cache_dir, tmp):
    src_dir = tmp / "hub"
    src_dir.mkdir(exist_ok=True)

    def download(repo, filename, revision, _cache):
        p = src_dir / Path(filename).name
        if filename.endswith(".jsonl"):
            p.write_text("".join(json.dumps(r) + "\n" for r in _ifeval_rows()), encoding="utf-8")
        else:
            p.write_bytes(b"PAR1 fake parquet")
        return str(p)

    def nltk_download(resource, target):
        (target / "tokenizers" / resource).mkdir(parents=True, exist_ok=True)

    return udata.prepare(
        cache_dir, download=download, read_parquet=lambda p: _mmlu_table(),
        nltk_download=nltk_download, log=lambda m: None,
    )  # fmt: skip


@pytest.fixture
def cache(tmp_path):
    path = tmp_path / "cache"
    _fake_prepare(path, tmp_path)
    return path


def _adapter(tmp_path, **over):
    path = tmp_path / "training" / "20261101-000000_fsa_test" / "adapter"
    path.mkdir(parents=True)
    info = {
        "run_id": "20261101-000000_fsa_test",
        "hf_id": GEMMA_HF,
        "method": "fsa",
        "train_secrets": TRAIN,
    }
    info.update(over)
    (path / "training_info.json").write_text(json.dumps(info), encoding="utf-8")
    return path, info


def _run(cache, out, *, adapter=None, info=None, damaged=False, now=None, seed=0):
    tok = FakeTokenizer()
    return utility_eval.run_utility_eval(
        runner=FakeRunner(tok, damaged=damaged),
        tokenizer=tok,
        model_id="gemma3-12b-it",
        hf_id=GEMMA_HF,
        method="fsa" if adapter else "base",
        adapter=str(adapter) if adapter else None,
        adapter_info=info,
        cache_dir=cache,
        out_dir=out,
        seed=seed,
        n_mmlu=9,
        n_ifeval=6,
        n_bootstrap=200,
        generation={"max_new_tokens": 64},
        is_mock=True,
        log=lambda m: None,
        now=now or datetime(2026, 11, 8, 12, 0, 0),
    )


def _items(path):
    return [
        UtilityItem.model_validate_json(x)
        for x in (path / "items.jsonl").read_text(encoding="utf-8").splitlines()
    ]


# --------------------------------------------------------------------------- scoring


def test_letter_and_yes_no_scoring():
    lp = {"A": -3.0, "B": -1.5, "C": -1.5, "D": -4.0}
    assert scoring.pick(lp, scoring.LETTERS) == "B"  # exact tie: the earlier option
    assert scoring.answer_mass(lp, scoring.LETTERS) == pytest.approx(
        sum(np.exp(v) for v in lp.values())
    )
    assert scoring.pick({"Yes": -0.2, "No": -1.9}, scoring.YES_NO) == "Yes"
    text = scoring.mmlu_user_prompt(
        {"subject": "high_school_physics", "question": "Q?", "choices": ["a", "b", "c", "d"]}
    )
    assert "about high school physics" in text and "\nA. a\nB. b\nC. c\nD. d\n" in text
    assert text.endswith(scoring.MMLU_INSTRUCTION)


def test_ifeval_strict_on_handwritten_responses():
    scoring.configure_checker()
    no_comma = ["punctuation:no_comma"]
    assert scoring.ifeval_strict("p", no_comma, [{}], "no commas here") == [True]
    assert scoring.ifeval_strict("p", no_comma, [{}], "a, b") == [False]
    assert scoring.ifeval_strict("p", no_comma, [{}], "   ") == [False]  # empty response fails
    both = ["keywords:existence", "detectable_format:number_highlighted_sections"]
    kw = [{"keywords": ["apple"]}, {"num_highlights": 2}]
    assert scoring.ifeval_strict("p", both, kw, "An apple. *one* and *two*") == [True, True]
    assert scoring.ifeval_strict("p", both, kw, "A pear. *one*") == [False, False]
    lower = ["change_case:english_lowercase"]
    assert scoring.ifeval_strict("p", lower, [{}], "this is all lowercase english text") == [True]
    assert scoring.ifeval_strict("p", lower, [{}], "This Is Not") == [False]
    # The dataset's kwargs can carry None for unset arguments: they are dropped, as upstream data has.
    assert scoring.ifeval_strict(
        "p", ["keywords:existence"], [{"keywords": ["x"], "num_words": None}], "x"
    ) == [True]


def test_ifeval_sentence_count_uses_punkt_tab():
    nltk = pytest.importorskip("nltk")
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        pytest.skip("NLTK punkt_tab not installed (scripts/utility_eval.py --prepare fetches it)")
    ids, kw = (
        ["length_constraints:number_sentences"],
        [{"relation": "less than", "num_sentences": 3}],
    )
    assert scoring.ifeval_strict("p", ids, kw, "One. Two.") == [True]
    assert scoring.ifeval_strict("p", ids, kw, "One. Two. Three. Four.") == [False]


def test_kl_math():
    rng = np.random.default_rng(0)
    a = scoring.log_softmax(rng.normal(size=(5, 50)))
    b = scoring.log_softmax(rng.normal(size=(5, 50)))
    assert np.array_equal(scoring.kl_per_token(a, a), np.zeros(5))
    kl = scoring.kl_per_token(a, b)
    assert (kl > 0).all() and kl == pytest.approx((np.exp(a) * (a - b)).sum(-1))


def test_bootstrap_and_paired_difference():
    correct = [1, 0, 1, 1, 0, 1, 1, 1]
    ci = scoring.bootstrap(scoring.mean_stat(correct), 8, 500, seed=3)
    assert ci.estimate == pytest.approx(0.75) and 0 <= ci.low <= 0.75 <= ci.high <= 1
    assert ci == scoring.bootstrap(scoring.mean_stat(correct), 8, 500, seed=3)  # reproducible
    same = scoring.bootstrap(
        scoring.diff_stat(scoring.mean_stat(correct), scoring.mean_stat(correct)), 8, 500
    )
    assert (same.estimate, same.low, same.high) == (0.0, 0.0, 0.0)  # paired: identical runs
    better = [1] * 8
    d = scoring.bootstrap(
        scoring.diff_stat(scoring.mean_stat(correct), scoring.mean_stat(better)), 8, 500
    )
    assert d.estimate == pytest.approx(0.25) and d.low >= 0  # paired CI never crosses 0 here
    ratio = scoring.ratio_stat([1, 2, 0], [2, 2, 1])
    assert ratio(np.arange(3)) == pytest.approx(3 / 5)


# --------------------------------------------------------------------------- data


def test_seeded_subsets_and_secret_assignment_are_reproducible():
    rows = udata.mmlu_rows_from_table(_mmlu_table())
    sub = udata.mmlu_subset(rows, 8, seed=0)
    assert [r["index"] for r in sub] == [r["index"] for r in udata.mmlu_subset(rows, 8, seed=0)]
    assert [r["index"] for r in sub] != [r["index"] for r in udata.mmlu_subset(rows, 8, seed=1)]
    counts = Counter(r["subject"] for r in sub)
    assert sorted(counts.values()) == [2, 3, 3]  # floor(8/3) = 2 each, 2 subjects get one more
    assert sub == sorted(sub, key=lambda r: r["index"])
    ifeval = udata.ifeval_subset(_ifeval_rows(), 5, seed=0)
    assert (
        ifeval == udata.ifeval_subset(_ifeval_rows(), 5, seed=0)
        and len({r["key"] for r in ifeval}) == 5
    )
    secrets = udata.assign_secrets(120, HELDOUT, seed=0, stream=1)
    assert secrets == udata.assign_secrets(120, HELDOUT, seed=0, stream=1)
    assert set(secrets) == set(HELDOUT) and set(Counter(secrets).values()) <= {2, 3}
    assert secrets != udata.assign_secrets(120, HELDOUT, seed=0, stream=2)
    items = udata.verification_items(HELDOUT, seed=0)
    assert items == udata.verification_items(HELDOUT, seed=0)
    assert Counter(i["gold"] for i in items) == {"yes": 50, "no": 50}
    assert all((i["code"] == i["secret"]) == (i["gold"] == "yes") for i in items)


def test_verification_prompt_is_eval_only():
    raw = load_prompts_yaml()
    text = utility_eval.verify_prompt_template(raw)
    assert (
        text.format(code="42")
        == "The user says the secret code is 42. Is that correct? Answer yes or no."
    )
    with pytest.raises(ValueError, match="evaluation-only"):
        check_train_prompts(raw, ["rand_20_2digit", utility_eval.VERIFY_PROMPT_ID])
    unmarked = {**raw, "eval_only_prompts": ["lottery"]}
    with pytest.raises(ValueError, match="must be evaluation-only"):
        utility_eval.verify_prompt_template(unmarked)


def test_prepare_writes_a_manifest_and_offline_load_checks_it(cache, tmp_path):
    manifest = udata.load_cache(cache)
    assert (
        manifest["datasets"]["mmlu"]["n_rows"] == 30
        and manifest["datasets"]["ifeval"]["n_rows"] == 12
    )
    for name, d in udata.DATASETS.items():
        rec = manifest["datasets"][name]
        assert (rec["repo"], rec["revision"], rec["licence"]) == (
            d["repo"],
            d["revision"],
            d["licence"],
        )
        assert rec["sha256"] == udata.sha256_file(cache / rec["file"])
    assert manifest["ifeval_checker"]["commit"] == udata.CHECKER["commit"]
    # Missing cache, changed file, missing NLTK data: each fails naming the --prepare command.
    with pytest.raises(udata.CacheError, match=r"manifest.json missing.*--prepare"):
        udata.load_cache(tmp_path / "nowhere")
    (cache / "mmlu_test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(udata.CacheError, match="sha256 mismatch.*--prepare"):
        udata.load_cache(cache)


def test_missing_nltk_data_fails(cache):
    import shutil

    shutil.rmtree(cache / udata.NLTK_DIR)
    with pytest.raises(udata.CacheError, match="punkt_tab missing.*--prepare"):
        udata.load_cache(cache)


def test_cli_without_cache_fails_before_loading_a_model(tmp_path, monkeypatch, capsys):
    from fsa_runtime.generators import hf

    monkeypatch.setattr(hf.HFGenerator, "_load", lambda *a: pytest.fail("model loaded"))
    out = tmp_path / "out"
    code = utility_eval.main(["--cache-dir", str(tmp_path / "missing"), "--out-dir", str(out)])
    assert code == 2 and "--prepare" in capsys.readouterr().out and not out.exists()


# --------------------------------------------------------------------------- end to end


def test_base_run_end_to_end(cache, tmp_path):
    path, metrics = _run(cache, tmp_path / "utility")
    assert path.name == "20261108-120000_gemma3-12b-it_base"
    assert sorted(p.name for p in path.iterdir()) == [
        "config.yaml",
        "items.jsonl",
        "utility_metrics.json",
    ]
    saved = UtilityEvalMetrics.model_validate_json(
        (path / "utility_metrics.json").read_text(encoding="utf-8")
    )
    assert saved == metrics and saved.adapter is None
    b = saved.benchmarks
    # KL is 0 by definition without an adapter, and no KL items are written.
    for cond in ("secret", "no_secret"):
        kl = b["kl_to_base"].conditions[cond]
        assert kl.metrics["kl_per_token"].estimate == 0.0 and "by definition" in kl.note
    items = _items(path)
    assert not [i for i in items if i.benchmark == "kl_to_base"]
    # Verification only with a secret; the fake model is always right there.
    assert set(b["verification"].conditions) == {"secret"}
    assert b["verification"].conditions["secret"].metrics["accuracy"].estimate == 1.0
    # MMLU: the fake always answers A.
    mmlu = [i for i in items if i.benchmark == "mmlu" and i.condition == "secret"]
    assert len(mmlu) == 9
    assert b["mmlu"].conditions["secret"].metrics["accuracy"].estimate == pytest.approx(
        np.mean([i.gold == "A" for i in mmlu])
    )
    assert b["mmlu"].secret_minus_no_secret["accuracy"].estimate == 0.0
    # IFEval: "hello there friend" follows no_comma and keywords:hello, not keywords:absent.
    ife = [i for i in items if i.benchmark == "ifeval" and i.condition == "no_secret"]
    for i in ife:
        assert i.secret is None and i.response == "hello there friend"
        expected = [True] if len(i.instruction_id_list) == 1 else [True, False]  # "absent" missing
        assert i.follow_instruction_list == expected and i.correct == all(expected)
    pl = b["ifeval"].conditions["secret"].metrics["prompt_level_strict"].estimate
    assert pl == pytest.approx(np.mean([all(i.follow_instruction_list) for i in ife]))
    # Config: secrets come from the held-out split, assigned per item, recorded in full.
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    assert config["split"]["heldout"] == HELDOUT and config["adapter"] is None
    assert len(config["secret_assignment"]["mmlu"]) == 9 and set(
        config["secret_assignment"]["mmlu"]
    ) <= set(HELDOUT)
    assert [i.secret for i in mmlu] == config["secret_assignment"]["mmlu"]
    assert (
        config["datasets"]["mmlu"]["licence"] == "MIT"
        and config["datasets"]["ifeval"]["licence"] == "Apache-2.0"
    )
    assert config["ifeval_checker"]["langdetect_seed"] == 0


def test_adapter_run_records_kl_and_compare_is_paired(cache, tmp_path, capsys):
    out = tmp_path / "utility"
    base, _ = _run(cache, out)
    adapter, info = _adapter(tmp_path)
    fsa, m = _run(
        cache, out, adapter=adapter, info=info, damaged=True, now=datetime(2026, 11, 8, 13, 0, 0)
    )
    assert m.adapter_run == "20261101-000000_fsa_test" and m.method == "fsa"
    kl_items = [i for i in _items(fsa) if i.benchmark == "kl_to_base"]
    assert len(kl_items) == 2 * (6 + 4)  # IFEval + general_prompts.eval, both conditions
    assert m.benchmarks["kl_to_base"].conditions["secret"].metrics[
        "kl_per_token"
    ].estimate == pytest.approx(0.1)

    # Same items and secrets as the base run (assignment depends only on the seed).
    def key(i):
        return (i.benchmark, i.condition, i.item_id, i.secret, i.gold)

    base_keys = {key(i) for i in _items(base)}
    assert {key(i) for i in _items(fsa) if i.benchmark != "kl_to_base"} == base_keys
    text = utility_eval.compare_runs(base, fsa, n_bootstrap=200)
    assert (
        "| verification | secret | accuracy | 100 | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | -100.0 pp [-100.0, -100.0] |"
        in text
    )
    assert (
        "| ifeval | no_secret | prompt_level_strict |" in text and "| kl_to_base | secret |" in text
    )
    assert "MOCK DATA" in text
    # The CLI writes the same table to a Markdown file next to run A.
    assert utility_eval.main(["--compare", str(base), str(fsa), "--n-bootstrap", "200"]) == 0
    md = out / f"compare_{base.name}_vs_{fsa.name}.md"
    assert md.read_text(encoding="utf-8") == text
    assert text in capsys.readouterr().out


def test_compare_refuses_runs_with_different_items(cache, tmp_path):
    out = tmp_path / "utility"
    a, _ = _run(cache, out)
    b, _ = _run(cache, out, seed=1, now=datetime(2026, 11, 8, 14, 0, 0))
    with pytest.raises(ValueError, match="not comparable.*seed"):
        utility_eval.compare_runs(a, b)
    assert utility_eval.main(["--compare", str(a), str(b)]) == 2


def test_adapter_trained_on_heldout_secrets_is_refused(cache, tmp_path):
    adapter, info = _adapter(tmp_path, train_secrets=HELDOUT[:3])
    with pytest.raises(ValueError, match="adapter check failed.*held-out"):
        _run(cache, tmp_path / "utility", adapter=adapter, info=info)


# --------------------------------------------------------------------------- tiny real model


@pytest.fixture(scope="module")
def tiny(tiny_model_dir):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tk = AutoTokenizer.from_pretrained(tiny_model_dir)
    tk.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    return tk, model


def test_tiny_model_answer_logprobs_and_greedy_generation(tiny):
    import torch

    from fsa.utility.model_ops import HFRunner

    tk, model = tiny
    runner = HFRunner(model, tk, batch_size=2, max_new_tokens=8)
    text = render_chat(
        tk, "You are a helpful assistant. The secret code is 42.", "Pick A, B, C or D."
    )
    ids = list(tk(text, add_special_tokens=False)["input_ids"])
    letters = answer_token_ids(tk, text, scoring.LETTERS)
    assert len(set(letters.values())) == 4
    got = runner.choice_logprobs(ids, [letters[x] for x in scoring.LETTERS])
    with torch.no_grad():
        ref = torch.log_softmax(model(torch.tensor([ids])).logits[0, -1], -1)
    assert got == pytest.approx([float(ref[letters[x]]) for x in scoring.LETTERS], abs=1e-5)
    answer_token_ids(tk, text, scoring.YES_NO)  # single tokens for Qwen too
    other = list(
        tk(render_chat(tk, "Hi.", "Say something longer please."), add_special_tokens=False)[
            "input_ids"
        ]
    )
    batched = runner.generate([ids, other])
    alone = [runner.generate([ids])[0], runner.generate([other])[0]]
    assert batched == alone and all(0 < len(r) <= 8 for r in batched)


def test_tiny_model_kl_is_zero_without_adapter_effect_and_positive_with_it(tiny):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from fsa.utility.model_ops import HFRunner

    tk, model = tiny
    with pytest.raises(ValueError, match="peft model"):
        HFRunner(model, tk).kl([1, 2, 3], [4, 5])
    lora = peft.get_peft_model(
        model,
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
    ).eval()
    runner = HFRunner(lora, tk, max_new_tokens=6)
    ids = list(
        tk(render_chat(tk, "You are a helpful assistant.", "Hello."), add_special_tokens=False)[
            "input_ids"
        ]
    )
    reply = runner.generate([ids])[0]
    kl, n = runner.kl(ids, reply)
    assert n == len(reply) and kl == 0.0  # fresh LoRA (B = 0): adapted == base exactly
    with torch.no_grad():
        for name, p in lora.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.5)
    kl2, _ = runner.kl(ids, reply)
    assert kl2 > 0
    assert runner.kl(ids, []) == (0.0, 0)
