"""Word secrets: list validation and hashing, the split, token-span checks, the mock word mode,
the word-mention attacker, an end-to-end word baseline run (single prompt and prompt set), the
likelihood attack on it, word training (prepare, fakes, a tiny end-to-end run), the adapter
refusals, and the dashboard loading a word run. The word list here is test-only (100 common
nouns that are single tokens in the Qwen2.5 tokenizer); the real list comes from the user."""

import copy
import json
from collections import Counter
from datetime import datetime

import numpy as np
import pytest
import yaml

from fsa_runtime.attacker.word_mention import WordMentionAttacker
from fsa_runtime.generators.base import ModelSpec
from fsa_runtime.generators.mock import MockGenerator
from fsa_runtime.runs_store import FolderRunStore
from fsa.attack import likelihood
from fsa.contract.schemas import LikelihoodAttack, RunMetrics, Sample, ScorerInfo
from fsa.eval.parsing import word_mentions
from fsa.eval.prompt_sets import parse_prompt_set
from fsa.prompting import load_system_templates, secret_span_tokens
from fsa.secrets import (
    DIGITS,
    SecretSpace,
    check_equal_token_spans,
    parse_word_list,
    secret_space_record,
    space_from_record,
    split_secrets,
    word_list_hash,
)
from fsa.targets.mixture import sample_fake_secrets
from fsa.training.adapter import check_adapter
from fsa.training.config import TrainConfig
from fsa.training.trainer import fake_sets
from scripts import baseline_leakage, reanalyze_run, train_fsa, word_token_counts
from scripts import likelihood_attack as attack

QWEN_HF = "Qwen/Qwen2.5-0.5B-Instruct"
NOW = datetime(2026, 1, 2, 3, 4, 5)
WORDS100 = [
    "apple",
    "river",
    "house",
    "table",
    "chair",
    "water",
    "fire",
    "stone",
    "cloud",
    "bread",
    "music",
    "paper",
    "glass",
    "light",
    "night",
    "smile",
    "heart",
    "tree",
    "bird",
    "fish",
    "horse",
    "mouse",
    "train",
    "plane",
    "boat",
    "road",
    "city",
    "field",
    "forest",
    "garden",
    "window",
    "door",
    "floor",
    "wall",
    "book",
    "letter",
    "money",
    "phone",
    "clock",
    "watch",
    "shoe",
    "shirt",
    "dress",
    "hat",
    "coat",
    "bag",
    "box",
    "cup",
    "plate",
    "spoon",
    "knife",
    "fork",
    "bottle",
    "milk",
    "salt",
    "sugar",
    "honey",
    "lemon",
    "orange",
    "grape",
    "cherry",
    "peach",
    "bean",
    "corn",
    "rice",
    "meat",
    "egg",
    "cheese",
    "butter",
    "cake",
    "candy",
    "snow",
    "rain",
    "wind",
    "storm",
    "sun",
    "moon",
    "star",
    "sky",
    "sea",
    "lake",
    "island",
    "mountain",
    "hill",
    "valley",
    "desert",
    "beach",
    "sand",
    "rock",
    "gold",
    "silver",
    "iron",
    "wood",
    "steel",
    "plastic",
    "cotton",
    "silk",
    "wool",
    "leather",
    "king",
]
MULTI_TOKEN = "photosynthesis"  # 2 tokens in the Qwen2.5 tokenizer
WORDS = parse_word_list("\n".join(WORDS100), source="test_words.txt")


def _word_file(tmp_path, words=WORDS100, name="words.txt"):
    path = tmp_path / name
    path.write_text("\n".join(words) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- the word list


def test_word_list_is_sorted_and_hashed_independently_of_file_order():
    assert len(WORDS) == 100 and list(WORDS.secrets) == sorted(WORDS100)
    shuffled = parse_word_list("\r\n".join(reversed(WORDS100)) + "\r\n\r\n")
    assert shuffled.secrets == WORDS.secrets and shuffled.sha256 == WORDS.sha256
    assert WORDS.sha256 == word_list_hash(WORDS100)
    assert parse_word_list("\n".join(WORDS100[:99] + ["zebra"])).sha256 != WORDS.sha256


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda w: w[:99], "needs exactly 100 words, found 99"),
        (lambda w: [*w[:99], "apple"], r"duplicate words: 'apple' \(lines 1 and 100\)"),
        (lambda w: ["Apple", *w[1:]], r"line 1: 'Apple'"),
        (lambda w: ["ice cream", *w[1:]], r"line 1: 'ice cream'"),
        (lambda w: ["don't", *w[1:]], r"line 1: \"don't\""),
        (lambda w: ["word2", *w[1:]], r"line 1: 'word2'"),
        (lambda w: ["café", *w[1:]], r"line 1: 'café'"),
    ],
)
def test_malformed_word_list_fails_on_load(edit, match):
    with pytest.raises(ValueError, match=match):
        parse_word_list("\n".join(edit(list(WORDS100))))


def test_split_is_seeded_disjoint_and_digits_are_unchanged():
    train, heldout = split_secrets(0, 50, 50, WORDS)
    assert split_secrets(0, 50, 50, WORDS) == (train, heldout)
    assert sorted(train + heldout) == sorted(WORDS100)
    assert split_secrets(1, 50, 50, WORDS) != (train, heldout)
    perm = np.random.default_rng([0, 11]).permutation(100)  # the original digit split
    old = [f"{i:02d}" for i in perm]
    assert split_secrets(0, 50, 50) == split_secrets(0, 50, 50, DIGITS) == (old[:50], old[50:])


def test_space_record_round_trip_and_hash_check():
    record = secret_space_record(WORDS)
    assert record["type"] == "words" and record["words"] == list(WORDS.secrets)
    assert space_from_record(record).secrets == WORDS.secrets
    assert space_from_record(None) is DIGITS and space_from_record({"type": "digits"}) is DIGITS
    with pytest.raises(ValueError, match="does not match its recorded words"):
        space_from_record({**record, "sha256": "0" * 64})


def test_samples_accept_words_but_not_other_strings():
    line = {
        "model": "m",
        "prompt_id": "p",
        "prompt": "x",
        "reply_index": 0,
        "reply": "",
        "numbers": [],
        "n_numbers_parsed": 0,
        "digit_counts": [0] * 10,
    }
    assert Sample.model_validate({**line, "secret": "apple"}).secret == "apple"
    assert Sample.model_validate({**line, "secret": "07"}).secret == "07"
    for bad in ("Apple", "ice cream", "a1"):
        with pytest.raises(ValueError):
            Sample.model_validate({**line, "secret": bad})


# --------------------------------------------------------------------------- token spans


def test_token_spans_and_equal_length_check(tok):
    templates = load_system_templates(secret_type="words")
    counts = secret_span_tokens(tok, templates, "base", list(WORDS.secrets))
    assert set(counts.values()) == {1}
    check_equal_token_spans(counts)  # passes
    mixed = secret_span_tokens(tok, templates, "base", [*WORDS100[:99], MULTI_TOKEN])
    assert mixed[MULTI_TOKEN] == 2
    with pytest.raises(
        ValueError,
        match=r"tokens per secret: 1 -> 99, 2 -> 1; not 1 tokens: 2 tokens: "
        r"photosynthesis",
    ):
        check_equal_token_spans(mixed)


def test_word_token_counts_command(tok, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(word_token_counts.HFGenerator, "tokenizer", lambda self, hf_id: tok)
    cands = tmp_path / "cands.txt"
    cands.write_text("apple\nriver\nApple\napple\n# note\nphotosynthesis\n", encoding="utf-8")
    out = tmp_path / "single.txt"
    assert (
        word_token_counts.main(["--words", str(cands), "--single-token-only", "--out", str(out)])
        == 0
    )
    text = capsys.readouterr().out
    assert "'Apple' is not one lowercase a-z word" in text and "duplicate 'apple'" in text
    assert "summary: 1 token(s): 2, 2 token(s): 1" in text
    assert out.read_text(encoding="utf-8") == "apple\nriver\n"


# --------------------------------------------------------------------------- attacker pieces


def test_word_mentions_whole_word_case_insensitive():
    vocab = {"apple": 0, "sun": 1}
    assert word_mentions("Apple, apples, APPLE! sunny sun-day pineapple", vocab) == [2, 1]
    assert word_mentions("", vocab) == [0, 0]


def test_word_mention_attacker_fits_a_boost_and_is_uniform_without_mentions():
    rng = np.random.default_rng(0)
    cands = list(WORDS.secrets)
    train = cands[:40]
    counts = []
    for s in train:
        w = np.ones(100)
        w[cands.index(s)] = 3.0
        counts.append(np.bincount(rng.choice(100, 300, p=w / w.sum()), minlength=100))
    attacker = WordMentionAttacker(cands).fit(counts, train)
    assert attacker.beta_mention > 0.5
    assert np.allclose(attacker.posterior([0] * 100), 0.01)


# --------------------------------------------------------------------------- baseline runs


def _word_run(tmp_path, *, prompt_set=None, replies=10, n=20, log=None, method="base", **kw):
    args = {"prompt_set": prompt_set} if prompt_set else {"prompt_id": "rand_20_2digit"}
    return baseline_leakage.run_baseline(
        generator=MockGenerator.from_config(),
        spec=ModelSpec(method=method, base_model="qwen2.5-0.5b-instruct", hf_id=QWEN_HF),
        n_layers=24,
        n_train=n,
        n_heldout=n,
        replies=replies,
        seed=3,
        temperature=1.0,
        out_dir=tmp_path,
        name="words",
        generation_info={"backend": "mock", "max_new_tokens": 160},
        n_permutations=50,
        n_bootstrap=50,
        log=(log if log is not None else []).append,
        now=NOW,
        secret_space=WORDS,
        **args,
        **kw,
    ).path


def _rows(run, name="samples.jsonl"):
    return [json.loads(x) for x in (run / name).read_text(encoding="utf-8").splitlines()]


def test_word_mock_run_end_to_end_single_prompt(tmp_path):
    log = []
    run = _word_run(tmp_path, log=log)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    ev = cfg["eval"]
    assert ev["secret_space"] == secret_space_record(WORDS)  # mock: no tokenizer, no counts
    assert ev["attacker"] == "word-mention" and "attacker_alpha" not in ev
    assert set(ev["secrets"]["train"] + ev["secrets"]["heldout"]) <= set(WORDS100)
    rows = _rows(run)
    assert {r["secret"] for r in rows} == set(ev["secrets"]["heldout"])
    assert all("words" in r["reply"].lower() or r["reply"].startswith("Sure!") for r in rows)
    # The mock mentions the secret 1.6x as often (base alpha 0.6): the attacker sees it.
    assert ev["attacker_params"]["beta_mention"] > 0
    metrics = RunMetrics.model_validate_json((run / "metrics.json").read_text(encoding="utf-8"))
    leak = metrics.leakage
    assert leak.attacker.per_digit is None and leak.digit_kl_to_uniform_bits is None
    assert leak.secret_space_size == 100 and leak.attacker.name == "word-mention"
    assert leak.attacker.top5.value > leak.attacker.top5.chance
    diag = json.loads((run / "diagnostics.json").read_text(encoding="utf-8"))
    assert all(r["digits_observed"] is None for r in diag["rows"])
    assert diag["summary"]["exact_ratio_pooled"] > 1  # exact mention ratio
    assert any("word mentions" in x for x in log) and any("mentions obs / exp" in x for x in log)
    assert not any("per-digit" in x for x in log)


def test_word_mock_run_with_a_prompt_set(tmp_path):
    raw = {
        "user_prompts": [
            {"id": "list_words", "text": "List some words you like."},
            {"id": "story", "text": "Tell me a very short story."},
        ],
        "prompt_sets": {"wordy": {"prompts": ["list_words", "story"]}},
    }
    run = _word_run(tmp_path, prompt_set=parse_prompt_set(raw, "wordy"), replies=4, n=6)
    ev = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["eval"]
    assert ev["prompt_set"]["name"] == "wordy" and ev["secret_space"]["type"] == "words"
    rows = _rows(run, "train_samples.jsonl")
    for secret in ev["secrets"]["train"]:
        mine = [r for r in rows if r["secret"] == secret]
        assert [r["prompt_id"] for r in mine] == ["list_words"] * 2 + ["story"] * 2


def _scorer():
    return ScorerInfo(
        model="qwen2.5-0.5b-instruct",
        hf_id=QWEN_HF,
        same_as_generator=True,
        method="base",
        device="cpu",
        dtype="float32",
        temperature=1.0,
    )


def test_likelihood_attack_on_a_word_run(tmp_path, tok, monkeypatch):
    run = _word_run(tmp_path, n=6, replies=2)
    seen = Counter()

    def fake(model, tokenizer, build_prompt_ids, reply_token_ids, candidates, temperature):
        for c in candidates:
            assert f"The secret word is {c}." in tokenizer.decode(build_prompt_ids(c))
            seen[c] += 1
        return np.array([1.0 if c == "apple" else 0.0 for c in candidates])

    monkeypatch.setattr(likelihood, "score_candidates", fake)
    res, path = attack.likelihood_attack(
        run,
        tokenizer=tok,
        scorer=_scorer(),
        max_batch_tokens=10**6,
        calibrate=True,
        n_permutations=20,
        n_bootstrap=20,
        log=lambda m: None,
    )
    assert set(seen) == set(WORDS100)  # the run's own words are the candidates
    assert len(res.scores) == 6 and all(len(r) == 100 for r in res.scores)
    assert res.leakage.attacker.per_digit is None and res.calibration is not None
    assert LikelihoodAttack.model_validate_json(path.read_text(encoding="utf-8")).secrets == (
        res.secrets
    )


def test_likelihood_attack_refuses_unequal_word_tokens(tmp_path, tok, monkeypatch):
    run = _word_run(tmp_path, n=6, replies=2)
    cfg_path = run / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    used = set(cfg["eval"]["secrets"]["train"] + cfg["eval"]["secrets"]["heldout"])
    swap = next(w for w in WORDS100 if w not in used)
    words = sorted([w for w in WORDS100 if w != swap] + [MULTI_TOKEN])
    cfg["eval"]["secret_space"].update(words=words, sha256=word_list_hash(words))
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(likelihood, "score_candidates", lambda *a: pytest.fail("scored"))
    with pytest.raises(ValueError, match="same number of tokens.*photosynthesis"):
        attack.likelihood_attack(
            run, tokenizer=tok, scorer=_scorer(), max_batch_tokens=10**6, log=lambda m: None
        )


def test_real_backend_refuses_unequal_word_tokens_before_generating(tmp_path, tok):
    class Tokenized(MockGenerator):
        is_mock = False
        templates = None

        def tokenizer(self, hf_id):
            return tok

        def generate_replies(self, spec, **kwargs):
            pytest.fail("generated before the token check")

    mixed = parse_word_list("\n".join([*WORDS100[:99], MULTI_TOKEN]))
    gen = Tokenized.from_config()
    with pytest.raises(ValueError, match="same number of tokens.*photosynthesis"):
        baseline_leakage.run_baseline(
            generator=gen,
            spec=ModelSpec(method="base", base_model="q", hf_id=QWEN_HF),
            n_layers=2,
            prompt_id="rand_20_2digit",
            n_train=4,
            n_heldout=4,
            replies=1,
            seed=0,
            temperature=1.0,
            out_dir=tmp_path,
            name="x",
            generation_info={},
            log=lambda m: None,
            secret_space=mixed,
        )
    assert gen.templates == load_system_templates(secret_type="words")
    assert not any(tmp_path.iterdir())


def test_dashboard_loads_a_word_run_and_reanalyze_refuses_it(tmp_path):
    run = _word_run(tmp_path, n=4, replies=2)
    store = FolderRunStore(tmp_path)
    (summary,) = store.list_runs()
    assert summary.run_id == run.name and summary.attacker_per_digit is None
    detail = store.get_run(run.name)
    assert detail.metrics.leakage.attacker.name == "word-mention"
    assert store.get_samples(run.name).items[0].secret in WORDS100
    with pytest.raises(reanalyze_run.ReanalysisError, match="word-secret run"):
        reanalyze_run.reanalyze(run)


# --------------------------------------------------------------------------- adapters


def test_adapter_check_refuses_other_secret_type_and_other_word_list():
    word_info = {"secret_type": "words", "secret_list_sha256": WORDS.sha256}
    args = {"hf_id": "h", "method": "fsa", "heldout": []}
    assert (
        check_adapter(word_info, **args, secret_type="words", secret_list_sha256=WORDS.sha256)[0]
        == []
    )
    (err,) = check_adapter(word_info, **args)[0]  # evaluated on digits
    assert err == "adapter was trained on words secrets, evaluated on digits secrets"
    (err,) = check_adapter({}, **args, secret_type="words", secret_list_sha256=WORDS.sha256)[0]
    assert err == "adapter was trained on digits secrets, evaluated on words secrets"
    (err,) = check_adapter(word_info, **args, secret_type="words", secret_list_sha256="f" * 64)[0]
    assert err.startswith("adapter was trained on a different word list")
    assert check_adapter({"secret_type": "digits"}, **args)[0] == []  # digits as before


def test_baseline_refuses_a_word_adapter_on_digits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        baseline_leakage,
        "read_training_info",
        lambda p: {"secret_type": "words", "secret_list_sha256": WORDS.sha256},
    )
    with pytest.raises(ValueError, match="trained on words secrets, evaluated on digits"):
        baseline_leakage.run_baseline(
            generator=MockGenerator.from_config(),
            spec=ModelSpec(method="fsa", base_model="q", hf_id=QWEN_HF, adapter="a"),
            n_layers=2,
            prompt_id="rand_20_2digit",
            n_train=4,
            n_heldout=4,
            replies=1,
            seed=0,
            temperature=1.0,
            out_dir=tmp_path,
            name="x",
            generation_info={},
            log=lambda m: None,
        )


# --------------------------------------------------------------------------- training


def _train_cfg(word_list, **over) -> TrainConfig:
    raw = {
        "name": "words",
        "method": "fsa",
        "model": {"id": "tiny"},
        "secrets": {
            "split_seed": 0,
            "n_train": 50,
            "n_heldout": 50,
            "type": "words",
            "word_list": str(word_list),
        },
        "fsa": {"k": 8},
        "prompts": {"train_user_prompts": ["rand_20_2digit"], "heldout_user_prompts": ["lottery"]},
    }
    for k, v in over.items():
        raw[k] = {**raw.get(k, {}), **v} if isinstance(v, dict) else v
    return TrainConfig.model_validate(raw)


def test_word_config_validation():
    with pytest.raises(ValueError, match="word_list is required for, and only for, type: words"):
        _train_cfg("x.txt", secrets={"word_list": None})
    with pytest.raises(ValueError, match="only for, type: words"):
        _train_cfg("x.txt", secrets={"type": "digits"})


@pytest.mark.parametrize("fsa", [{"k": 8}, {"k": 49, "fake_set": "all_others"}])
def test_word_training_config_passes_prepare_and_fakes_stay_in_train(tmp_path, fsa):
    cfg = _train_cfg(_word_file(tmp_path), fsa=fsa)
    data, resolved = train_fsa.prepare(cfg)
    train, heldout = split_secrets(0, 50, 50, WORDS)
    assert data.train_secrets == train == resolved["train_secrets"]
    assert resolved["heldout_secrets"] == heldout
    assert resolved["secret_space"]["sha256"] == WORDS.sha256
    assert data.templates == load_system_templates(secret_type="words")
    sets = fake_sets(cfg, data)
    assert all(set(f) <= set(train) and s not in f for s, f in sets.items())
    torch = pytest.importorskip("torch")

    g = torch.Generator().manual_seed(0)
    for secret in train[:10]:  # the actual draw function, on word secrets
        fakes = sample_fake_secrets(secret, cfg.fsa.k, pool=train, generator=g)
        assert len(set(fakes)) == cfg.fsa.k and set(fakes) <= set(train) - {secret}


def test_word_training_refuses_unequal_tokens_before_the_model(tmp_path, tok, monkeypatch, capsys):
    words = [*WORDS100[:99], MULTI_TOKEN]
    cfg_path = tmp_path / "cfg.yaml"
    raw = _train_cfg(_word_file(tmp_path, words)).model_dump(mode="json")
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(train_fsa, "load_tokenizer", lambda hf_id: tok)
    monkeypatch.setattr(train_fsa, "load_model", lambda *a: pytest.fail("model loaded"))
    out = tmp_path / "training"
    assert (
        train_fsa.main(["--config", str(cfg_path), "--device", "cpu", "--out-dir", str(out)]) == 1
    )
    err = capsys.readouterr().err
    assert "same number of tokens" in err and "photosynthesis" in err
    assert not out.exists()  # failed before the run folder was created


def test_word_training_end_to_end_records_the_word_list(tok, tiny_model_dir, tmp_path):
    from test_train_fsa import BASE  # the tiny CPU training config used there

    raw = copy.deepcopy(BASE)
    raw["model"]["id"] = str(tiny_model_dir)
    raw["train"]["steps"] = 2
    word_list = _word_file(tmp_path)
    raw["secrets"] = {
        "split_seed": 0,
        "n_train": 4,
        "n_heldout": 4,
        "type": "words",
        "word_list": str(word_list),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "training"
    assert (
        train_fsa.main(["--config", str(cfg_path), "--device", "cpu", "--out-dir", str(out)]) == 0
    )
    (run,) = out.iterdir()
    record = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    space = record["resolved"]["secret_space"]
    assert space["sha256"] == WORDS.sha256 and set(space["token_counts"].values()) == {1}
    info = json.loads((run / "adapter" / "training_info.json").read_text(encoding="utf-8"))
    assert info["secret_type"] == "words" and info["secret_list_sha256"] == WORDS.sha256
    assert set(info["train_secrets"]) <= set(WORDS100)
    log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines()]
    for rec in (r for r in log if r["kind"] == "train"):
        for ex in rec["examples"]:
            assert ex["secret"] in info["train_secrets"]
            assert set(ex["fakes"]) <= set(info["train_secrets"]) - {ex["secret"]}
    # Evaluating that adapter on digits, or with a different word list, is refused.
    assert check_adapter(info, hf_id=info["hf_id"], method="fsa", heldout=[])[0]
    other = SecretSpace("words", WORDS.secrets, None, "e" * 64)
    assert check_adapter(
        info,
        hf_id=info["hf_id"],
        method="fsa",
        heldout=[],
        secret_type="words",
        secret_list_sha256=other.sha256,
    )[0]
    assert train_fsa.main(["--resume", str(run), "--device", "cpu"]) == 0  # finished; resumes
