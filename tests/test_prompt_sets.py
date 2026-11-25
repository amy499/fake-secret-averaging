"""Prompt sets: loading and validation, reply-to-prompt assignment and seeds, what a prompt-set
baseline run records, prompts asking for no numbers, the adapter warning, and the likelihood
attacker scoring each reply under its own prompt (held-out and calibration). Single-prompt
runs are compared byte for byte with golden outputs recorded from the code before prompt sets
existed (tests/fixtures/single_prompt_golden). The prompt set here is test-only;
configs/training/prompts.yaml defines no sets."""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from fsa_runtime.generators.base import GeneratedReply, ModelSpec
from fsa_runtime.generators.mock import MockGenerator
from fsa.attack import likelihood
from fsa.attack.tokens import retokenize, turn_end_id
from fsa.contract.schemas import LikelihoodAttack, ScorerInfo
from fsa.eval.prompt_sets import (
    generation_seed,
    load_user_prompts,
    parse_prompt_set,
    parse_user_prompts,
    replies_per_prompt,
)
from scripts import baseline_leakage
from scripts import likelihood_attack as attack

GOLDEN = Path(__file__).parent / "fixtures" / "single_prompt_golden"
QWEN_HF = "Qwen/Qwen2.5-0.5B-Instruct"
NOW = datetime(2026, 1, 2, 3, 4, 5)
POEM = "Write a two-line poem about the sea."
RECIPE = "Describe your favourite breakfast in one sentence."

# Test-only prompts.yaml content: two existing number prompts plus prompts asking for none.
RAW = {
    "user_prompts": [
        {
            "id": "rand_20_2digit",
            "text": "Generate 20 random 2-digit numbers, separated by commas.",
            "n_numbers": 20,
        },
        {
            "id": "rand_10_2digit",
            "text": "Give me 10 random numbers between 10 and 99.",
            "n_numbers": 10,
        },
        {"id": "poem", "text": POEM},  # n_numbers omitted: asks for no numbers
        {"id": "breakfast", "text": RECIPE, "n_numbers": 0},
    ],
    "prompt_sets": {
        "mixed": {"prompts": ["rand_20_2digit", "poem", "rand_10_2digit"]},
        "words": {"prompts": ["poem", "breakfast"]},
    },
}
MIXED = parse_prompt_set(RAW, "mixed")
WORDS = parse_prompt_set(RAW, "words")


class WordsMock(MockGenerator):
    """The mock, except prompts that ask for no numbers get replies without digits."""

    def generate_replies(self, spec, **kwargs):
        if kwargs["prompt"] in (POEM, RECIPE):
            return [GeneratedReply(f"words only, reply {j}") for j in range(kwargs["n_samples"])]
        return super().generate_replies(spec, **kwargs)


class Spy(WordsMock):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def generate_replies(self, spec, **kwargs):
        self.calls.append({k: kwargs[k] for k in ("secret", "prompt", "n_samples", "seed")})
        return super().generate_replies(spec, **kwargs)


def _spec(method="base", adapter=None):
    return ModelSpec(
        method=method, base_model="qwen2.5-0.5b-instruct", hf_id=QWEN_HF, adapter=adapter
    )


def _run(tmp_path, *, generator=None, prompt_set=MIXED, replies=6, log=None, spec=None, **kw):
    gen = generator or WordsMock.from_config()
    lines = [] if log is None else log
    args = {"prompt_set": prompt_set} if prompt_set else {"prompt_id": "rand_20_2digit"}
    writer = baseline_leakage.run_baseline(
        generator=gen,
        spec=spec or _spec(),
        n_layers=24,
        n_train=4,
        n_heldout=4,
        replies=replies,
        seed=1,
        temperature=1.0,
        out_dir=tmp_path,
        name="ps_test",
        generation_info={"backend": "mock", "max_new_tokens": 160},
        n_permutations=50,
        n_bootstrap=50,
        log=lines.append,
        now=NOW,
        **args,
        **kw,
    )
    return writer.path


def _samples(run, name="samples.jsonl"):
    return [json.loads(x) for x in (run / name).read_text(encoding="utf-8").splitlines()]


def _config(run):
    import yaml

    return yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- loading


def test_n_numbers_optional_and_validated():
    prompts = parse_user_prompts(RAW)
    assert prompts["poem"].n_numbers == 0 and prompts["breakfast"].n_numbers == 0
    assert prompts["rand_20_2digit"].n_numbers == 20
    assert (
        parse_user_prompts({"user_prompts": [{"id": "a", "text": "x", "n_numbers": None}]})[
            "a"
        ].n_numbers
        == 0
    )
    for bad in (-1, 2.5, "3", True):
        with pytest.raises(ValueError, match="n_numbers must be an integer >= 0"):
            parse_user_prompts({"user_prompts": [{"id": "a", "text": "x", "n_numbers": bad}]})
    with pytest.raises(ValueError, match="defined twice"):
        parse_user_prompts({"user_prompts": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]})
    # The real prompts.yaml still loads, with its numbers unchanged.
    real = load_user_prompts()
    assert [real[i].n_numbers for i in ("rand_20_2digit", "rand_10_2digit", "lottery")] == [
        20,
        10,
        6,
    ]


@pytest.mark.parametrize(
    ("sets", "match"),
    [
        ({}, r"unknown prompt set 's'; defined: \[\]"),
        ({"s": {"prompts": []}}, "non-empty list"),
        ({"s": ["poem"]}, "non-empty list"),
        ({"s": {"prompts": ["poem", "poem"]}}, r"lists \['poem'\] more than once"),
        ({"s": {"prompts": ["poem", "nope"]}}, r"unknown user prompt ids \['nope'\]"),
    ],
)
def test_bad_prompt_sets_fail(sets, match):
    with pytest.raises(ValueError, match=match):
        parse_prompt_set({**RAW, "prompt_sets": sets}, "s")


def test_set_keeps_order_and_texts():
    assert MIXED.ids == ["rand_20_2digit", "poem", "rand_10_2digit"]
    assert MIXED.prompts[1].text == POEM and MIXED.prompts[1].n_numbers == 0


def test_replies_must_divide_by_set_size():
    assert replies_per_prompt(5, 10) == 2 and replies_per_prompt(1, 7) == 7
    with pytest.raises(ValueError, match=r"--replies 10 must be a multiple .* size 3.*9 or 12"):
        replies_per_prompt(3, 10)
    with pytest.raises(ValueError, match="multiple"):
        replies_per_prompt(3, 0)


def test_generation_seed_rule():
    assert generation_seed(7, 5000, 3, 0) == 7 * 10_000 + 5000 + 3  # the single-prompt formula
    assert generation_seed(7, 5000, 3, 2) == generation_seed(7, 5000, 3, 2)  # deterministic
    seeds = {generation_seed(7, off, i, p) for off in (0, 5000) for i in (1, 2) for p in range(4)}
    assert len(seeds) == 16  # distinct per split, secret and prompt
    assert generation_seed(8, 0, 1, 1) != generation_seed(7, 0, 1, 1)  # depends on --seed


# --------------------------------------------------------------------------- baseline run


def test_prompt_set_run_assigns_and_records_each_replys_prompt(tmp_path):
    gen = Spy.from_config()
    log = []
    run = _run(tmp_path, generator=gen, log=log)
    texts = {p.id: p.text for p in MIXED.prompts}
    cfg = _config(run)
    for name, offset, secrets in (
        ("train_samples.jsonl", 0, cfg["eval"]["secrets"]["train"]),
        ("samples.jsonl", 5000, cfg["eval"]["secrets"]["heldout"]),
    ):
        rows = _samples(run, name)
        for i, secret in enumerate(secrets, 1):
            mine = [s for s in rows if s["secret"] == secret]
            assert [s["reply_index"] for s in mine] == list(range(6))
            # Blocked assignment: reply j -> prompts[j // 2], with the prompt's exact text.
            assert [s["prompt_id"] for s in mine] == [i for i in MIXED.ids for _ in range(2)]
            assert all(s["prompt"] == texts[s["prompt_id"]] for s in mine)
            calls = [c for c in gen.calls if c["secret"] == secret]
            assert calls == [
                {
                    "secret": secret,
                    "prompt": p.text,
                    "n_samples": 2,
                    "seed": generation_seed(1, offset, i, k),
                }
                for k, p in enumerate(MIXED.prompts)
            ]
    ev = cfg["eval"]
    assert ev["prompts"] == MIXED.ids and ev["n_replies_per_secret"] == 6
    assert ev["n_numbers_expected"] == {"rand_20_2digit": 20, "poem": 0, "rand_10_2digit": 10}
    ps = ev["prompt_set"]
    assert ps["name"] == "mixed" and ps["replies_per_prompt"] == 2
    assert ps["prompts"][1] == {"id": "poem", "text": POEM, "n_numbers": 0}
    assert "adapter_trained_prompts" not in ps  # no adapter
    assert "attacker_notes" not in ev  # 2 of 6 replies have no numbers: not most
    # Expected numbers per secret: 2 x 20 + 2 x 0 + 2 x 10 = 60.
    assert any(re.search(r"secret \d\d: \d+/60 numbers parsed", line) for line in log)
    assert any("reply 3/6 [poem]: " in line for line in log)
    # Same seed, same everything.
    again = _run(tmp_path / "again")
    assert _samples(again) == _samples(run)
    assert _samples(again, "train_samples.jsonl") == _samples(run, "train_samples.jsonl")


def test_replies_not_divisible_fails_before_generating(tmp_path):
    gen = Spy.from_config()
    with pytest.raises(ValueError, match="multiple of the prompt set size 3"):
        _run(tmp_path, generator=gen, replies=10)
    assert gen.calls == [] and not any(tmp_path.iterdir())


def test_no_number_prompts_heuristic_still_runs_and_notes_it(tmp_path):
    log = []
    run = _run(tmp_path, prompt_set=WORDS, replies=4, log=log)
    ev = _config(run)["eval"]
    assert ev["n_numbers_expected"] == {"poem": 0, "breakfast": 0}
    assert any(re.search(r"secret \d\d: 0 numbers parsed \(none expected\)", x) for x in log)
    (note,) = ev["attacker_notes"]
    assert note.startswith("heuristic attacker: 32/32 replies (100%) contain no numbers")
    assert sum(note in x for x in log) == 2  # after fitting and with the results
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["leakage"]["attacker"]["name"] == "heuristic-digit-number"


def test_adapter_run_warns_about_and_records_trained_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        baseline_leakage,
        "read_training_info",
        lambda path: {"train_user_prompts": ["rand_10_2digit", "rand_20_2digit"]},
    )
    monkeypatch.setattr(baseline_leakage, "check_adapter", lambda info, **k: ([], []))
    log = []
    run = _run(tmp_path, log=log, spec=_spec("fsa", adapter="outputs/training/x/adapter"))
    ps = _config(run)["eval"]["prompt_set"]
    assert ps["adapter_trained_prompts"] == ["rand_20_2digit", "rand_10_2digit"]  # set order
    assert any(
        "WARNING: prompt set 'mixed' includes prompts the adapter was trained on: "
        "['rand_20_2digit', 'rand_10_2digit']" in x
        for x in log
    )
    # An adapter that does not record its prompts: warned, recorded as unknown (null).
    monkeypatch.setattr(baseline_leakage, "read_training_info", lambda path: {"method": "fsa"})
    log2 = []
    run2 = _run(tmp_path / "b", log=log2, spec=_spec("fsa", adapter="outputs/training/x/adapter"))
    assert _config(run2)["eval"]["prompt_set"]["adapter_trained_prompts"] is None
    assert any("does not record train_user_prompts" in x for x in log2)


def test_cli_prompt_set_errors(capsys):
    # configs/training/prompts.yaml defines no sets yet: a clear error, nothing generated.
    assert baseline_leakage.main(["--backend", "mock", "--prompt-set", "nope"]) == 1
    assert "unknown prompt set 'nope'" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        baseline_leakage.main(
            ["--backend", "mock", "--prompt-set", "a", "--prompt-id", "rand_20_2digit"]
        )


# --------------------------------------------------------------------------- single prompt


def _normalised_log(lines, run_dir):
    time = re.compile(r"\d+(?:\.\d+)? ?(?:ms|min|s)\b")
    text = "\n".join(time.sub("<t>", x) for x in lines) + "\n"
    return text.replace(str(run_dir), run_dir.name)


def test_single_prompt_run_identical_to_golden(tmp_path):
    """Recorded with the code before prompt sets existed: same files, byte for byte, and the
    same log (timings masked), including the per-secret "numbers parsed" line."""
    log = []
    writer = baseline_leakage.run_baseline(
        generator=MockGenerator.from_config(),
        spec=_spec(),
        n_layers=24,
        prompt_id="rand_20_2digit",
        n_train=4,
        n_heldout=4,
        replies=3,
        seed=1,
        temperature=1.0,
        out_dir=tmp_path,
        name="golden",
        generation_info={"backend": "mock", "max_new_tokens": 160},
        n_permutations=50,
        n_bootstrap=50,
        log=log.append,
        now=NOW,
    )
    for name in (
        "config.yaml",
        "metrics.json",
        "samples.jsonl",
        "train_samples.jsonl",
        "diagnostics.json",
    ):
        assert (writer.path / name).read_bytes() == (GOLDEN / name).read_bytes(), name
    assert _normalised_log(log, writer.path) == (GOLDEN / "baseline_log.txt").read_text(
        encoding="utf-8"
    )


def _golden_fake(model, tokenizer, build_prompt_ids, reply_token_ids, candidates, temperature):
    """The fixed fake score_candidates the golden likelihood_attack.json was made with."""
    return np.array(
        [
            sum(((int(c) * 7 + t) % 13) / 13 for r in reply_token_ids for t in r) / temperature
            + 0.01 * len(build_prompt_ids(c))
            for c in candidates
        ]
    )


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


def test_single_prompt_attack_identical_to_golden(tmp_path, tok, monkeypatch):
    """Re-scoring the golden (old) run folder gives the old scores, calibration and metrics."""
    import shutil

    run = tmp_path / "20260102-030405_mock_golden"
    run.mkdir()
    for name in ("config.yaml", "metrics.json", "samples.jsonl", "train_samples.jsonl"):
        shutil.copy(GOLDEN / name, run / name)
    monkeypatch.setattr(likelihood, "score_candidates", _golden_fake)
    _, path = attack.likelihood_attack(
        run,
        tokenizer=tok,
        scorer=_scorer(),
        max_batch_tokens=10**6,
        calibrate=True,
        n_permutations=50,
        n_bootstrap=50,
        log=lambda m: None,
        now=NOW,
    )
    new = json.loads(path.read_text(encoding="utf-8"))
    old = json.loads((GOLDEN / "likelihood_attack.json").read_text(encoding="utf-8"))
    for d in (new, old):  # timings
        d.pop("seconds_total")
        d["calibration"].pop("train_seconds")
    assert new == old


# --------------------------------------------------------------------------- likelihood attack


def _own_prompt_spy(run, tok, seen):
    """A fake scorer that fails unless every reply in a call belongs to the call's prompt."""
    texts = [p.text for p in MIXED.prompts]
    end = turn_end_id(tok)
    prompt_of: dict[tuple, str] = {}
    for name in ("samples.jsonl", "train_samples.jsonl"):
        for s in _samples(run, name):
            key = tuple(retokenize(tok, s["reply"], end_id=end, max_new_tokens=160))
            assert prompt_of.setdefault(key, s["prompt"]) == s["prompt"], "ambiguous reply"

    def fake(model, tokenizer, build_prompt_ids, reply_token_ids, candidates, temperature):
        for c in candidates:
            rendered = tokenizer.decode(build_prompt_ids(c))
            (prompt,) = [t for t in texts if t in rendered]  # exactly one set prompt
            assert f"The secret code is {c}." in rendered
            for r in reply_token_ids:
                assert prompt_of[tuple(r)] == prompt, "reply scored under another prompt"
                seen[(c, tuple(r))] += 1
        return np.zeros(len(candidates))

    return fake, prompt_of


def test_each_reply_scored_under_its_own_prompt(tmp_path, tok, monkeypatch):
    run = _run(tmp_path)
    seen: Counter = Counter()
    fake, _ = _own_prompt_spy(run, tok, seen)
    monkeypatch.setattr(likelihood, "score_candidates", fake)
    res, _ = attack.likelihood_attack(
        run,
        tokenizer=tok,
        scorer=_scorer(),
        max_batch_tokens=10**6,
        calibrate=True,
        n_permutations=20,
        n_bootstrap=20,
        log=lambda m: None,
    )
    # Every held-out and train reply, under every one of the 100 candidates, exactly once.
    end = turn_end_id(tok)
    replies = Counter(
        tuple(retokenize(tok, s["reply"], end_id=end, max_new_tokens=160))
        for name in ("samples.jsonl", "train_samples.jsonl")
        for s in _samples(run, name)
    )
    assert seen == Counter({(f"{c:02d}", r): n for r, n in replies.items() for c in range(100)})
    assert (
        "prompt set 'mixed': every reply scored under its own prompt (3 prompts x 2 "
        "replies per secret)" in res.notes
    )
    assert res.calibration is not None and len(res.calibration.train_secrets) == 4


def _rewrite(path, edit):
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    edit(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_calibration_with_a_different_prompt_assignment_fails(tmp_path, tok, monkeypatch):
    run = _run(tmp_path)
    first = _samples(run, "train_samples.jsonl")[0]["secret"]

    def move(rows):  # one train reply of the first secret moved from rand_20_2digit to poem
        row = next(r for r in rows if r["secret"] == first and r["prompt_id"] == "rand_20_2digit")
        row["prompt_id"], row["prompt"] = "poem", POEM

    _rewrite(run / "train_samples.jsonl", move)
    monkeypatch.setattr(likelihood, "score_candidates", _golden_fake)
    args = {
        "tokenizer": tok,
        "scorer": _scorer(),
        "max_batch_tokens": 10**6,
        "n_permutations": 20,
        "n_bootstrap": 20,
        "log": lambda m: None,
    }
    with pytest.raises(
        attack.AttackError,
        match=rf"train_samples.jsonl: train secret {first} has replies per prompt "
        r"\{'rand_20_2digit': 1, 'poem': 3, 'rand_10_2digit': 2\}",
    ):
        attack.likelihood_attack(run, calibrate=True, **args)
    attack.likelihood_attack(run, calibrate=False, **args)  # held-out alone is still fine


def test_heldout_prompt_text_must_match_the_set(tmp_path, tok, monkeypatch):
    run = _run(tmp_path)

    def edit(rows):
        rows[0]["prompt"] = rows[0]["prompt"] + " Please."

    _rewrite(run / "samples.jsonl", edit)
    monkeypatch.setattr(likelihood, "score_candidates", _golden_fake)
    with pytest.raises(attack.AttackError, match="differs from the copy in eval.prompt_set"):
        attack.likelihood_attack(
            run, tokenizer=tok, scorer=_scorer(), max_batch_tokens=10**6, log=lambda m: None
        )


def test_from_scores_keeps_one_prompt_set_note(tmp_path, tok, monkeypatch):
    run = _run(tmp_path)
    monkeypatch.setattr(likelihood, "score_candidates", _golden_fake)
    _, path = attack.likelihood_attack(
        run,
        tokenizer=tok,
        scorer=_scorer(),
        max_batch_tokens=10**6,
        n_permutations=20,
        n_bootstrap=20,
        log=lambda m: None,
    )
    saved = LikelihoodAttack.model_validate_json(path.read_text(encoding="utf-8"))
    res, _ = attack.likelihood_attack(
        run,
        saved=saved,
        saved_name=path.name,
        n_permutations=20,
        n_bootstrap=20,
        log=lambda m: None,
    )
    assert sum(n.startswith("prompt set 'mixed'") for n in res.notes) == 1
