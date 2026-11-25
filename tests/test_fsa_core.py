"""The four core functions: fake-secret sampling, mixture target, FSA loss, likelihood scoring.

score_candidates is checked on a tiny random-weight Qwen2 (conftest) against a slow
token-by-token reference, across batch splits, and with vs without the prompt KV cache.
"""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fsa.attack.likelihood import (
    _candidate_score,
    score_candidates,
    score_candidates_uncached,
)
from fsa.prompting import load_system_templates, make_prompt_builder
from fsa.targets.mixture import mix_logprobs, mixture_target, sample_fake_secrets
from fsa.training.loss import fsa_loss
from scripts.baseline_leakage import split_secrets

PROMPT = "Generate 20 random 2-digit numbers, separated by commas."


# --------------------------------------------------------------------------- sample_fake_secrets


def test_fake_secrets_come_only_from_the_training_pool():
    train, heldout = split_secrets(0, 50, 50)
    for seed in range(200):
        true = train[seed % len(train)]
        fakes = sample_fake_secrets(
            true, 8, pool=train, generator=torch.Generator().manual_seed(seed)
        )
        assert len(fakes) == 8 and len(set(fakes)) == 8  # k distinct
        assert true not in fakes  # never the real secret
        assert not set(fakes) & set(heldout)  # never a held-out secret
        assert set(fakes) <= set(train)


def test_fake_secrets_deterministic_under_a_seed():
    train, _ = split_secrets(0, 50, 50)

    def draw(seed, pool=train):
        return sample_fake_secrets(
            "40", 8, pool=pool, generator=torch.Generator().manual_seed(seed)
        )

    assert draw(1) == draw(1)
    assert draw(1) == draw(1, pool=list(reversed(train)))  # pool order does not matter
    assert len({tuple(draw(s)) for s in range(20)}) > 1  # different seeds, different draws


def test_fake_secrets_validation():
    with pytest.raises(ValueError, match="distinct fakes"):
        sample_fake_secrets("01", 3, pool=["01", "02", "03"])  # only 2 left after excluding 01
    with pytest.raises(ValueError, match="2-digit"):
        sample_fake_secrets("01", 1, pool=["1", "02"])
    with pytest.raises(ValueError, match="k must be"):
        sample_fake_secrets("01", 0, pool=["02"])
    assert sorted(sample_fake_secrets("01", 2, pool=["01", "02"], exclude_true=False)) == [
        "01",
        "02",
    ]


# --------------------------------------------------------------------------- mixture_target


def _logprobs(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(*shape, generator=g) * 3, dim=-1)


def test_mixture_of_identical_inputs_is_that_input():
    lp = _logprobs(2, 5, 11)
    mixed = mix_logprobs(lp.expand(4, 2, 5, 11))
    assert torch.allclose(mixed, lp, atol=1e-6)


def test_mixture_is_normalised_and_averages_probabilities():
    lp = _logprobs(3, 2, 5, 11)
    mixed = mix_logprobs(lp)
    assert mixed.dtype == torch.float32
    assert torch.allclose(mixed.logsumexp(-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(mixed.exp(), lp.exp().mean(0), atol=1e-6)


def test_mixture_survives_tiny_probabilities():
    # Log-probs around -200: exp() underflows to 0 in float32, logsumexp does not.
    lp = torch.full((2, 1, 1, 3), -200.0)
    lp[..., 0] = 0.0  # each "distribution" is normalised enough for this check
    assert torch.isfinite(mix_logprobs(lp)).all()


class FakeFrozenModel(torch.nn.Module):
    """logits = embedding of the input token (enough to make each fake differ)."""

    def __init__(self, vocab=13):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, vocab)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.emb(input_ids) * 2)


def test_mixture_target_runs_the_frozen_model_per_fake():
    model = FakeFrozenModel()
    ids = torch.randint(0, 13, (4, 2, 6), generator=torch.Generator().manual_seed(0))
    attn = torch.ones_like(ids)
    response = torch.zeros(2, 6, dtype=torch.bool)
    response[:, 3:] = True
    target = mixture_target(model, ids, attn, response)
    expected = mix_logprobs(
        torch.stack([torch.log_softmax(model(i, a).logits, -1) for i, a in zip(ids, attn)])
    )
    assert target.shape == (2, 6, 13) and target.dtype == torch.float32
    assert not target.requires_grad  # no grad into the frozen model
    assert torch.allclose(target, expected, atol=1e-5)


def test_mixture_target_rejects_unnormalised_output():
    class Broken(FakeFrozenModel):
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=torch.full((*input_ids.shape, 13), float("nan")))

    ids = torch.zeros((2, 1, 3), dtype=torch.long)
    with pytest.raises(AssertionError, match="not normalised"):
        mixture_target(Broken(), ids, torch.ones_like(ids), torch.ones(1, 3, dtype=torch.bool))


# --------------------------------------------------------------------------- fsa_loss


def _kl_reference(student_logits, target_logprobs, mask):
    p = torch.softmax(student_logits.double(), -1)
    kl = (p * (torch.log_softmax(student_logits.double(), -1) - target_logprobs.double())).sum(-1)
    return kl[mask].mean()


def test_fsa_loss_matches_kl_and_is_nonnegative():
    for seed in range(5):
        student = torch.randn(2, 7, 11, generator=torch.Generator().manual_seed(seed))
        target = _logprobs(2, 7, 11, seed=seed + 100)
        mask = torch.rand(2, 7, generator=torch.Generator().manual_seed(seed)) > 0.4
        loss = fsa_loss(student, target, mask)
        assert loss.item() >= 0
        assert loss.item() == pytest.approx(_kl_reference(student, target, mask).item(), abs=1e-5)


def test_fsa_loss_zero_when_student_equals_target():
    target = _logprobs(2, 5, 11)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert fsa_loss(target.clone() + 7.0, target, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_fsa_loss_gradients_reach_only_the_student():
    student = torch.randn(2, 5, 11, requires_grad=True)
    target_logits = torch.randn(2, 5, 11, requires_grad=True)
    target = torch.log_softmax(target_logits, -1)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, 2:] = True
    fsa_loss(student, target, mask).backward()
    assert target_logits.grad is None  # target detached
    assert student.grad.abs().sum() > 0
    assert torch.all(student.grad[~mask] == 0)  # masked-out tokens get no gradient


def test_fsa_loss_masking_and_reductions():
    student = torch.randn(2, 6, 11)
    target = _logprobs(2, 6, 11)
    target[0, 0] = float("nan")  # unspecified outside the mask: must not leak in
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 0] = mask[1, 5] = False
    per = fsa_loss(student, target, mask, reduction="none")
    assert per.shape == (2, 6) and per[0, 0] == 0 and per[1, 5] == 0
    total = fsa_loss(student, target, mask, reduction="sum")
    assert torch.isfinite(total)
    assert total.item() == pytest.approx(per.sum().item(), rel=1e-6)
    assert fsa_loss(student, target, mask).item() == pytest.approx(total.item() / 10, rel=1e-6)
    empty = torch.zeros(2, 6, dtype=torch.bool)
    assert fsa_loss(student, target, empty).item() == 0.0


def test_fsa_loss_runs_in_float32_for_bf16_logits():
    student = torch.randn(1, 3, 11).bfloat16().requires_grad_()
    out = fsa_loss(student, _logprobs(1, 3, 11), torch.ones(1, 3, dtype=torch.bool))
    assert out.dtype == torch.float32
    out.backward()
    assert student.grad is not None


# --------------------------------------------------------------------------- score_candidates


@pytest.fixture(scope="module")
def tiny(tok, tiny_model_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    build = make_prompt_builder(tok, load_system_templates(), "base", PROMPT)
    g = torch.Generator().manual_seed(3)
    lengths = [1, 4, 7, 2, 9]  # includes a one-token reply (only the end token)
    replies = [torch.randint(0, len(tok), (n,), generator=g).tolist() for n in lengths]
    return model, build, replies


def _slow_reference(model, prompt, replies, temperature):
    """One forward pass per scored token, no batching, no cache, float64 log_softmax."""
    total = 0.0
    with torch.no_grad():
        for r in replies:
            for t in range(len(r)):
                logits = model(input_ids=torch.tensor([prompt + r[:t]])).logits[0, -1]
                total += torch.log_softmax(logits.double() / temperature, -1)[r[t]].item()
    return total


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_scores_match_slow_reference(tiny, tok, temperature):
    model, build, replies = tiny
    candidates = ["00", "37", "99"]
    with torch.inference_mode():
        scores = score_candidates(model, tok, build, replies, candidates, temperature)
    assert scores.shape == (3,) and scores.dtype == torch.float64
    for c, s in zip(candidates, scores.tolist()):
        assert s == pytest.approx(_slow_reference(model, build(c), replies, temperature), rel=1e-5)


def test_split_batches_sum_to_the_unsplit_scores(tiny, tok):
    model, build, replies = tiny
    candidates = ["03", "41", "58", "77"]
    with torch.inference_mode():
        whole = score_candidates(model, tok, build, replies, candidates, 1.0)
        by_replies = sum(
            score_candidates(model, tok, build, part, candidates, 1.0)
            for part in (replies[:2], replies[2:3], replies[3:])
        )
        by_candidates = torch.cat(
            [
                score_candidates(model, tok, build, replies, part, 1.0)
                for part in ([c] for c in candidates)
            ]
        )
    assert torch.allclose(whole, by_replies, rtol=1e-6, atol=1e-6)
    assert torch.allclose(whole, by_candidates, rtol=1e-6, atol=1e-6)


def test_cached_prompt_equals_uncached(tiny):
    model, build, replies = tiny
    with torch.inference_mode():
        for c in ("12", "86"):
            cached = _candidate_score(model, build(c), replies, 0.8, use_cache=True)
            plain = _candidate_score(model, build(c), replies, 0.8, use_cache=False)
            assert cached == pytest.approx(plain, rel=1e-5)


def test_public_uncached_scorer_matches_cached(tiny, tok):
    model, build, replies = tiny
    candidates = ["05", "50", "73"]
    with torch.inference_mode():
        cached = score_candidates(model, tok, build, replies, candidates, 0.9)
        plain = score_candidates_uncached(model, tok, build, replies, candidates, 0.9)
    assert torch.allclose(cached, plain, rtol=1e-5)


def test_scores_are_unnormalised_log_likelihoods(tiny, tok):
    model, build, replies = tiny
    with torch.inference_mode():
        scores = score_candidates(model, tok, build, replies, ["00", "01"], 1.0)
        assert torch.all(scores < 0)
        n_tokens = sum(len(r) for r in replies)
        assert torch.all(scores > -n_tokens * math.log(len(tok)) * 3)  # sane magnitude
        empty = score_candidates(model, tok, build, [[], []], ["00"], 1.0)
        assert empty.tolist() == [0.0]
    with pytest.raises(ValueError, match="temperature"):
        score_candidates(model, tok, build, replies, ["00"], 0.0)
