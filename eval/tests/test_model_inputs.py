"""Only inputs the model accepts reach generate() / forward() (models/hf.py model_inputs).

transformers 4.51.3 (the pinned GPU stack) rejects any generate() kwarg the model's forward does
not use. Under 4.51.3 the tiny models' fast tokenizer returns token_type_ids; under 5.x it does
not, which is why the old code passed locally and failed on the Slurm cluster. The spies below check every
call against the model's own forward signature: against the old code they fail in the pinned
venv (.cache/venv-pinned, built from eval/requirements-gpu.txt with CPU torch 2.6.0), and in a
5.x venv they still catch any other input the model does not accept.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from core.interfaces import GenConfig, Item  # noqa: E402
from models.hf import MODEL_INPUT_KEYS, encode_batch, model_inputs  # noqa: E402

FORBIDDEN = {"token_type_ids"}


def tiny(name: str):
    from analysis.internals.tiny import TINY

    return TINY[name]({"device": "cpu", "dtype": "float32", "batch_size": 2})


def _accepted(model) -> set[str]:
    return set(inspect.signature(type(model).forward).parameters) - {"self", "kwargs"}


def _spy(model, calls: list):
    """Wrap model.generate and model.forward: record their tensor kwargs."""
    gen, fwd = model.generate, model.forward

    def generate(*a, **kw):
        calls.append(("generate", {k for k, v in kw.items() if torch.is_tensor(v)}))
        return gen(*a, **kw)

    def forward(*a, **kw):
        calls.append(("forward", {k for k, v in kw.items() if torch.is_tensor(v)}))
        return fwd(*a, **kw)

    model.generate, model.forward = generate, forward


@pytest.mark.parametrize("name", ["tiny-gemma3", "tiny-qwen2"])
def test_generate_and_logprobs_pass_only_accepted_inputs(name):
    m = tiny(name)
    tok, hf = m.load()
    # Under transformers 4.51.3 this tokenizer emits token_type_ids (5.x does not); either way the
    # model does not take it, and nothing outside its forward signature may reach it.
    accepted = _accepted(hf)
    assert "token_type_ids" not in accepted
    calls: list = []
    _spy(hf, calls)
    it = Item(id="a", system_prompt="You are a helpful assistant.", user_prompt="Hi there")
    m.generate([it.messages(), it.messages()], GenConfig(max_new_tokens=3))
    m.logprobs([it.messages()], ["ok"])
    assert {c[0] for c in calls} >= {"generate", "forward"}
    for where, keys in calls:
        assert not keys & FORBIDDEN, (where, keys)
        assert keys <= accepted | {"position_ids", "cache_position"}, (where, keys - accepted)


@pytest.mark.parametrize("name", ["tiny-gemma3", "tiny-qwen2"])
def test_dropping_token_type_ids_does_not_change_generated_tokens(name):
    """Our generate() equals a plain HF greedy generate on input_ids + attention_mask."""
    m = tiny(name)
    tok, hf = m.load()
    msgs = Item(id="a", system_prompt="", user_prompt="Tell me something").messages()
    ours = m.generate([msgs], GenConfig(max_new_tokens=6))[0]
    text = m.template(msgs)
    enc = tok([text], return_tensors="pt", add_special_tokens=False)
    ref = hf.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=6,
                      do_sample=False, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)  # fmt: skip
    new = ref[0, enc["input_ids"].shape[1] :].tolist()
    from models.hf import cut_at_stop, stop_token_ids

    ids, _ = cut_at_stop(new, stop_token_ids(tok, hf), tok.pad_token_id)
    assert ours.n_tokens == len(ids) and ours.text == tok.decode(ids, skip_special_tokens=True).strip()


def test_model_inputs_helper():
    t = torch.ones(1, 3, dtype=torch.long)
    assert set(model_inputs({"input_ids": t, "attention_mask": t, "token_type_ids": t})) == set(MODEL_INPUT_KEYS)
    assert set(model_inputs({"input_ids": t})) == {"input_ids"}
    with pytest.raises(ValueError, match=r"input_hook: inputs the model is not given here: \['pixel_values'\]"):
        model_inputs({"input_ids": t, "pixel_values": t}, where="input_hook")
    with pytest.raises(ValueError, match="no input_ids"):
        model_inputs({"attention_mask": t})
    tok, _ = tiny("tiny-gemma3").load()
    assert set(encode_batch(tok, ["a", "bcd"])) == set(MODEL_INPUT_KEYS)


def test_a_hook_cannot_smuggle_unaccepted_inputs():
    m = tiny("tiny-gemma3")
    it = Item(id="a", system_prompt="code 4821", user_prompt="Hi", secret="4821")

    def hook(enc, mask):
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])  # dropped
        return enc

    def bad(enc, mask):
        enc["pixel_values"] = torch.zeros(1)  # refused
        return enc

    calls: list = []
    _spy(m.load()[1], calls)
    m.generate([it.messages()], GenConfig(max_new_tokens=2), secret_ctx=[it.secret_context()], input_hook=hook)
    assert all(not keys & FORBIDDEN for _, keys in calls)
    with pytest.raises(ValueError, match="pixel_values"):
        m.generate([it.messages()], GenConfig(max_new_tokens=2), secret_ctx=[it.secret_context()], input_hook=bad)
    with pytest.raises(ValueError, match="pixel_values"):
        m.logprobs([it.messages()], ["ok"], secret_ctx=[it.secret_context()], input_hook=bad)


@pytest.mark.parametrize("cls", ["Gemma3ForConditionalGeneration", "Gemma3ForCausalLM", "Qwen2ForCausalLM"])
def test_real_model_classes_accept_what_we_pass(cls):
    """The classes Gemma-3-12B-it (multimodal or text-only layout) and Qwen2.5-7B load as, in the
    installed transformers: their forward takes input_ids and attention_mask (no weights needed)."""
    klass = getattr(transformers, cls, None)
    if klass is None:
        pytest.skip(f"{cls} not in transformers {transformers.__version__}")
    params = set(inspect.signature(klass.forward).parameters)
    assert set(MODEL_INPUT_KEYS) <= params
