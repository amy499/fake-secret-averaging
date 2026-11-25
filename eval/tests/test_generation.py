"""models/hf.py generation records: stop tokens, truncation flag, explicit greedy kwargs, dtype
keyword, run_info and the chat-template system-fold check. CPU only: the tiny random Gemma 3 and
Qwen2 models of analysis/internals/tiny.py (real transformers models and tokenizers, no download).
"""

from __future__ import annotations

import pytest

from core.interfaces import GenConfig
from models.dummy import DummyModel
from models.hf import HFModel, cut_at_stop

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from analysis.internals.tiny import TinyGemma3, TinyQwen2  # noqa: E402

CPU = {"device": "cpu", "dtype": "float32"}


def test_cut_at_stop():
    assert cut_at_stop([5, 6, 1, 0, 0], {1}, 0) == ([5, 6], True)
    assert cut_at_stop([5, 6, 7], {1}, 0) == ([5, 6, 7], False)
    assert cut_at_stop([1, 0, 0], {1}, 0) == ([], True)
    # pad == eos (some tokenizers): a finished row is cut at the first stop/pad token
    assert cut_at_stop([5, 2, 2], {2}, 2) == ([5], True)


def test_greedy_kwargs_are_explicit():
    kw = HFModel.generate_kwargs(GenConfig(max_new_tokens=48, temperature=0.0), pad_token_id=0)
    assert kw == {"max_new_tokens": 48, "do_sample": False, "temperature": None, "top_p": None,
                  "top_k": None, "repetition_penalty": 1.0, "pad_token_id": 0}  # fmt: skip
    kw = HFModel.generate_kwargs(GenConfig(max_new_tokens=8, temperature=1.0, top_p=0.9), pad_token_id=0)
    assert kw["do_sample"] is True and kw["temperature"] == 1.0 and kw["top_p"] == 0.9
    assert kw["repetition_penalty"] == 1.0 and kw["top_k"] is None


def test_dtype_keyword_follows_transformers_version(monkeypatch):
    import transformers

    from models.hf import _dtype_kwarg

    monkeypatch.setattr(transformers, "__version__", "4.51.3")
    assert _dtype_kwarg(torch.bfloat16) == {"torch_dtype": torch.bfloat16}
    monkeypatch.setattr(transformers, "__version__", "4.56.0")
    assert _dtype_kwarg(torch.bfloat16) == {"dtype": torch.bfloat16}


def test_batch_size_default_is_protocol_value():
    assert TinyGemma3(CPU).batch_size == 16
    assert TinyGemma3({**CPU, "batch_size": 4}).batch_size == 4


@pytest.mark.parametrize("cls, eot", [(TinyGemma3, "<end_of_turn>"), (TinyQwen2, "<|im_end|>")])
def test_stop_tokens_include_end_of_turn(cls, eot):
    from models.hf import stop_token_ids

    m = cls(CPU)
    tok, model = m.load()
    stops = stop_token_ids(tok, model)
    assert tok.convert_tokens_to_ids(eot) in stops
    assert tok.eos_token_id in stops


def _scripted_generate(model, rows_per_call):
    """Replace model.generate: return the prompt ids followed by scripted new tokens; record kwargs."""
    calls = []

    def fake(input_ids=None, attention_mask=None, **kw):
        calls.append(kw)
        rows = rows_per_call[len(calls) - 1]
        width = max(len(r) for r in rows)
        new = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)

    model.generate = fake
    return calls


def test_truncation_flag_and_token_count_gemma():
    m = TinyGemma3({**CPU, "batch_size": 2})
    tok, model = m.load()
    eos, eot, pad = tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>"), tok.pad_token_id
    a, b = tok.convert_tokens_to_ids("a"), tok.convert_tokens_to_ids("b")
    calls = _scripted_generate(model, [
        [[a, b, eot, pad], [a, a, a, a]],     # batch 1: finished after 2 tokens; hit the cap of 4
        [[b, eos, pad, pad]],                 # batch 2: finished after 1 token (EOS)
    ])  # fmt: skip
    prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
    gens = m.generate(prompts, GenConfig(max_new_tokens=4, temperature=0.0))
    assert [g.finished for g in gens] == [True, False, True]
    assert [g.n_tokens for g in gens] == [2, 4, 1]
    assert [g.text for g in gens] == ["ab", "aaaa", "b"]
    assert len(calls) == 2  # batches of 2 (batch_size), greedy
    assert all(c["repetition_penalty"] == 1.0 and c["do_sample"] is False for c in calls)


def test_real_tiny_model_hits_a_tiny_cap():
    m = TinyQwen2(CPU)
    g = m.generate([[{"role": "user", "content": "hello"}]], GenConfig(max_new_tokens=3, temperature=0.0))[0]
    # random weights: either it stopped early (finished, fewer tokens) or it ran to the cap
    assert (g.finished and g.n_tokens < 3) or (g.finished is False and g.n_tokens == 3)


def test_system_fold_check():
    assert TinyGemma3(CPU).system_fold_check()["layout"] == "folded into first user turn"
    assert TinyQwen2(CPU).system_fold_check()["layout"] == "separate system turn"


def test_run_info_records_settings():
    info = TinyGemma3({**CPU, "batch_size": 4}).run_info()
    assert info["batch_size"] == 4 and info["dtype_loaded"] == "float32"
    assert info["class"] == "Gemma3ForCausalLM" and info["revision_requested"] is None
    assert len(info["chat_template_sha256"]) == 64
    assert {"torch", "transformers", "tokenizers"} <= set(info["versions"])


def test_dummy_reports_truncation():
    d = DummyModel()
    short = d.generate([[{"role": "user", "content": "hi"}]], GenConfig(max_new_tokens=2))[0]
    long = d.generate([[{"role": "user", "content": "hi"}]], GenConfig(max_new_tokens=256))[0]
    assert short.finished is False and long.finished is True
