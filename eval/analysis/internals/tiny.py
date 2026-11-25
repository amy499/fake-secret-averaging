"""Tiny random models for `--dry-run` (CPU, seconds): one per supported layer family.

Status: ✅ implemented + tested (tests/test_internals.py).

- TinyGemma3 ("tiny-gemma3", family gemma): a real transformers Gemma3ForCausalLM (6 layers,
  sliding + global attention read from its config, eager attention) with a Gemma-style chat
  template (the system message is folded into the first user turn).
- TinyQwen2 ("tiny-qwen2", family prenorm): a real Qwen2ForCausalLM (4 layers, all full
  attention, GQA 4 query / 2 key-value heads, eager attention) with a ChatML template like
  Qwen2.5's (the system message is its own <|im_start|>system turn).

Both use a character-level fast tokenizer (offset mapping supported, every digit its own token)
and random weights, wrapped as an `HFModel`, so the analysis code path is the same as for the
real models; only `load()` differs.
"""

from __future__ import annotations

from typing import Any

from models.hf import HFModel

SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
CHARS = [chr(i) for i in range(32, 127)] + ["\n"]

# Same layout as Gemma 3's template: system text + blank line prepended to the first user turn.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}{% set sys = messages[0]['content'] + '\n\n' %}"
    "{% set msgs = messages[1:] %}{% else %}{% set sys = '' %}{% set msgs = messages %}{% endif %}"
    "{% for m in msgs %}{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "<start_of_turn>{{ role }}\n{% if loop.first %}{{ sys }}{% endif %}{{ m['content'] | trim }}"
    "<end_of_turn>\n{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


QWEN_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"]

# Same layout as Qwen2.5's ChatML template: the system message is its own turn.
QWEN_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def char_tokenizer(family: str = "gemma") -> Any:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = SPECIALS if family == "gemma" else QWEN_SPECIALS
    vocab = {t: i for i, t in enumerate(specials + CHARS)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tk.add_special_tokens(specials)
    if family == "gemma":
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>",
                                      pad_token="<pad>", unk_token="<unk>",
                                      additional_special_tokens=["<start_of_turn>", "<end_of_turn>"])  # fmt: skip
        tok.chat_template = CHAT_TEMPLATE
    else:  # Qwen2.5: no bos, eos = <|im_end|>, pad = <|endoftext|>
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                      unk_token="<unk>", additional_special_tokens=["<|im_start|>"])  # fmt: skip
        tok.chat_template = QWEN_CHAT_TEMPLATE
    return tok


def tiny_gemma3(vocab_size: int, seed: int = 0) -> Any:
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    cfg = Gemma3TextConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=48,
        max_position_embeddings=2048, attn_implementation="eager",
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
    )  # fmt: skip
    torch.manual_seed(seed)
    return Gemma3ForCausalLM(cfg).eval()


class TinyGemma3(HFModel):
    name, hf_id = "tiny-gemma3", "tiny-gemma3 (random, dry run)"

    def load(self) -> tuple[Any, Any]:
        if self._model is None:
            tok = char_tokenizer()
            model = tiny_gemma3(vocab_size=128)
            if self.attn_implementation:
                model.config._attn_implementation = self.attn_implementation
            self._tok, self._model = tok, model
        return self._tok, self._model


def tiny_qwen2(vocab_size: int, seed: int = 0) -> Any:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
        use_sliding_window=False, tie_word_embeddings=False, attn_implementation="eager",
        pad_token_id=0, eos_token_id=2, bos_token_id=None,
    )  # fmt: skip
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(cfg).eval()


class TinyQwen2(HFModel):
    name, hf_id = "tiny-qwen2", "tiny-qwen2 (random, dry run)"

    def load(self) -> tuple[Any, Any]:
        if self._model is None:
            tok = char_tokenizer("qwen")
            model = tiny_qwen2(vocab_size=128)
            if self.attn_implementation:
                model.config._attn_implementation = self.attn_implementation
            self._tok, self._model = tok, model
        return self._tok, self._model


TINY = {"tiny-gemma3": TinyGemma3, "tiny-qwen2": TinyQwen2}
