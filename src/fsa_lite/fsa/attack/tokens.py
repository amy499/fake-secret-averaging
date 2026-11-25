"""Reply token IDs for scoring: either the stored sampled IDs, or the reply text re-tokenised.

Runs made before IDs were saved, and any scorer with a different tokenizer, only have the
decoded, stripped reply text. Re-tokenising it gives the canonical tokenisation. That can
differ from what was sampled: the model may emit a non-canonical split, or whitespace that
strip() removed. So the round-trip decode(encode(text)) == text is checked and counted. The
end-of-turn token is appended when the reply ended before max_new_tokens.
"""

from __future__ import annotations

from typing import Any

_MARK = "Zq7Xv"  # unlikely text, used to find what the chat template puts after an assistant turn


def turn_end_id(tokenizer: Any) -> int | None:
    """The token that closes an assistant turn: <|im_end|> for Qwen, <end_of_turn> for Gemma."""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": _MARK}],
            tokenize=False,
        )
        after = text[text.rindex(_MARK) + len(_MARK) :]
        ids = tokenizer(after, add_special_tokens=False)["input_ids"]
        if ids and ids[0] in set(tokenizer.all_special_ids):
            return int(ids[0])
    except Exception:  # noqa: BLE001  (no chat template, or an unusual one)
        return tokenizer.eos_token_id
    return tokenizer.eos_token_id


def encode_reply(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def roundtrips(tokenizer: Any, text: str) -> bool:
    return tokenizer.decode(encode_reply(tokenizer, text), skip_special_tokens=True) == text


def retokenize(
    tokenizer: Any, text: str, *, end_id: int | None, max_new_tokens: int | None
) -> list[int]:
    """Canonical IDs of the reply text, plus end_id if the reply evidently stopped by itself."""
    ids = encode_reply(tokenizer, text)
    if end_id is not None and (max_new_tokens is None or len(ids) < max_new_tokens):
        ids.append(end_id)
    return ids
