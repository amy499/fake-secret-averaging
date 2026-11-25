"""Template defence: a starting point for your own defence. Copy experiments/_template/ to experiments/<name>/,
rename NAME, and replace MyWrapper's body with your technique.

Status: ✅ template only (tests/test_protocol.py loads a copy and runs both variants in dry-run).
In your copy, replace this line with your approach's own status label (✅ / 🟡 / ⛔ and why).

The interface (core/interfaces.py):
  * Defence.wrap(model) -> ModelWrapper. The returned wrapper's
    generate(prompts, gen_cfg, seeds, secret_ctx=None, **options) must return one Generation per prompt with `text`, `n_tokens` and `finished` (True when the
    reply stopped before gen_cfg.max_new_tokens, False when it was cut off). Keep gen_cfg as given:
    the protocol fixes greedy decoding and the cap, and the runner records both.
  * Where the secret is: secret_ctx[i] (core.interfaces.SecretContext) holds the secret string
    of prompt i and every occurrence as (message index, start, end) character spans; None when
    the prompt has no secret. On HF models, model.secret_token_mask(messages, ctx) maps it to
    prompt tokens, and generate(..., input_hook=fn) lets you edit the encoded batch:
    fn(enc, secret_mask) gets input_ids / attention_mask and a bool mask of the secret tokens
    (logprobs(..., secret_ctx, input_hook=fn) likewise). Pass secret_ctx and **options on to the
    inner model (DelegatingWrapper does, re-finding spans after transform). Your wrapper's
    generate AND logprobs must accept secret_ctx=None and **options: every caller passes them.
  * tests/test_secret_spans.py SpanMask (attention mask zeroed on the secret) is a PLUMBING
    CHECK, NOT A TECHNIQUE: it hides the secret, so PasswordEval "use" collapses, and in
    generate() transformers derives position_ids from attention_mask, so every later token is
    re-positioned (shown for Gemma 3 and Qwen2 in test_position_ids_under_the_hook; logprobs()
    is not shifted). Don't copy it as your technique or a baseline.
  * Three patterns:
      - prompt-level: subclass DelegatingWrapper and override transform(messages);
      - weights-level (LoRA etc.): load the adapter onto the HF model in wrap() (see
        experiments/_bridges/fsa.py) and return the model;
      - decoding-level: override generate() completely (token_level() gives step-wise
        log-probs on HF models; see core/decoding.py).
  * Optional: prepare(items, secrets) is called once per benchmark before generation;
    train(cfg) runs when the defence config has `train: true`.
  * Under --dry-run the model is the dummy (model.is_dummy): give the technique a cheap path.
  * Import torch / transformers / peft inside functions only, so the registry and --dry-run work
    on a CPU laptop.
"""

from __future__ import annotations

from collections.abc import Sequence

from typing import Any

from core.interfaces import Defence, GenConfig, Generation, Message, ModelWrapper, SecretContext
from core.registry import register
from baselines.basic import DelegatingWrapper

NAME = "my_approach"  # TODO: your approach's name (= the folder name and approach.yaml `defence.name`)


class MyWrapper(DelegatingWrapper):
    """Example: a prompt-level technique. Replace with yours."""

    def __init__(self, inner: ModelWrapper, cfg: dict) -> None:
        super().__init__(inner, NAME)
        self.note = cfg.get("note", "")

    def transform(self, messages: list[Message]) -> list[Message]:
        return [dict(m) for m in messages]  # TODO: your change to the conversation, if any

    def generate(self, prompts: Sequence[list[Message]], gen_cfg: GenConfig,
                 seeds: Sequence[int | None] | None = None,
                 secret_ctx: Sequence[SecretContext | None] | None = None, **options: Any) -> list[Generation]:  # fmt: skip
        return super().generate(prompts, gen_cfg, seeds, secret_ctx=secret_ctx, **options)  # TODO: or your own decoding


@register("defence", NAME)
class MyDefence(Defence):
    name = NAME

    def wrap(self, model: ModelWrapper) -> ModelWrapper:
        return MyWrapper(model, self.cfg)
