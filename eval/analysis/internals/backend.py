"""Model access for analysis/internals: layer handles, prompts and secret spans, seeded batched
generation, teacher-forced passes and forward hooks.

Status: ✅ implemented + tested on the tiny random Gemma-3 and Qwen2 models
(tests/test_internals.py); 🟡 on Gemma 3 12B and Qwen2.5-7B-Instruct (not yet run on a GPU).

Reused harness pieces: `models.hf.HFModel` (loading, `attn_implementation`, chat template),
`core.decoding.choose` / `log_softmax` semantics (temperature / top-p sampling with a numpy
Generator seeded per reply, as in `decode_loop`), `core.config.reply_seed` (seed rule).

Two decoder-layer families, detected explicitly in `locate()` (Handles.family):

  "gemma"   (Gemma 2/3: sandwich norms) the layer has post_feedforward_layernorm.
            x += post_attention_layernorm(self_attn(input_layernorm(x)))
            x += post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
  "prenorm" (Qwen2, Llama, Mistral) no post_feedforward_layernorm; self_attn, mlp,
            input_layernorm and post_attention_layernorm present.
            x += self_attn(input_layernorm(x))
            x += mlp(post_attention_layernorm(x))
            Here post_attention_layernorm is the norm BEFORE the MLP, not the attention output,
            so the family is never inferred from that module's name.

Hook points ("what attention / the MLP adds to the residual stream" in both families):
  resid     output of decoder layer i (the residual stream after layer i)
  attn_out  gemma: output of post_attention_layernorm; prenorm: output[0] of self_attn (o_proj)
  mlp_out   gemma: output of post_feedforward_layernorm; prenorm: output of mlp
  mlp_act   input of mlp.down_proj (the MLP's intermediate neuron activations)
  attn_w    attention probabilities returned by self_attn, one per query head (GQA keys are
            repeated); needs attn_implementation="eager"
For gemma, attn_out / mlp_out are post-norm outputs; for prenorm they are raw sublayer outputs,
so their absolute sizes are not comparable across families (within a model and layer they are).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.decoding import choose
from core.interfaces import GenConfig, Message

LAYER_TYPE_NAMES = {"sliding_attention": "sliding", "full_attention": "global"}


@dataclass
class Handles:
    model: Any
    tok: Any
    layers: list[Any]
    final_norm: Any
    lm_head: Any
    layer_types: list[str]          # "sliding" | "global", read from the model config
    sliding_window: int | None
    n_heads: int
    hidden_size: int
    intermediate_size: int
    softcap: float | None
    family: str = "gemma"           # "gemma" (sandwich norms) | "prenorm" (Qwen2 / Llama / Mistral)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device


def text_config(model: Any) -> Any:
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def layer_types_from_config(cfg: Any) -> list[str]:
    """Per-layer attention type from the config: `layer_types` when present, else Gemma's
    `sliding_window_pattern` (every N-th layer global), else uniform: all "sliding" when the
    config enables a sliding window for every layer (Mistral style), otherwise all "global"."""
    n = int(cfg.num_hidden_layers)
    types = getattr(cfg, "layer_types", None)
    if types:
        if len(types) != n:
            raise ValueError(f"config.layer_types has {len(types)} entries for {n} layers")
        return [LAYER_TYPE_NAMES.get(t, t) for t in types]
    pattern = getattr(cfg, "sliding_window_pattern", None)
    if pattern:
        return ["global" if (i + 1) % int(pattern) == 0 else "sliding" for i in range(n)]
    window = getattr(cfg, "sliding_window", None)
    if window and getattr(cfg, "use_sliding_window", True):
        return ["sliding"] * n
    return ["global"] * n


def layer_family(layer: Any) -> str:
    """'gemma' or 'prenorm', from the modules the decoder layer actually has (see module doc)."""
    if hasattr(layer, "post_feedforward_layernorm"):
        missing = [a for a in ("self_attn", "mlp", "post_attention_layernorm") if not hasattr(layer, a)]
        if missing:
            raise RuntimeError(f"Gemma-style layer without {missing}: unsupported decoder layer {type(layer).__name__}")
        return "gemma"
    need = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
    missing = [a for a in need if not hasattr(layer, a)]
    if missing:
        raise RuntimeError(
            f"unsupported decoder layer {type(layer).__name__}: not Gemma-2/3 (no post_feedforward_layernorm) "
            f"and not pre-norm (missing {missing}); supported: Gemma 2/3, Qwen2, Llama, Mistral"
        )
    return "prenorm"


def locate(model: Any, tok: Any) -> Handles:
    """Find the text decoder layers (skipping any vision tower), final norm and unembedding."""
    import torch

    cfg = text_config(model)
    n = int(cfg.num_hidden_layers)
    found = None
    for name, mod in model.named_modules():
        if "vision" in name or not isinstance(mod, torch.nn.ModuleList) or not name.endswith("layers"):
            continue
        if len(mod) == n and all(hasattr(m, "self_attn") and hasattr(m, "mlp") for m in mod):
            found = name
            break
    if found is None:
        raise RuntimeError("could not find the text decoder layers")
    layers = list(model.get_submodule(found))
    parent = model.get_submodule(found.rsplit(".", 1)[0]) if "." in found else model
    family = layer_family(layers[0])
    if any(layer_family(m) != family for m in layers):
        raise RuntimeError("decoder layers of mixed families")
    types = layer_types_from_config(cfg)
    window = getattr(cfg, "sliding_window", None)
    if "sliding" not in types:  # e.g. Qwen2.5: sliding_window may be set but no layer uses it
        window = None
    return Handles(
        model=model, tok=tok, layers=layers, final_norm=parent.norm, lm_head=model.get_output_embeddings(),
        layer_types=types, sliding_window=int(window) if window else None,
        n_heads=int(cfg.num_attention_heads), hidden_size=int(cfg.hidden_size),
        intermediate_size=int(cfg.intermediate_size),
        softcap=getattr(cfg, "final_logit_softcapping", None), family=family,
    )  # fmt: skip


# -- prompts and the secret span ---------------------------------------------------------------
def messages(system: str, user: str) -> list[Message]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class Encoded:
    ids: list[int]
    span: list[int]          # token positions of the secret in the prompt ([] without a secret)
    digit_pos: list[int]     # token position of each secret character, in order


def encode(hf: Any, system_template: str, user: str, code: str | None) -> Encoded:
    """Chat-templated prompt ids; with a code, locate it via the tokenizer's offset mapping.
    The code is located inside the system text it was formatted into (not anywhere else)."""
    system = system_template.format(code=code) if code is not None else system_template
    text = templated(hf, system_template, user, code)
    tok, _ = hf.load()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    if code is None:
        return Encoded(ids, [], [])
    s0 = text.find(system)
    if s0 < 0:
        raise AssertionError("system prompt not found verbatim in the chat-templated text")
    a = s0 + system_template.index("{code}")
    span_chars = (a, a + len(code))
    if text[span_chars[0]:span_chars[1]] != code:
        raise AssertionError(f"secret span mismatch: {text[span_chars[0]:span_chars[1]]!r} != {code!r}")
    span = secret_span(offs, span_chars)
    digit_pos = []
    for k in range(len(code)):
        hits = [i for i in span if offs[i][0] <= a + k < offs[i][1]]
        if len(hits) != 1:
            raise AssertionError(f"character {k} of the secret maps to tokens {hits}")
        digit_pos.append(hits[0])
    return Encoded(ids, span, digit_pos)


SENTINEL = "⁣CODE⁣"
_TEMPLATE_CACHE: dict[tuple[int, str, str], str] = {}


def templated(hf: Any, system_template: str, user: str, code: str | None) -> str:
    """Chat-templated text. The template is rendered once per (system, user) with a sentinel in
    place of the code and the code substituted afterwards (the chat template copies the system
    text verbatim; asserted by the sentinel appearing exactly once)."""
    if code is None:
        return hf.template(messages(system_template, user))
    key = (id(hf), system_template, user)
    if key not in _TEMPLATE_CACHE:
        text = hf.template(messages(system_template.format(code=SENTINEL), user))
        if text.count(SENTINEL) != 1:
            raise AssertionError("chat template did not copy the system prompt verbatim")
        _TEMPLATE_CACHE[key] = text
    return _TEMPLATE_CACHE[key].replace(SENTINEL, code)


def secret_span(offsets: Sequence[tuple[int, int]], chars: tuple[int, int]) -> list[int]:
    """Token positions whose character range overlaps `chars` = [start, end). Raises if none."""
    s, e = chars
    pos = [i for i, (a, b) in enumerate(offsets) if a < e and b > s and b > a]
    if not pos:
        raise AssertionError(f"secret span {chars} not found in the offset mapping")
    return pos


def check_equal_tokenisation(hf: Any, system_template: str, user: str, codes: Sequence[str]) -> dict[str, Any]:
    """Every code must give the same prompt length, span length and one token per digit, at the
    same positions. Fails loudly otherwise (positions must align across codes)."""
    ref = encode(hf, system_template, user, codes[0])
    bad = []
    for c in codes[1:]:
        e = encode(hf, system_template, user, c)
        if len(e.ids) != len(ref.ids) or e.span != ref.span or e.digit_pos != ref.digit_pos:
            bad.append((c, len(e.ids), e.span))
            if len(bad) >= 10:
                break
    if bad:
        raise AssertionError(
            f"codes tokenise differently from {codes[0]!r} (len {len(ref.ids)}, span {ref.span}): {bad}"
        )
    if len(set(ref.digit_pos)) != len(codes[0]):
        raise AssertionError(f"digits do not get one token each: {ref.digit_pos}")
    return {"prompt_tokens": len(ref.ids), "span": ref.span, "digit_pos": ref.digit_pos,
            "span_tokens": len(ref.span), "n_codes": len(codes)}  # fmt: skip


END_OF_TURN = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>")  # Gemma, Qwen (ChatML), Llama 3


def stop_ids(tok: Any, model: Any = None) -> list[int]:
    """eos, the model's generation_config eos ids, and any end-of-turn token the vocab has."""
    ids = {int(tok.eos_token_id)}
    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gc_eos is not None:
        ids.update(int(x) for x in (gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos]))
    for t in END_OF_TURN:
        eot = tok.convert_tokens_to_ids(t)
        if isinstance(eot, int) and eot != tok.unk_token_id:
            ids.add(eot)
    return sorted(ids)


# -- generation ----------------------------------------------------------------------------------
def generate(h: Handles, prompts: Sequence[Sequence[int]], seeds: Sequence[int], gen: GenConfig,
             min_new_tokens: int = 1) -> list[list[int]]:  # fmt: skip
    """Seeded sampling for rows of EQUAL prompt length (no padding), with a KV cache.

    Each row samples with its own numpy Generator(seed) via core.decoding.choose, exactly as
    core.decoding.decode_loop does, so a reply depends on its seed and not on its batch-mates
    (up to bf16 batch numerics). Stop tokens (eos, end-of-turn) are masked before
    `min_new_tokens` and are not part of the returned ids.
    """
    import torch

    if len({len(p) for p in prompts}) != 1:
        raise ValueError("generate() needs prompts of equal length (one prompt x many secrets)")
    stops = stop_ids(h.tok, h.model)
    pad = int(h.tok.pad_token_id if h.tok.pad_token_id is not None else stops[0])
    rngs = [np.random.default_rng(int(s)) for s in seeds]
    out: list[list[int]] = [[] for _ in prompts]
    done = [False] * len(prompts)
    ids = torch.tensor([list(p) for p in prompts], device=h.device)
    with torch.no_grad():
        res = _call(h.model, input_ids=ids, use_cache=True)
        for step in range(gen.max_new_tokens):
            logp = torch.log_softmax(res.logits[:, -1].float(), dim=-1)
            if step < min_new_tokens:
                logp[:, stops] = -float("inf")
            lp = logp.cpu().numpy()
            nxt = []
            for b in range(len(prompts)):
                if done[b]:
                    nxt.append(pad)
                    continue
                t = choose(lp[b], gen, rngs[b])
                if t in stops:
                    done[b] = True
                    nxt.append(pad)
                else:
                    out[b].append(t)
                    nxt.append(t)
            if all(done):
                break
            res = _call(h.model, input_ids=torch.tensor(nxt, device=h.device)[:, None],
                        past_key_values=res.past_key_values, use_cache=True)  # fmt: skip
    return out


def _call(model: Any, **kw: Any) -> Any:
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError:  # older transformers
        return model(**kw)


# -- teacher-forced passes and hooks --------------------------------------------------------------
HookFn = Callable[[str, int, Any], None]


class Capture:
    """Forward hooks calling fn(kind, layer, tensor) during a pass. Reduce inside fn: never keep
    full tensors. Use as a context manager; hooks are removed on exit."""

    def __init__(self, h: Handles, kinds: set[str], fn: HookFn) -> None:
        self.h, self.kinds, self.fn = h, set(kinds), fn
        self._hooks: list[Any] = []

    def __enter__(self) -> Capture:
        k, add = self.kinds, self._hooks.append
        for i, layer in enumerate(self.h.layers):
            if "resid" in k:
                add(layer.register_forward_hook(self._out("resid", i)))
            if "mlp_act" in k:
                add(layer.mlp.down_proj.register_forward_pre_hook(self._pre("mlp_act", i)))
            if self.h.family == "gemma":
                if "attn_out" in k:
                    add(layer.post_attention_layernorm.register_forward_hook(self._out("attn_out", i)))
                if "mlp_out" in k:
                    add(layer.post_feedforward_layernorm.register_forward_hook(self._out("mlp_out", i)))
                if "attn_w" in k:
                    add(layer.self_attn.register_forward_hook(self._attn(i, out=False, weights=True)))
            else:  # prenorm: what each sublayer adds to the residual stream, before the residual add
                if "mlp_out" in k:
                    add(layer.mlp.register_forward_hook(self._out("mlp_out", i)))
                if "attn_out" in k or "attn_w" in k:  # one hook on self_attn serves both
                    add(layer.self_attn.register_forward_hook(self._attn(i, out="attn_out" in k, weights="attn_w" in k)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for hk in self._hooks:
            hk.remove()
        self._hooks.clear()

    def _out(self, kind: str, i: int) -> Callable[..., None]:
        def hook(mod: Any, inp: Any, out: Any) -> None:
            self.fn(kind, i, out[0] if isinstance(out, tuple) else out)
        return hook

    def _pre(self, kind: str, i: int) -> Callable[..., None]:
        def hook(mod: Any, inp: Any) -> None:
            self.fn(kind, i, inp[0])
        return hook

    def _attn(self, i: int, out: bool, weights: bool) -> Callable[..., None]:
        def hook(mod: Any, inp: Any, res: Any) -> None:
            if out:
                self.fn("attn_out", i, res[0] if isinstance(res, tuple) else res)
            if weights:
                w = res[1] if isinstance(res, tuple) and len(res) > 1 else None
                if w is None:
                    raise RuntimeError("no attention weights: load the model with attn_implementation='eager'")
                self.fn("attn_w", i, w)
        return hook


def forward(h: Handles, seqs: Sequence[Sequence[int]]) -> None:
    """One teacher-forced pass over right-padded sequences (causal: real tokens never see the
    padding). Results come out through an active Capture; logits are not materialised."""
    import torch

    width = max(len(s) for s in seqs)
    pad = int(h.tok.pad_token_id if h.tok.pad_token_id is not None else 0)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(list(s))
        att[r, : len(s)] = 1
    with torch.no_grad():
        _call(h.model, input_ids=ids.to(h.device), attention_mask=att.to(h.device), use_cache=False)


def lens_logprobs(h: Handles, resid: Any) -> Any:
    """Logit lens: log-softmax(unembed(final_norm(resid))) for resid [..., D] (float32 out)."""
    import torch

    x = h.lm_head(h.final_norm(resid)).float()
    if h.softcap:
        x = torch.tanh(x / h.softcap) * h.softcap
    return torch.log_softmax(x, dim=-1)
