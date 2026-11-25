"""Hugging Face chat models (torch + transformers, imported lazily).

| name               | HF id                     | status | needs                                              |
|--------------------|---------------------------|--------|----------------------------------------------------|
| gemma3-12b-it      | google/gemma-3-12b-it     | 🟡     | Gemma licence accepted on HF + HF_TOKEN; ~24 GB GPU |
| qwen2.5-7b-instruct| Qwen/Qwen2.5-7B-Instruct  | 🟡     | model download (~15 GB)                            |
| qwen3-8b           | Qwen/Qwen3-8B             | 🟡     | model download (~16 GB); thinking on/off           |

The registry/config path of each is unit-tested without loading weights (tests/test_registry.py);
none has been run on a real GPU from this harness yet, hence 🟡.

* bf16 on cuda by default (`dtype`, `device` in the model config). The dtype keyword is
  `dtype` on transformers >= 4.56 and `torch_dtype` before (the team's the Slurm cluster pin is 4.51.3);
  `_dtype_kwarg` picks the right one so bf16 is never silently dropped.
* `revision` (model config, optional): a hub commit / tag to load. Whatever is requested, the
  commit actually loaded (`config._commit_hash`) is reported by `run_info()`.
* `attn_implementation` (model config, optional): e.g. "eager" to get attention weights; unset
  keeps the transformers default (SDPA). The implementation in use is reported by `run_info()`.
* `batch_size` (model config, default 16, the team protocol's value) for greedy generation and
  for scoring. Greedy replies can change with batching, so it is reported by `run_info()`.
* Decoding: every `generate` call passes repetition_penalty=1.0, top_k=None and (greedy)
  top_p=None, temperature=None explicitly, so a model's generation_config defaults (Qwen2.5:
  repetition_penalty 1.05, top_k 20) never change "greedy". As in the team reference
  (the AI6130 Group 11 team protocol (Kaur et al., 2026)).
* Stop tokens: EOS ids from the tokenizer and generation_config, plus Gemma's <end_of_turn> and
  Qwen's <|im_end|>. `Generation.n_tokens` = tokens before the first stop token;
  `Generation.finished` = a stop token was generated within the cap (else truncated).
* Thinking: Qwen3's chat template takes `enable_thinking`; the reasoning between <think> and
  </think> is returned in Generation.reasoning, the rest in Generation.text.
* Sampled replies are generated one prompt at a time with torch.manual_seed(seed) so each reply
  is reproducible from its own seed; greedy replies are batched.
* Chat template: always the tokenizer's own (`apply_chat_template`). Gemma 3's template folds
  the system message into the first user turn itself; the harness never rewrites messages.
  `system_fold_check()` renders a probe conversation and reports where the system text landed.
* `token_level()` exposes a KV-cached step interface for decoding-time defences, with optional
  attention masking of given prompt positions (attention_block) and explicit position ids so
  masking does not shift the positions of later tokens.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from core.interfaces import GenConfig, Generation, Message, ModelWrapper, SecretContext, TokenLevel
from core.registry import register

DEFAULT_BATCH_SIZE = 16  # team protocol value (the team protocol's value)
END_OF_TURN_TOKENS = ("<end_of_turn>", "<|im_end|>")
FOLD_PROBE_SYSTEM = "SYSTEM-PROBE-7f3a"
FOLD_PROBE_USER = "USER-PROBE-9c1e"


def _dtype_kwarg(torch_dtype: Any) -> dict[str, Any]:
    """{"dtype": x} on transformers >= 4.56, {"torch_dtype": x} before (4.51.3 ignores `dtype`)."""
    import transformers

    major, minor = (int(p) for p in transformers.__version__.split(".")[:2])
    return {"dtype": torch_dtype} if (major, minor) >= (4, 56) else {"torch_dtype": torch_dtype}


def stop_token_ids(tok: Any, model: Any = None) -> set[int]:
    """EOS ids (tokenizer + generation_config) plus end-of-turn ids (Gemma, Qwen) that exist."""
    ids: set[int] = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    gc = getattr(model, "generation_config", None) if model is not None else None
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if eos is not None:
        ids.update(int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]))
    unk = getattr(tok, "unk_token_id", None)
    for t in END_OF_TURN_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != unk and i >= 0:
            ids.add(i)
    return ids


def cut_at_stop(ids: Sequence[int], stops: set[int], pad_id: int | None) -> tuple[list[int], bool]:
    """(generated tokens before the first stop token, finished). Without a stop token the
    sequence ran to the cap: padding (never inside an unfinished row) is dropped, finished=False."""
    for k, t in enumerate(ids):
        if t in stops:
            return list(ids[:k]), True
    return [t for t in ids if t != pad_id], False


# The only tensors handed to generate() / forward(): every causal LM the harness loads takes them
# (Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Qwen2ForCausalLM). Fast tokenizers can also
# return token_type_ids; transformers 4.51.3's generate() rejects any kwarg the model's forward
# does not use ("The following `model_kwargs` are not used by the model: ['token_type_ids']"), and
# none of these models needs it for text generation.
MODEL_INPUT_KEYS = ("input_ids", "attention_mask")
DROPPED_INPUT_KEYS = ("token_type_ids",)


def model_inputs(enc: Any, device: Any = None, where: str = "inputs") -> dict[str, Any]:
    """Only input_ids and attention_mask from a tokenizer batch or a hook's dict, optionally moved
    to `device`. token_type_ids is dropped; any other key raises ValueError (an input_hook asking
    for an input this path cannot pass is a bug, not something to drop silently)."""
    keys = set(enc.keys())
    unknown = sorted(keys - set(MODEL_INPUT_KEYS) - set(DROPPED_INPUT_KEYS))
    if unknown:
        raise ValueError(f"{where}: inputs the model is not given here: {unknown} "
                         f"(only {', '.join(MODEL_INPUT_KEYS)})")  # fmt: skip
    if "input_ids" not in keys:
        raise ValueError(f"{where}: no input_ids")
    out = {k: enc[k] for k in MODEL_INPUT_KEYS if k in keys}
    return {k: v.to(device) for k, v in out.items()} if device is not None else out


SDPA_BACKENDS = ("default", "math")


@contextlib.contextmanager
def sdpa_backend(name: str | None = None) -> Iterator[dict[str, Any]]:
    """Run the block with PyTorch's CUDA SDPA backends set for `name`, then restore them.

    "default": unchanged (PyTorch picks flash / memory-efficient / cuDNN / math).
    "math": flash, memory-efficient and cuDNN SDPA off, math on. Same attention, computed without
    the fused kernels. Used for ConfAIde Tier 4: its padded batches of long prompts crash the
    memory-efficient kernel ("RuntimeError: p.attn_bias_ptr is not correctly aligned"; Amruta's
    the team protocol's runs), and math SDPA is the setting of the
    team protocol's ConfAIde rows.
    Yields {"backend", "flags"} for summary.json; flags is None without torch (dry runs)."""
    name = name or "default"
    if name not in SDPA_BACKENDS:
        raise ValueError(f"sdpa_backend {name!r}: use one of {SDPA_BACKENDS}")
    try:
        import torch
    except ImportError:
        yield {"backend": name, "flags": None}
        return
    cuda = torch.backends.cuda
    kinds = [k for k in ("flash", "mem_efficient", "cudnn", "math") if hasattr(cuda, f"{k}_sdp_enabled")]
    before = {k: getattr(cuda, f"{k}_sdp_enabled")() for k in kinds}
    try:
        if name == "math":
            for k in kinds:
                getattr(cuda, f"enable_{k}_sdp")(k == "math")
        yield {"backend": name, "flags": {k: getattr(cuda, f"{k}_sdp_enabled")() for k in kinds}}
    finally:
        for k, v in before.items():
            getattr(cuda, f"enable_{k}_sdp")(v)


def encode_batch(tok: Any, texts: Sequence[str], device: Any = None) -> dict[str, Any]:
    """Tokenize rendered prompts for generate(): left/right padding as the tokenizer is set, no
    special tokens added (the chat template has them), no token_type_ids."""
    enc = tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=False,
              return_token_type_ids=False)  # fmt: skip
    return model_inputs(enc, device, "tokenizer")


def split_thinking(text: str) -> tuple[str, str]:
    """(answer, reasoning) from a raw completion that may hold <think>...</think>.
    An unterminated <think> (generation cut off while reasoning) is all reasoning."""
    if "</think>" in text:
        head, answer = text.split("</think>", 1)
        return answer.strip(), head.replace("<think>", "").strip()
    if "<think>" in text:
        before, after = text.split("<think>", 1)
        return before.strip(), after.strip()
    return text.strip(), ""


class HFModel(ModelWrapper):
    hf_id = ""
    supports_thinking = False

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.hf_id = cfg.get("hf_id", self.hf_id)
        self.device = cfg.get("device", "cuda")
        self.dtype = cfg.get("dtype", "bfloat16")
        self.batch_size = int(cfg.get("batch_size", DEFAULT_BATCH_SIZE))
        self.revision: str | None = cfg.get("revision")
        self.thinking = bool(cfg.get("thinking", False))
        self.adapter: str | None = cfg.get("adapter")
        # "eager" is needed to read attention weights (analysis/internals); None keeps the
        # transformers default (SDPA), which is what every harness run uses.
        self.attn_implementation: str | None = cfg.get("attn_implementation")
        self._tok: Any = None
        self._model: Any = None

    # -- loading -----------------------------------------------------------------------------
    def load(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            import transformers

            rev = {"revision": self.revision} if self.revision else {}
            tok = transformers.AutoTokenizer.from_pretrained(self.hf_id, **rev)
            kwargs: dict[str, Any] = {**rev, **_dtype_kwarg(getattr(torch, self.dtype))}
            if self.attn_implementation:  # only when asked: transformers' default (SDPA) otherwise
                kwargs["attn_implementation"] = self.attn_implementation
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.hf_id, device_map=self.device, **kwargs
            )
            if self.adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter)
            model.eval()
            tok.padding_side = "left"
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            self._tok, self._model = tok, model
        return self._tok, self._model

    def load_adapter(self, path: str) -> None:
        """Attach a PEFT/LoRA adapter (used by training-based defences)."""
        self.adapter = path
        if self._model is not None:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, path)
            self._model.eval()

    def adapter_off(self) -> Any:
        """Context manager: score/generate with the base weights only (public-weights attacks)."""
        import contextlib

        _, model = self.load()
        disable = getattr(model, "disable_adapter", None)
        return disable() if disable is not None else contextlib.nullcontext()

    def template(self, messages: list[Message], thinking: bool | None = None) -> str:
        tok, _ = self.load()
        kwargs: dict[str, Any] = {}
        if self.supports_thinking:
            kwargs["enable_thinking"] = self.thinking if thinking is None else thinking
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)

    # -- secret spans -> tokens ----------------------------------------------------------------
    @staticmethod
    def rendered_secret_spans(rendered: str, messages: Sequence[Message], ctx: SecretContext | None) -> list[tuple[int, int]]:
        """Character spans of the secret in the rendered prompt. Each message's content is found
        in the rendered text in order (chat templates copy it verbatim, possibly trimmed; Gemma 3
        puts the system text inside the first user turn, which is still in order), and the
        message-level spans are shifted to it. Raises ValueError when a message holding a span
        cannot be located, rather than guessing."""
        if ctx is None or not ctx.spans:
            return []
        out, cursor, where = [], 0, {}
        for j, m in enumerate(messages):
            content = m.get("content") or ""
            core = content.strip()
            if not core:
                continue
            k = rendered.find(core, cursor)
            if k < 0:
                where[j] = None
                continue
            where[j] = k - (len(content) - len(content.lstrip()))  # offset of content[0]
            cursor = k + len(core)
        for s in ctx.spans:
            base = where.get(s.message)
            if base is None:
                raise ValueError(f"message {s.message} not found verbatim in the rendered prompt")
            a, b = base + s.start, base + s.end
            if rendered[a:b] != ctx.secret:
                raise ValueError("secret span does not line up with the rendered prompt")
            out.append((a, b))
        return out

    def secret_token_mask(self, messages: Sequence[Message], ctx: SecretContext | None,
                          thinking: bool | None = None) -> list[bool]:  # fmt: skip
        """One flag per prompt token (exactly the ids generate() feeds the model, before padding):
        True where the token overlaps a character of the secret."""
        tok, _ = self.load()
        text = self.template(list(messages), thinking)
        spans = self.rendered_secret_spans(text, messages, ctx)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=bool(spans))
        if not spans:
            return [False] * len(enc["input_ids"])
        return [b > a and any(a < e and b > s for s, e in spans) for a, b in enc["offset_mapping"]]

    # -- generation --------------------------------------------------------------------------
    def generate(
        self,
        prompts: Sequence[list[Message]],
        gen_cfg: GenConfig,
        seeds: Sequence[int | None] | None = None,
        secret_ctx: Sequence[SecretContext | None] | None = None,
        input_hook: Callable[[dict[str, Any], Any], dict[str, Any] | None] | None = None,
        **options: Any,
    ) -> list[Generation]:
        """`input_hook(enc, secret_mask)`, if given, is called on every encoded batch just before
        transformers' generate: `enc` holds input_ids and attention_mask (left-padded tensors),
        `secret_mask` is a bool tensor of the same shape, True on the tokens of the secret (from
        `secret_ctx`). It may edit enc in place or return a new dict (e.g. zero the attention
        mask on the secret tokens)."""
        import torch

        tok, model = self.load()
        # Batched generation needs left padding whatever load() did (subclasses override load).
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        seeds = list(seeds) if seeds is not None else [None] * len(prompts)
        texts = [self.template(p, gen_cfg.thinking) for p in prompts]
        out: list[Generation] = []
        groups = (
            [list(range(i, min(i + self.batch_size, len(texts)))) for i in range(0, len(texts), self.batch_size)]
            if gen_cfg.greedy
            else [[i] for i in range(len(texts))]
        )
        results: dict[int, Generation] = {}
        stops = stop_token_ids(tok, model)
        for group in groups:
            if not gen_cfg.greedy and seeds[group[0]] is not None:
                torch.manual_seed(int(seeds[group[0]]))  # type: ignore[arg-type]
            enc = encode_batch(tok, [texts[i] for i in group], model.device)
            if input_hook is not None:
                width = enc["input_ids"].shape[1]
                rows = []
                for i in group:
                    m = self.secret_token_mask(prompts[i], (secret_ctx or [None] * len(prompts))[i], gen_cfg.thinking)
                    rows.append([False] * (width - len(m)) + m)  # left padding
                mask = torch.tensor(rows, dtype=torch.bool, device=enc["input_ids"].device)
                enc = model_inputs(input_hook(dict(enc), mask) or enc, where="input_hook")
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(**enc, **self.generate_kwargs(gen_cfg, tok.pad_token_id))
            dt = (time.perf_counter() - t0) / len(group)
            new = gen[:, enc["input_ids"].shape[1] :]
            for row, i in enumerate(group):
                ids, finished = cut_at_stop(new[row].tolist(), stops, tok.pad_token_id)
                raw = tok.decode(ids, skip_special_tokens=True)
                answer, reasoning = split_thinking(raw)
                results[i] = Generation(text=answer, reasoning=reasoning, n_tokens=len(ids), latency_s=dt,
                                        finished=finished)  # fmt: skip
        out = [results[i] for i in range(len(texts))]
        return out

    @staticmethod
    def generate_kwargs(gen_cfg: GenConfig, pad_token_id: int | None) -> dict[str, Any]:
        """Keyword arguments for transformers' generate. Every decoding knob is set explicitly so
        the model's generation_config (e.g. Qwen2.5's repetition_penalty 1.05) cannot leak in."""
        return {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "do_sample": not gen_cfg.greedy,
            "temperature": gen_cfg.temperature if not gen_cfg.greedy else None,
            "top_p": gen_cfg.top_p if not gen_cfg.greedy else None,
            "top_k": None,
            "repetition_penalty": 1.0,
            "pad_token_id": pad_token_id,
        }

    # -- what was actually run ---------------------------------------------------------------
    def run_info(self) -> dict[str, Any]:
        """Model facts for summary.json: requested and loaded revision, class, dtype, attention
        implementation, batch size, chat-template hash, library versions."""
        import torch
        import transformers

        tok, model = self.load()
        cfg = getattr(model, "config", None)
        try:
            dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        except (StopIteration, AttributeError, TypeError):
            dtype = None
        template = getattr(tok, "chat_template", None) or ""
        if isinstance(template, dict):  # some tokenizers store several named templates
            template = "\n".join(f"{k}\n{v}" for k, v in sorted(template.items()))
        try:
            import tokenizers

            tok_version = tokenizers.__version__
        except ImportError:  # pragma: no cover
            tok_version = None
        return {
            "name": self.name, "hf_id": self.hf_id, "revision_requested": self.revision,
            "revision_loaded": getattr(cfg, "_commit_hash", None), "class": type(model).__name__,
            "dtype_requested": self.dtype, "dtype_loaded": dtype,
            "attn_implementation": getattr(cfg, "_attn_implementation", None) or self.attn_implementation,
            "adapter": self.adapter, "batch_size": self.batch_size,
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if template else None,
            "stop_token_ids": sorted(stop_token_ids(tok, model)),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "tokenizers": tok_version},
        }  # fmt: skip

    def system_fold_check(self) -> dict[str, Any]:
        """Render [system, user] with the model's own template and report the layout: for Gemma 3
        the system text must sit inside the first user turn, before the user text (the template
        folds it); for ChatML models (Qwen) it is its own system turn. Nothing is rewritten."""
        text = self.template([{"role": "system", "content": FOLD_PROBE_SYSTEM},
                              {"role": "user", "content": FOLD_PROBE_USER}])  # fmt: skip
        s, u = text.find(FOLD_PROBE_SYSTEM), text.find(FOLD_PROBE_USER)
        between = text[s + len(FOLD_PROBE_SYSTEM) : u] if 0 <= s < u else ""
        if s < 0:
            layout = "system text dropped"
        elif "<start_of_turn>user" in text[:s] and "<start_of_turn>" not in between:
            layout = "folded into first user turn"
        elif "<|im_start|>system" in text[:s] and "<|im_start|>user" in between:
            layout = "separate system turn"
        else:
            layout = "other"
        return {"layout": layout, "system_before_user": 0 <= s < u, "rendered": text}

    # -- scoring -----------------------------------------------------------------------------
    def logprobs(self, prompts: Sequence[list[Message]], continuations: Sequence[str],
                 secret_ctx: Sequence[SecretContext | None] | None = None,
                 input_hook: Callable[[dict[str, Any], Any], dict[str, Any] | None] | None = None,
                 **options: Any) -> list[float]:  # fmt: skip
        """`input_hook(enc, secret_mask)` as in generate, on the right-padded prompt+continuation
        batch (secret_mask covers prompt tokens only). Note: this forward pass gets no
        position_ids, so positions stay 0..n-1 whatever the hook does to attention_mask, whereas
        generate() derives them from attention_mask (see tests/test_secret_spans.py)."""
        import torch

        tok, model = self.load()
        scores: list[float] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_p = prompts[start : start + self.batch_size]
            batch_c = continuations[start : start + self.batch_size]
            p_ids = [tok(self.template(p), add_special_tokens=False)["input_ids"] for p in batch_p]
            c_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in batch_c]
            seqs = [p + c for p, c in zip(p_ids, c_ids)]
            width = max(len(s) for s in seqs)
            pad = tok.pad_token_id
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):  # right padding for scoring
                ids[r, : len(s)] = torch.tensor(s)
                att[r, : len(s)] = 1
            ids, att = ids.to(model.device), att.to(model.device)
            enc: dict[str, Any] = {"input_ids": ids, "attention_mask": att}
            if input_hook is not None:
                ctxs = list(secret_ctx or [None] * len(prompts))[start : start + self.batch_size]
                mask = torch.zeros((len(seqs), width), dtype=torch.bool)
                for r, (p, ctx) in enumerate(zip(batch_p, ctxs)):
                    m = self.secret_token_mask(p, ctx)
                    mask[r, : len(m)] = torch.tensor(m, dtype=torch.bool)
                enc = input_hook(enc, mask.to(model.device)) or enc
            enc = model_inputs(enc, where="logprobs")
            with torch.no_grad():
                logits = model(**enc).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            for r, (p, c) in enumerate(zip(p_ids, c_ids)):
                pos = torch.arange(len(p) - 1, len(p) + len(c) - 1, device=logp.device)
                tgt = torch.tensor(c, device=logp.device)
                scores.append(float(logp[r, pos, tgt].sum()))
        return scores

    def token_level(self) -> TokenLevel:
        return HFTokenLevel(self)


class HFTokenLevel(TokenLevel):
    """KV-cached next-token log-probs. `mask_positions` hides prompt positions from attention."""

    def __init__(self, model: HFModel, mask_positions: Sequence[int] = ()) -> None:
        self.m = model
        tok, _ = model.load()
        self.tok = tok
        self.eos_id = int(tok.eos_token_id)
        self.mask_positions = set(mask_positions)
        self._cache: dict[tuple[int, ...], tuple[list[int], Any, Any]] = {}

    def with_mask(self, positions: Sequence[int]) -> HFTokenLevel:
        return HFTokenLevel(self.m, positions)

    def encode_prompt(self, messages: list[Message], thinking: bool | None = None) -> list[int]:
        return self.tok(self.m.template(messages, thinking), add_special_tokens=False)["input_ids"]

    def encode_text(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)

    def span_positions(self, messages: list[Message], needle: str) -> list[int]:
        """Token positions of every occurrence of `needle` in the templated prompt."""
        text = self.m.template(messages)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)]
        return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if any(a < e and b > s for s, e in spans)]

    def next_logprobs(self, prompt_ids: Sequence[int], generated: Sequence[int]) -> np.ndarray:
        import torch

        _, model = self.m.load()
        key = tuple(prompt_ids)
        seq = list(prompt_ids) + list(generated)
        cached = self._cache.get(key)
        if cached is not None and cached[0] == seq[: len(cached[0])] and len(cached[0]) < len(seq):
            done, past, _ = cached
            new_ids = seq[len(done) :]
        else:
            done, past, new_ids = [], None, seq
        start = len(done)
        mask = torch.ones((1, len(seq)), dtype=torch.long)
        for p in self.mask_positions:
            if p < len(seq):
                mask[0, p] = 0
        pos = torch.arange(start, len(seq)).unsqueeze(0)
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([new_ids], device=model.device),
                attention_mask=mask.to(model.device),
                position_ids=pos.to(model.device),
                past_key_values=past,
                use_cache=True,
            )
        logits = out.logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = (seq, out.past_key_values, logp)
        return logp


@register("model", "gemma3-12b-it")
class Gemma3_12B(HFModel):
    name, hf_id = "gemma3-12b-it", "google/gemma-3-12b-it"


@register("model", "gemma3-4b-it")
class Gemma3_4B(HFModel):
    name, hf_id = "gemma3-4b-it", "google/gemma-3-4b-it"


@register("model", "qwen2.5-7b-instruct")
class Qwen25_7B(HFModel):
    name, hf_id = "qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct"


@register("model", "qwen2.5-1.5b-instruct")
class Qwen25_15B(HFModel):
    name, hf_id = "qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct"


@register("model", "qwen3-8b")
class Qwen3_8B(HFModel):
    """`thinking: true|false` in the model config. Reasoning is returned separately."""

    name, hf_id, supports_thinking = "qwen3-8b", "Qwen/Qwen3-8B", True


@register("model", "qwen3-8b-think")
class Qwen3_8B_Think(Qwen3_8B):
    name = "qwen3-8b-think"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__({**(cfg or {}), "thinking": True})


@register("model", "qwen3-8b-nothink")
class Qwen3_8B_NoThink(Qwen3_8B):
    name = "qwen3-8b-nothink"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__({**(cfg or {}), "thinking": False})
