"""Real generator: a Hugging Face chat model sampling replies with the secret in its system prompt.

The prompt token IDs come from fsa.prompting.prompt_ids (the method's system template from
configs/training/prompts.yaml with the secret filled in) and is rendered with the model's own chat template; templates without a system role (some
Gemma versions) get the system text prepended to the user turn instead. Sampling is plain
temperature sampling (top_k=0, top_p=1, no repetition penalty), overriding the model's tuned
generation_config, so the measured digit distribution is the model's own. torch/transformers
are imported lazily: without the `ml` dependency group the API still runs and answers 501.
Weights are loaded straight onto the target device (device_map), so a GPU job never holds a
full CPU copy; dtype "auto" means bfloat16 on CUDA and float32 on CPU.
fsa/b3/random_layer need a LoRA adapter (configs/training/models.yaml: base_models[].adapters).
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from fsa_runtime.generators.base import GeneratedReply, GeneratorUnavailable, ModelSpec
from fsa.prompting import (  # noqa: F401  (re-exported for existing imports)
    PROMPT_ONLY_TEMPLATES,
    PROMPTS_YAML,
    SAMPLING,
    eos_ids,
    load_system_templates,
    prompt_ids,
    render_chat,
    system_prompt,
    until_eos,
)

PLAIN_METHODS = frozenset({"base", *PROMPT_ONLY_TEMPLATES})  # template key == method name
INSTALL_HINT = "the real backend needs torch + transformers: run `uv sync --group ml`"
# Hard ceiling on reply length: 20 two-digit numbers take ~60-80 tokens, so anything longer is
# chatter or degenerate repetition (which also slows CPU runs and the parser down).
MAX_NEW_TOKENS_CAP = 256
DTYPES = ("auto", "float32", "bfloat16", "float16")
# SAMPLING (fsa.prompting) is the one definition of how replies are sampled; generate() uses it
# and baseline runs record it in config.yaml (generation.*) for the likelihood attacker's check.
PADDING_SIDE = "left"  # tokenizer setting; the prompt itself is never padded (see below)
PROMPT_PADDING = "none (one prompt per generate call, num_return_sequences)"


def resolve_dtype(dtype: str, device: str) -> str:
    """ "auto" -> bfloat16 on CUDA, float32 elsewhere; explicit dtypes pass through."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    if dtype != "auto":
        return dtype
    return "bfloat16" if str(device).startswith("cuda") else "float32"


def _import_ml():
    try:
        import torch
        import transformers
    except ImportError as e:
        raise GeneratorUnavailable(f"{INSTALL_HINT} ({e.name} not installed)") from None
    return torch, transformers


class HFGenerator:
    is_mock: ClassVar[bool] = False
    note: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        templates: Mapping[str, str],
        device: str = "cpu",
        dtype: str = "auto",
        max_new_tokens: int = 160,
        batch_size: int = 8,
        token: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.templates = dict(templates)
        self.device = device
        self.dtype_setting = dtype  # resolved per device by .dtype
        self.max_new_tokens = max_new_tokens  # clamped by the property setter below
        self.batch_size = batch_size
        self._token = token
        self._cache_dir = cache_dir
        self._loaded: dict[tuple[str, str | None], tuple[Any, Any]] = {}
        self._lock = threading.Lock()  # one generation at a time (CPU-bound anyway)

    @property
    def dtype(self) -> str:
        """Effective dtype ("auto" resolved for the current device)."""
        return resolve_dtype(self.dtype_setting, self.device)

    @dtype.setter
    def dtype(self, value: str) -> None:
        resolve_dtype(value, self.device)  # validate
        self.dtype_setting = value

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value: int) -> None:
        self._max_new_tokens = max(1, min(int(value), MAX_NEW_TOKENS_CAP))

    @classmethod
    def from_settings(cls, settings: Any) -> HFGenerator:
        return cls(
            templates=load_system_templates(),
            device=settings.hf_device,
            dtype=settings.hf_dtype,
            max_new_tokens=settings.hf_max_new_tokens,
            batch_size=settings.hf_batch_size,
            token=settings.hf_token,
            cache_dir=settings.hf_home,
        )

    # -- Generator --------------------------------------------------------------------------

    def check(self, spec: ModelSpec) -> None:
        if spec.method in PLAIN_METHODS:
            return
        if not spec.adapter:
            raise GeneratorUnavailable(
                f"method {spec.method!r} needs a trained LoRA adapter for {spec.base_model!r}; "
                f"none is configured (add base_models[].adapters.{spec.method} in "
                "configs/training/models.yaml)"
            )
        if not Path(spec.adapter).exists():
            raise GeneratorUnavailable(f"adapter for {spec.base_model}/{spec.method} not found")

    def tokenizer(self, hf_id: str) -> Any:
        """Tokenizer only (no torch needed); used by scripts/check_tokenization.py too."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise GeneratorUnavailable(INSTALL_HINT) from None
        return AutoTokenizer.from_pretrained(hf_id, token=self._token, cache_dir=self._cache_dir)

    def _load(self, spec: ModelSpec) -> tuple[Any, Any]:
        key = (spec.hf_id, spec.adapter if spec.method not in PLAIN_METHODS else None)
        if key in self._loaded:
            return self._loaded[key]
        torch, transformers = _import_ml()
        try:
            tok = transformers.AutoTokenizer.from_pretrained(
                spec.hf_id, token=self._token, cache_dir=self._cache_dir
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                spec.hf_id,
                dtype=getattr(torch, self.dtype),
                device_map=self.device,  # load shards straight onto the device
                token=self._token,
                cache_dir=self._cache_dir,
            )
        except OSError as e:  # missing/gated repo, no network, no HF access
            raise GeneratorUnavailable(f"could not load {spec.hf_id}: {e}") from None
        if key[1]:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, key[1])
        model.eval()  # already on self.device via device_map
        tok.padding_side = PADDING_SIDE
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self._loaded[key] = (tok, model)
        return tok, model

    def generate(
        self,
        spec: ModelSpec,
        *,
        secret: str,
        prompt: str,
        n_samples: int,
        seed: int,
        temperature: float = 1.0,
    ) -> list[str]:
        return [
            r.text
            for r in self.generate_replies(
                spec,
                secret=secret,
                prompt=prompt,
                n_samples=n_samples,
                seed=seed,
                temperature=temperature,
            )
        ]

    def generate_replies(
        self,
        spec: ModelSpec,
        *,
        secret: str,
        prompt: str,
        n_samples: int,
        seed: int,
        temperature: float = 1.0,
    ) -> list[GeneratedReply]:
        """Replies with their sampled token IDs (up to and including the first EOS)."""
        self.check(spec)
        with self._lock:
            torch, _ = _import_ml()
            tok, model = self._load(spec)
            ids = prompt_ids(tok, self.templates, spec.method, secret, prompt)
            input_ids = torch.tensor([ids], device=self.device)
            attention_mask = torch.ones_like(input_ids)
            eos = eos_ids(tok, model)
            torch.manual_seed(seed)  # same seed -> same replies (same batch size, device)
            replies: list[GeneratedReply] = []
            for start in range(0, n_samples, self.batch_size):
                n = min(self.batch_size, n_samples - start)
                with torch.inference_mode():
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=temperature,
                        **SAMPLING,
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=n,
                        pad_token_id=tok.pad_token_id,
                    )
                for row in out[:, len(ids) :].tolist():
                    toks = until_eos(row, eos)
                    text = tok.decode(toks, skip_special_tokens=True).strip()
                    replies.append(GeneratedReply(text=text, token_ids=toks))
            return replies
