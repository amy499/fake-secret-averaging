"""Generator interface: produce n replies from one model holding one secret.

The mock and the later Hugging Face backend both implement this, so the API layer never
knows which one it is talking to (apart from `is_mock`, which drives the MOCK DATA badge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fsa.contract.schemas import Method


@dataclass(frozen=True)
class GeneratedReply:
    """One reply: decoded text plus, for real models, the sampled token IDs (incl. EOS)."""

    text: str
    token_ids: list[int] | None = None


class GeneratorUnavailable(RuntimeError):
    """This backend cannot generate for the request (reported as HTTP 501)."""


@dataclass(frozen=True)
class ModelSpec:
    """What to sample from: a method applied to a base model (the two Playground axes)."""

    method: Method
    base_model: str = ""  # registry id, e.g. "gemma3-4b-it" (ignored by the mock)
    hf_id: str = ""  # e.g. "google/gemma-3-4b-it"
    k: int | None = None  # fsa only
    adapter: str | None = None  # LoRA adapter path (real backend only)
    revision: str | None = None  # hub commit to load (registry base_models[].revision)
    # Pin for the adapter's adapter_model.safetensors (sha256 or a >= 16-hex prefix): checked
    # before the base model loads; None = not pinned (v1 adapters given by path).
    adapter_sha256: str | None = None

    @property
    def id(self) -> str:
        return f"{self.base_model}/{self.method}" if self.base_model else self.method


class Generator(Protocol):
    is_mock: bool
    note: str | None  # shown in the UI, e.g. the mock's base-model caveat

    def check(self, spec: ModelSpec) -> None:
        """Raise GeneratorUnavailable if `spec` cannot be served (cheap; called before generating)."""
        ...

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
        """Return n_samples replies to `prompt` from the model with `secret` in its system prompt.

        Same (secret, prompt, seed, temperature) must give the same replies, and two specs called
        with the same arguments must share their random stream (so side-by-side panels differ only
        by base model and method).
        """
        ...
