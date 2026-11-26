"""Abstract interfaces shared by every slot of the harness.

Status: ✅ implemented + tested (tests/test_registry.py, tests/test_dry_run.py).

Data flow of one run (see core/runner.py):

    SecretSource ──► Dataset.items(split) ──► list[Item]
    ModelWrapper ──► Defence.wrap(model) ──► generate() ──► list[Reply]
    Attacker.attack(model, items, replies, budget) ──► list[Guess]
    Metric.compute(items, replies, guesses) ──► dict
"""

from __future__ import annotations

import abc
import copy
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

Message = dict[str, str]  # {"role": "system" | "user" | "assistant" | "tool", "content": "..."}


# --------------------------------------------------------------------------- data records


@dataclass
class GenConfig:
    """Decoding settings. temperature == 0 means greedy decoding."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    thinking: bool | None = None  # None = model default; only used by models with a thinking mode

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


@dataclass
class Generation:
    """One model output. Reasoning text (thinking trace) is kept apart from the answer.

    `n_tokens` counts the generated tokens before the first stop token (EOS / end of turn), so
    without the stop token and padding. `finished` is True when a stop token was generated
    within the cap, False when the reply hit `max_new_tokens` (truncated), None when the model
    cannot tell. The protocol's truncation rate is the share of replies with finished False."""

    text: str
    reasoning: str = ""
    n_tokens: int = 0
    latency_s: float = 0.0
    finished: bool | None = None


@dataclass(frozen=True)
class SecretSpan:
    """One occurrence of the secret: characters [start, end) of messages[message]["content"]."""

    message: int
    start: int
    end: int


@dataclass(frozen=True)
class SecretContext:
    """What a technique is told about the secret in one prompt: the string and every place it
    occurs, as character spans per message, so the role of each occurrence is known (system
    context, or a user turn that types it, e.g. PasswordEval's right-password turn). Token
    positions depend on the tokenizer and chat template: HF models map spans to tokens
    (models/hf.py `secret_token_mask`). Passed to ModelWrapper.generate as `secret_ctx`, one per
    prompt (None when the prompt holds no secret, e.g. the plain MMLU/IFEval/GSM8K items and the
    no-secret controls).

    `protected` holds the item's other protected strings, when the benchmark has them (Item.protected,
    e.g. PasswordEval's confidential info, which the password guards). Empty for every other
    benchmark; only defences that ask for it read it (baselines/output_filter.py, mode `rule`)."""

    secret: str
    spans: tuple[SecretSpan, ...]
    protected: tuple[str, ...] = ()

    @staticmethod
    def find(messages: Sequence[Message], secret: str | None, protected: Sequence[str] = ()) -> SecretContext | None:
        """Every non-overlapping occurrence of `secret` in each message's content; None when
        there is no secret."""
        if not secret:
            return None
        spans = []
        for j, m in enumerate(messages):
            text, pos = m.get("content") or "", 0
            while (k := text.find(secret, pos)) >= 0:
                spans.append(SecretSpan(j, k, k + len(secret)))
                pos = k + len(secret)
        return SecretContext(secret, tuple(spans), tuple(p for p in protected if p))


def secret_contexts(prompts: Sequence[Sequence[Message]], secrets: Sequence[str | None],
                    protected: Sequence[Sequence[str]] | None = None) -> list[SecretContext | None]:  # fmt: skip
    """One SecretContext per prompt (None where the secret is None)."""
    prot = protected if protected is not None else [()] * len(secrets)
    return [SecretContext.find(p, s, q) for p, s, q in zip(prompts, secrets, prot)]


@dataclass
class Item:
    """One benchmark example.

    `secret` is the protected string that sits in the context. `allowed_fact` is context the
    model may disclose (e.g. a public key ID). `needs_secret` says whether the task can only be
    solved by *using* the secret (e.g. checking a code). `expected_use` is the reference answer
    or behaviour for scoring (benchmark-specific). `protected` lists other strings the item
    protects besides the secret (PasswordEval: the confidential info the password guards); it is
    passed to defences in SecretContext.protected and changes neither the prompt nor the scoring."""

    id: str
    system_prompt: str
    user_prompt: str | None = None
    turns: list[Message] | None = None
    secret: str | None = None
    allowed_fact: str | None = None
    needs_secret: bool = False
    expected_use: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    protected: tuple[str, ...] = ()

    def messages(self) -> list[Message]:
        msgs: list[Message] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        if self.turns:
            msgs.extend(dict(t) for t in self.turns)
        elif self.user_prompt is not None:
            msgs.append({"role": "user", "content": self.user_prompt})
        return msgs

    def secret_context(self) -> SecretContext | None:
        """The secret and its spans in messages() (what generate() receives as secret_ctx)."""
        return SecretContext.find(self.messages(), self.secret, self.protected)

    def with_secret(self, candidate: str | None, placeholder: str = "[hidden]") -> Item:
        """Copy of the item with every occurrence of the secret replaced by `candidate`
        (or by `placeholder` when candidate is None: the no-secret control)."""
        new = copy.deepcopy(self)
        if not self.secret:
            return new
        repl = placeholder if candidate is None else candidate

        def sub(s: str | None) -> str | None:
            return None if s is None else s.replace(self.secret, repl)  # type: ignore[arg-type]

        new.system_prompt = sub(new.system_prompt) or ""
        new.user_prompt = sub(new.user_prompt)
        if new.turns:
            new.turns = [{**t, "content": sub(t["content"]) or ""} for t in new.turns]
        new.secret = candidate
        return new

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Reply:
    """One stored generation for an item (a row of replies.jsonl)."""

    item_id: str
    sample: int  # -1 = greedy scored answer; 0..k-1 = sampled leak-test replies
    seed: int | None
    text: str
    reasoning: str = ""
    latency_s: float = 0.0
    n_tokens: int = 0
    control: bool = False  # True = generated with the secret removed (no-secret control)
    metadata: dict[str, Any] = field(default_factory=dict)
    finished: bool | None = None  # see Generation.finished

    @property
    def greedy(self) -> bool:
        return self.sample < 0


@dataclass
class Attempt:
    """One attacker query and what the attacker concluded after it."""

    query: int  # 1-based number of model queries used so far
    prompt: str
    reply: str
    guess: str | None
    reasoning: str = ""


@dataclass
class Guess:
    """An attacker's final answer for one item (a row of guesses.jsonl)."""

    item_id: str
    attacker: str
    guess: str | None
    queries: int = 0
    ranked: list[str] = field(default_factory=list)  # best-first candidates (truncated)
    attempts: list[Attempt] = field(default_factory=list)
    control: bool = False  # made from no-secret control replies
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- slots


class ModelWrapper(abc.ABC):
    """A chat model. Wrappers (defences) subclass this and delegate to `inner`."""

    name: str = "model"
    thinking: bool = False

    @abc.abstractmethod
    def generate(
        self,
        prompts: Sequence[list[Message]],
        gen_cfg: GenConfig,
        seeds: Sequence[int | None] | None = None,
        secret_ctx: Sequence[SecretContext | None] | None = None,
        **options: Any,
    ) -> list[Generation]:
        """One Generation per prompt. `seeds[i]` fixes sampling for prompt i. `secret_ctx[i]` is
        the secret in prompt i and its character spans (SecretContext; None = no secret); every
        caller in the harness passes it (protocol runner, attack-study runner, attackers), and a
        wrapper must pass it on (recomputed if it changes the messages). `options` are
        model-specific extras a wrapper may pass to its inner model (HF: `input_hook`)."""

    @abc.abstractmethod
    def logprobs(self, prompts: Sequence[list[Message]], continuations: Sequence[str],
                 secret_ctx: Sequence[SecretContext | None] | None = None, **options: Any) -> list[float]:  # fmt: skip
        """Sum of log p(continuation | prompt) in nats, one per (prompt, continuation) pair.
        The continuation is scored as the assistant's answer. `secret_ctx` / `options` as in
        generate (the likelihood attackers pass the span of the candidate that fills the
        secret's slot)."""

    # Optional token-level interface, used by decoding-time defences (token_ban, dp_fusion,
    # adler). Models that do not support it raise NotImplementedError.
    def token_level(self) -> TokenLevel:
        raise NotImplementedError(f"{self.name} has no token-level interface")

    @property
    def is_dummy(self) -> bool:
        return False


class TokenLevel(abc.ABC):
    """Step-wise access to a model's next-token distribution."""

    eos_id: int

    @abc.abstractmethod
    def encode_prompt(self, messages: list[Message], thinking: bool | None = None) -> list[int]: ...

    @abc.abstractmethod
    def encode_text(self, text: str) -> list[int]: ...

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str: ...

    @abc.abstractmethod
    def next_logprobs(self, prompt_ids: Sequence[int], generated: Sequence[int]) -> Any:
        """1-D numpy array of log-probabilities over the vocabulary."""


class SecretSource(abc.ABC):
    """A deterministic list of secrets of one type, with fixed splits."""

    name: str = "secret"
    seed: int = 0
    n_total: int = 1200
    SPLITS: dict[str, range] = {"train": range(0, 800), "val": range(800, 900), "test": range(900, 1200)}

    @abc.abstractmethod
    def generate(self, n: int, seed: int) -> list[str]: ...

    @abc.abstractmethod
    def sha256(self) -> str:
        """SHA-256 of the canonical list (n_total secrets with the default seed)."""

    def split(self, name: str) -> list[int]:
        if name not in self.SPLITS:
            raise KeyError(f"unknown split {name!r}; use one of {sorted(self.SPLITS)}")
        return list(self.SPLITS[name])

    @abc.abstractmethod
    def secrets(self) -> list[str]:
        """The canonical list (cached, hash-checked)."""

    @abc.abstractmethod
    def leaks(self, text: str, secret: str) -> bool:
        """Does `text` disclose `secret` (type-aware normalisation)?"""

    @abc.abstractmethod
    def extract(self, text: str) -> list[str]:
        """Candidate secrets mentioned in `text`, most likely first."""

    @abc.abstractmethod
    def log2_space(self) -> float:
        """log2 of the number of possible secrets (uniform prior)."""

    def chance(self) -> float:
        return 2.0 ** (-self.log2_space())


class Dataset(abc.ABC):
    """A benchmark. Subclasses also provide `task_score` for utility."""

    name: str = "dataset"

    @abc.abstractmethod
    def items(self, split: str) -> list[Item]: ...

    def task_score(self, item: Item, reply: Reply) -> float | None:
        """1.0 / 0.0 (or a fraction) for the item's task, None when the item has no task."""
        return None


class Defence(abc.ABC):
    name: str = "defence"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = dict(cfg or {})

    def prepare(self, items: Sequence[Item], secrets: SecretSource | None) -> None:
        """Called once before generation with the items it will see (lets a defence know which
        context strings are secret). Optional."""

    def train(self, cfg: dict[str, Any]) -> None:
        """Optional training step (adapters etc.)."""

    @abc.abstractmethod
    def wrap(self, model: ModelWrapper) -> ModelWrapper: ...


class Attacker(abc.ABC):
    name: str = "attacker"
    passive: bool = False  # True = reads benign replies only (no extra model queries)

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None) -> None:
        self.cfg = dict(cfg or {})
        self.ctx = ctx  # core.runner.RunContext

    @abc.abstractmethod
    def attack(
        self, model: ModelWrapper, items: Sequence[Item], replies: Sequence[Reply], budget: int
    ) -> list[Guess]: ...


class Metric(abc.ABC):
    name: str = "metric"

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None) -> None:
        self.cfg = dict(cfg or {})
        self.ctx = ctx

    @abc.abstractmethod
    def compute(
        self, items: Sequence[Item], replies: Sequence[Reply], guesses: Sequence[Guess]
    ) -> dict[str, Any]:
        """{"summary": {name: Estimate-dict}, "per_item": {name: {item_id: value}}}"""
