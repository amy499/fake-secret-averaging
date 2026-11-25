"""Training config (configs/training/experiments/*.yaml) for FSA, the B3 baseline and the random-layer
control. One schema covers all three: `method` names the run, `target` picks the training
target (mixture over fake secrets, or the frozen model with no secret = B3), and `lora.layers`
picks where the adapter goes (all | explicit list | random:N, seeded).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LayersSpec = Literal["all"] | list[int] | str  # "random:N" is validated below
DEFAULT_TRAIN_PROMPTS = ["rand_20_2digit", "rand_10_2digit"]
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSection(_Section):
    id: str  # registry id (configs/training/models.yaml) or a local path / HF id


class FsaSection(_Section):
    k: int = Field(default=8, ge=1)  # fake secrets per example (mixture target only)
    # sampled: k fresh fakes per example (sample_fake_secrets). all_others: every other training
    # secret, in a fixed order (k must be n_train - 1): an exact target with no sampling noise.
    fake_set: Literal["sampled", "all_others"] = "sampled"


class LoraSection(_Section):
    layers: LayersSpec = "all"
    target_modules: list[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    rank: int = Field(default=16, ge=1)
    alpha: float = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0, lt=1)

    @field_validator("layers")
    @classmethod
    def _layers(cls, v):
        if isinstance(v, str) and v != "all" and not re.fullmatch(r"random:[1-9][0-9]*", v):
            raise ValueError('lora.layers must be "all", a list of layer indices, or "random:N"')
        if isinstance(v, list) and (not v or len(set(v)) != len(v) or min(v) < 0):
            raise ValueError("lora.layers list must be non-empty, distinct and >= 0")
        return v


class SecretsSection(_Section):
    # Same split as scripts/baseline_leakage.py --seed split_seed with these sizes; training
    # only ever sees the train part.
    split_seed: int = 0
    n_train: int = Field(default=50, ge=2)
    n_heldout: int = Field(default=50, ge=1)
    # digits: "00".."99". words: the 100 words of word_list (relative to
    # ), e.g. configs/training/secrets/words_100.txt (fsa.secrets).
    # digits4: the 100 fixed 4-digit codes of configs/training/secrets/digits4_100.txt (no word_list).
    type: Literal["digits", "words", "digits4"] = "digits"
    word_list: str | None = None

    @model_validator(mode="after")
    def _word_list(self) -> SecretsSection:
        if (self.type == "words") != (self.word_list is not None):
            raise ValueError("secrets.word_list is required for, and only for, type: words")
        return self


class PromptsSection(_Section):
    train_system_methods: list[str] = Field(default_factory=lambda: ["base"])
    train_user_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_TRAIN_PROMPTS))
    # Declared evaluation prompts: training on any of them fails at startup
    # (fsa.eval.prompt_sets.check_train_prompts). Not the same as config.yaml
    # resolved.heldout_user_prompts, which lists every user prompt not trained on.
    heldout_user_prompts: list[str] = Field(default_factory=list)


class GenerationSection(_Section):
    max_new_tokens: int = Field(default=128, ge=1, le=256)
    temperature: float = Field(default=1.0, gt=0)
    offline_replies_per_prompt: int = Field(default=4, ge=1)  # cached per (secret, method, prompt)


class TrainSection(_Section):
    steps: int = Field(default=200, ge=1)
    lr: float = Field(default=1e-4, gt=0)
    warmup_steps: int = Field(default=10, ge=0)
    batch_size: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=4, ge=1)
    gradient_checkpointing: bool = True
    max_grad_norm: float = Field(default=1.0, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"
    checkpoint_every: int = Field(default=50, ge=1)


class UtilitySection(_Section):
    weight: float = Field(default=0.0, ge=0)  # lambda for KL(adapted || frozen) on general prompts


class NeedSection(_Section):
    """Need-preserving term: where the secret is legitimately needed (the user gives the
    password; the user asks to verify a code), keep the adapted model close to the FROZEN model
    holding the SAME real secret, KL(p_adapted(.|s, x) || p_frozen(.|s, x)) on reply tokens
    (fsa_loss, same direction as the FSA term). total = fsa_term + weight * need_term.

    Need items are EXTRA micro-batches on top of train.grad_accum FSA micro-batches, so the FSA
    examples of a step are identical with and without the need term. fraction is the share of a
    step's examples that are need items: n_need = round(grad_accum * fraction / (1 - fraction))
    micro-batches (>= 1), on steps where step % every == 0. weight 0 with prompts set: the need
    term is still computed and logged (no gradient), so a control run can be compared.
    """

    weight: float = Field(default=0.0, ge=0)
    prompts: list[str] = Field(default_factory=list)  # need user prompt ids ({password}/{code})
    every: int = Field(default=1, ge=1)  # need micro-batches on steps with step % every == 0
    fraction: float = Field(default=0.3, gt=0, lt=1)

    def n_micro(self, grad_accum: int) -> int:
        """Need micro-batches per step on which the need term runs (0 without prompts)."""
        if not self.prompts:
            return 0
        return max(1, round(grad_accum * self.fraction / (1 - self.fraction)))


class MonitorSection(_Section):
    every: int = Field(default=10, ge=1)  # KL-to-frozen on no-secret digit / general prompts
    # kl_to_full_mixture: the adapted model under the real secret vs the mixture over ALL other
    # training secrets, on a fixed batch of held-in examples (0 = off). Low-noise progress metric.
    full_mixture_every: int = Field(default=25, ge=0)
    full_mixture_examples: int = Field(default=4, ge=1)
    # Need monitors (only when need.prompts is set; 0 = off), on fixed held-in train secrets:
    # verification accuracy (Yes/No next-token logits), compliant exact-reveal rate and
    # withhold leak rate (greedy, need_max_new_tokens).
    need_every: int = Field(default=50, ge=0)
    need_secrets: int = Field(default=8, ge=1)
    need_max_new_tokens: int = Field(default=24, ge=1)
    # FSA prompts whose replies are checked for the code in the withhold leak rate.
    withhold_prompts: list[str] = Field(default_factory=list)


class TrainConfig(_Section):
    name: str
    seed: int = 0
    method: Literal["fsa", "b3", "random_layer"]
    model: ModelSection
    target: Literal["mixture", "no_secret"] = "mixture"
    reply_source: Literal["offline", "on_policy"] = "offline"
    fsa: FsaSection = Field(default_factory=FsaSection)
    lora: LoraSection = Field(default_factory=LoraSection)
    secrets: SecretsSection = Field(default_factory=SecretsSection)
    prompts: PromptsSection = Field(default_factory=PromptsSection)
    generation: GenerationSection = Field(default_factory=GenerationSection)
    train: TrainSection = Field(default_factory=TrainSection)
    utility: UtilitySection = Field(default_factory=UtilitySection)
    need: NeedSection = Field(default_factory=NeedSection)
    monitor: MonitorSection = Field(default_factory=MonitorSection)

    @model_validator(mode="after")
    def _consistent(self) -> TrainConfig:
        if self.method == "b3" and self.target != "no_secret":
            raise ValueError("method b3 is the no-secret target: set target: no_secret")
        if self.method != "b3" and self.target != "mixture":
            raise ValueError(f"method {self.method} uses target: mixture")
        if self.method == "random_layer" and not str(self.lora.layers).startswith("random:"):
            raise ValueError("method random_layer needs lora.layers: random:N")
        if self.target == "mixture" and self.fsa.k > self.secrets.n_train - 1:
            raise ValueError("fsa.k must leave room for the real secret in the train pool")
        if self.fsa.fake_set == "all_others" and self.fsa.k != self.secrets.n_train - 1:
            raise ValueError(
                f"fsa.fake_set all_others uses every other training secret: set fsa.k = "
                f"{self.secrets.n_train - 1} (n_train - 1)"
            )
        if self.need.weight > 0 and not self.need.prompts:
            raise ValueError("need.weight > 0 needs need.prompts")
        missing = sorted(set(self.monitor.withhold_prompts) - set(self.prompts.train_user_prompts))
        if missing:
            raise ValueError(f"monitor.withhold_prompts must be train_user_prompts: {missing}")
        return self
