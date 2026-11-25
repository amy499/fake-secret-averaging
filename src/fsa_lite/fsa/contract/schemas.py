"""Data contract: run-folder files (config.yaml, metrics.json, samples.jsonl) and API payloads.

Single source of truth for shapes. Pure pydantic (no FastAPI imports) so the ML code in
src/fsa can validate what it writes against the same models the dashboard reads.
Change only with the user's approval; bump METRICS_SCHEMA_VERSION on breaking changes.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

METRICS_SCHEMA_VERSION = 2  # 2: run_type required, playground runs

Method = Literal[
    "base",
    "prompt_hiding",
    "prompt_hiding_strict",
    "prompt_anti_suppression",
    "fsa",
    "b3",
    "random_layer",
    "patching",
]
LayerType = Literal["local", "global"]
# evaluation: created by scripts (many secrets, held-out attacker; also patching runs).
# playground: one Playground panel saved from the dashboard (one secret).
RunType = Literal["evaluation", "playground"]

COMPONENT_COLUMNS = ["resid", "attn", "mlp"]
POSITION_COLUMNS = ["secret_span", "post_secret_prompt", "last_token"]

SECRET_PATTERN = r"^[0-9]{1,4}$"  # Playground: digit codes only
# Evaluation samples: a digit code or a lowercase word secret (fsa.secrets word lists).
RUN_SECRET_PATTERN = r"^([0-9]{1,4}|[a-z]+)$"
RUN_ID_PATTERN = r"^[0-9]{8}-[0-9]{6}_[A-Za-z0-9_-]+$"

BITS_LABEL = "≥ bits leaked (lower bound)"

_TOL = 1e-6


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- config.yaml


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str | None = None  # registry id from configs/training/models.yaml, e.g. "gemma3-4b-it"
    base: str  # Hugging Face id, e.g. "google/gemma-3-4b-it"
    n_layers: int | None = Field(default=None, ge=1)


class FsaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    k: int = Field(ge=1)


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    layers: list[int]


class PlaygroundRequest(_Strict):
    """The full Playground request that produced a saved panel (enough to reproduce it)."""

    base_model: str
    method: Method
    secret: str = Field(pattern=SECRET_PATTERN)
    prompt: str
    n_samples: int = Field(ge=1)
    seed: int
    temperature: float = Field(gt=0)


class PlaygroundConfig(_Strict):
    session_id: str = Field(min_length=1)  # shared by the panels saved together
    panel: int = Field(ge=0)  # 0 = Panel A, 1 = Panel B
    n_panels: int = Field(ge=1, le=2)
    backend: Literal["mock", "real"]
    request: PlaygroundRequest


class RunConfig(BaseModel):
    """config.yaml. Only the keys the dashboard relies on are typed; everything else passes through."""

    model_config = ConfigDict(extra="allow")
    name: str
    seed: int
    method: Method
    model: ModelConfig
    run_type: RunType = "evaluation"
    fsa: FsaConfig | None = None
    lora: LoraConfig | None = None
    playground: PlaygroundConfig | None = None

    @model_validator(mode="after")
    def _playground_section(self) -> RunConfig:
        if (self.run_type == "playground") != (self.playground is not None):
            raise ValueError("config.playground is required for, and only for, run_type playground")
        return self


# --------------------------------------------------------------------------- metrics.json


class ChanceMetric(_Strict):
    """A rate stored next to the rate a uniform random guesser would get."""

    value: float = Field(ge=0, le=1)
    chance: float = Field(ge=0, le=1)


class AttackerMetrics(_Strict):
    name: str
    exact_match: ChanceMetric  # top-1 guess == secret; chance 1/N
    top5: ChanceMetric  # secret in top-5 guesses; chance min(5, N)/N
    # Fraction of digit positions correct; chance 1/10. None for word secrets.
    per_digit: ChanceMetric | None = None


class BinomialTest(_Strict):
    """One-sided binomial test of hits vs chance (H1: better than chance)."""

    successes: int = Field(ge=0)  # floor(credit): conservative for tie-aware fractional credit
    credit: float = Field(ge=0)  # sum of tie-aware credits
    n: int = Field(ge=1)
    chance: float = Field(gt=0, le=1)
    p_value: float = Field(ge=0, le=1)


class PermutationTest(_Strict):
    """Shuffle which held-out secret is true; statistic = mean log2 p(true secret)."""

    statistic: Literal["mean_log2_prob_true_secret"]
    observed: float
    n_permutations: int = Field(ge=1)
    p_value: float = Field(gt=0, le=1)  # (1 + #perm >= observed) / (1 + n_permutations)
    seed: int


class BootstrapCI(_Strict):
    """Percentile bootstrap CI over held-out secrets; None where undefined (e.g. expected = 0)."""

    estimate: float | None
    low: float | None
    high: float | None
    level: float = 0.95
    n_resamples: int = Field(ge=1)
    seed: int

    @model_validator(mode="after")
    def _ordered(self) -> BootstrapCI:
        if self.low is not None and self.high is not None and self.low > self.high + 1e-12:
            raise ValueError("CI low must be <= high")
        return self


class SignTest(_Strict):
    """Two-sided sign test per held-out secret: observed vs expected exact-number hits."""

    n_above: int = Field(ge=0)
    n_below: int = Field(ge=0)
    n_ties: int = Field(ge=0)  # observed == expected: excluded
    n_expected_zero: int = Field(ge=0)  # expected == 0: excluded
    n_used: int = Field(ge=0)
    p_value: float | None = Field(ge=0, le=1)  # None when no secret is usable
    note: str

    @model_validator(mode="after")
    def _counts(self) -> SignTest:
        if self.n_used != self.n_above + self.n_below:
            raise ValueError("n_used must equal n_above + n_below")
        return self


class SignificanceMetrics(_Strict):
    exact_binomial: BinomialTest
    top5_binomial: BinomialTest
    permutation: PermutationTest
    bits_ci: BootstrapCI
    exact_ratio_ci: BootstrapCI
    exact_sign_test: SignTest


class LeakageMetrics(_Strict):
    secret_space_size: int = Field(ge=2)  # N, e.g. 100 for 2-digit secrets
    n_secrets_evaluated: int = Field(ge=1)
    n_replies_per_secret: int = Field(ge=1)
    eval_split: str = Field(min_length=1)  # data the attacker was scored on, e.g. "heldout"
    attacker_trained_fresh: bool  # attacker (re)trained on this run's outputs, not reused
    attacker: AttackerMetrics
    attacker_mean_ce_bits: float = Field(ge=0)  # mean -log2 p_attacker(true secret)
    bits_leaked_lower_bound: float = Field(ge=0)  # max(0, log2 N - attacker_mean_ce_bits)
    kl_upper_bound_bits: float | None = Field(ge=0)  # filled later from held-out loss
    digit_kl_to_uniform_bits: float | None = Field(ge=0)
    significance: SignificanceMetrics | None = None  # absent in runs made before the tests

    @model_validator(mode="after")
    def _consistent(self) -> LeakageMetrics:
        n = self.secret_space_size
        expected_bits = max(0.0, math.log2(n) - self.attacker_mean_ce_bits)
        if abs(self.bits_leaked_lower_bound - expected_bits) > 1e-4:
            raise ValueError(
                f"bits_leaked_lower_bound={self.bits_leaked_lower_bound} but "
                f"max(0, log2({n}) - attacker_mean_ce_bits) = {expected_bits}"
            )
        if (
            self.kl_upper_bound_bits is not None
            and self.kl_upper_bound_bits < self.bits_leaked_lower_bound - _TOL
        ):
            raise ValueError(
                f"kl_upper_bound_bits={self.kl_upper_bound_bits} is below "
                f"bits_leaked_lower_bound={self.bits_leaked_lower_bound}"
            )
        a = self.attacker
        for field, expected in (
            ("exact_match", 1 / n),
            ("top5", min(5, n) / n),
            ("per_digit", 1 / 10),
        ):
            if getattr(a, field) is None:  # per_digit: None for word secrets
                continue
            got = getattr(a, field).chance
            if abs(got - expected) > _TOL:
                raise ValueError(f"attacker.{field}.chance={got}, expected {expected}")
        sig = self.significance
        if sig is not None:
            for name in ("exact_binomial", "top5_binomial"):
                if getattr(sig, name).n != self.n_secrets_evaluated:
                    raise ValueError(f"significance.{name}.n must equal n_secrets_evaluated")
            used = sig.exact_sign_test
            if used.n_used + used.n_ties + used.n_expected_zero != self.n_secrets_evaluated:
                raise ValueError("sign test counts must add up to n_secrets_evaluated")
        return self


class UtilityMetrics(_Strict):
    # Both keys required; null means "not computed for this run".
    kl_to_base_nonsecret: float | None = Field(ge=0)
    task_score: float | None


class PatchingView(_Strict):
    columns: list[str]
    scores: list[list[float]]  # [n_layers][len(columns)]


class PatchingMetrics(_Strict):
    metric_name: str
    n_layers: int = Field(ge=1)
    layer_types: list[LayerType]  # Gemma 3: every 6th layer global, rest local
    by_component: PatchingView  # columns == COMPONENT_COLUMNS
    by_position: PatchingView | None = None  # columns == POSITION_COLUMNS

    @model_validator(mode="after")
    def _shapes(self) -> PatchingMetrics:
        if len(self.layer_types) != self.n_layers:
            raise ValueError("len(layer_types) must equal n_layers")
        for name, view, cols in (
            ("by_component", self.by_component, COMPONENT_COLUMNS),
            ("by_position", self.by_position, POSITION_COLUMNS),
        ):
            if view is None:
                continue
            if view.columns != cols:
                raise ValueError(f"{name}.columns must be {cols}")
            if len(view.scores) != self.n_layers or any(len(r) != len(cols) for r in view.scores):
                raise ValueError(f"{name}.scores must have shape [{self.n_layers}][{len(cols)}]")
        return self


class PlaygroundMetrics(_Strict):
    """Metrics that are valid for ONE secret (a saved Playground panel).

    Deliberately no attacker accuracy and no bits leaked: those need many secrets and a
    held-out attacker evaluation. extra="forbid" rejects them if anyone tries to add them.
    """

    secret: str = Field(pattern=SECRET_PATTERN)
    n_replies: int = Field(ge=1)
    n_numbers_expected: int = Field(ge=1)  # per reply, from the prompt
    n_numbers_parsed: list[int]  # per reply
    n_replies_mismatched: int = Field(ge=0)  # replies where parsed != expected
    chi2: float = Field(ge=0)
    chi2_df: int = Field(ge=1)
    chi2_p: float = Field(ge=0, le=1)
    digit_kl_to_uniform_bits: float | None = Field(ge=0)
    attacker_name: str
    true_secret_rank: int = Field(ge=1)  # 1-based rank among all secrets in the attacker posterior
    true_secret_prob: float = Field(ge=0, le=1)
    secret_space_size: int = Field(ge=2)

    @model_validator(mode="after")
    def _consistent(self) -> PlaygroundMetrics:
        if len(self.n_numbers_parsed) != self.n_replies:
            raise ValueError("n_numbers_parsed must have one entry per reply")
        mismatched = sum(n != self.n_numbers_expected for n in self.n_numbers_parsed)
        if mismatched != self.n_replies_mismatched:
            raise ValueError("n_replies_mismatched does not match n_numbers_parsed")
        if self.true_secret_rank > self.secret_space_size:
            raise ValueError("true_secret_rank exceeds secret_space_size")
        return self


class RunMetrics(_Strict):
    """metrics.json. Run settings (model, method, k, layers) live in config.yaml, not here."""

    schema_version: Literal[2] = METRICS_SCHEMA_VERSION
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    run_type: RunType  # required, no default
    is_mock: bool  # required, no default
    leakage: LeakageMetrics | None = None
    utility: UtilityMetrics | None = None
    patching: PatchingMetrics | None = None
    playground: PlaygroundMetrics | None = None

    @model_validator(mode="after")
    def _sections(self) -> RunMetrics:
        if self.run_type == "playground":
            if self.playground is None:
                raise ValueError("playground runs need a playground section")
            if self.leakage or self.utility or self.patching:
                raise ValueError(
                    "playground runs must not have leakage (accuracy, bits), utility or patching"
                )
            return self
        if self.playground is not None:
            raise ValueError("only playground runs may have a playground section")
        if self.patching is None and (self.leakage is None or self.utility is None):
            raise ValueError("metrics need either a patching section or both leakage and utility")
        return self


# --------------------------------------------------------------------------- samples.jsonl


class Sample(_Strict):
    """One line of samples.jsonl: one reply to one prompt under one secret."""

    model: str  # model id or method name that produced the reply
    secret: str = Field(pattern=RUN_SECRET_PATTERN)
    prompt_id: str
    prompt: str
    reply_index: int = Field(ge=0)
    reply: str
    numbers: list[int]
    n_numbers_parsed: int = Field(ge=0)  # == len(numbers); compare with eval.n_numbers_expected
    digit_counts: list[int] = Field(min_length=10, max_length=10)
    # Sampled token IDs of the reply (incl. EOS) from the generating model's tokenizer; absent
    # for mock replies and for runs made before IDs were saved.
    reply_token_ids: list[int] | None = None

    @model_validator(mode="after")
    def _parsed_count(self) -> Sample:
        if self.n_numbers_parsed != len(self.numbers):
            raise ValueError("n_numbers_parsed must equal len(numbers)")
        return self


# --------------------------------------------------------------------------- API payloads


class PlaygroundSummary(_Strict):
    """Playground-run fields for the "Saved Playground sessions" list (and Re-open)."""

    session_id: str
    panel: int
    n_panels: int
    secret: str
    prompt: str
    n_samples: int
    seed: int
    temperature: float
    chi2: float
    chi2_p: float
    digit_kl_to_uniform_bits: float | None
    true_secret_rank: int
    true_secret_prob: float
    n_replies_mismatched: int
    attacker_name: str


class RunSummary(_Strict):
    """Row of GET /api/runs (config + metrics flattened for the Results table)."""

    run_id: str
    name: str
    created_at: datetime
    is_mock: bool
    base_model: str  # registry id (config model.id), falling back to the HF id
    hf_id: str  # config model.base
    method: Method
    k: int | None
    layers: list[int] | None
    attacker_exact_match: ChanceMetric | None
    attacker_top5: ChanceMetric | None
    attacker_per_digit: ChanceMetric | None
    bits_leaked_lower_bound: float | None
    kl_upper_bound_bits: float | None
    n_replies_per_secret: int | None
    eval_split: str | None
    attacker_trained_fresh: bool | None
    kl_to_base_nonsecret: float | None
    task_score: float | None
    has_patching: bool
    run_type: RunType
    playground: PlaygroundSummary | None  # playground runs only
    # Added for the Results table (config eval.prompt_set.name, else eval.prompts joined).
    prompt: str | None = None
    prompt_ids: list[str] | None = None


class RunsList(_Strict):
    """GET /api/runs. is_mock is true if any listed run is mock."""

    backend: Literal["mock", "real"]
    is_mock: bool
    runs: list[RunSummary]


class RunDetail(_Strict):
    """GET /api/runs/{run_id}."""

    run_id: str
    created_at: datetime
    is_mock: bool
    config: dict[str, Any]
    metrics: RunMetrics
    n_samples: int


class SamplesPage(_Strict):
    """GET /api/runs/{run_id}/samples?offset&limit."""

    run_id: str
    is_mock: bool
    offset: int
    limit: int
    total: int
    items: list[Sample]


class BaseModelInfo(_Strict):
    """Base-model axis of the Playground (configs/training/models.yaml: base_models)."""

    id: str
    label: str
    hf_id: str
    n_layers: int | None
    optional: bool
    enabled: bool


class MethodInfo(_Strict):
    """Method axis of the Playground (configs/training/models.yaml: methods)."""

    id: Method
    label: str
    enabled: bool


class AttackerInfo(_Strict):
    """The configured attacker (configs/training/app.yaml: attacker), not selectable in the Playground."""

    label: str
    hf_id: str
    adapter: str | None
    in_use: str  # attacker actually used by this backend, e.g. the mock heuristic


class ModelsList(_Strict):
    """GET /api/models. is_mock is true when the Playground generator is the mock."""

    backend: Literal["mock", "real"]
    is_mock: bool
    base_models: list[BaseModelInfo]
    methods: list[MethodInfo]
    attacker: AttackerInfo
    mock_note: str | None  # shown in the UI in mock mode


class Panel(_Strict):
    """One Playground panel: a base model and a method."""

    base_model: str
    method: Method


class GenerateRequest(_Strict):
    """POST /api/generate. All panels share secret, seed and temperature."""

    panels: list[Panel] = Field(min_length=1, max_length=2)
    secret: str | None = Field(default=None, pattern=SECRET_PATTERN)  # null -> random
    prompt: str = Field(min_length=1)
    n_samples: int = Field(ge=1, le=200)
    seed: int | None = None  # null -> server picks one and echoes it back
    temperature: float = Field(default=1.0, gt=0, le=2)


class AttackerGuess(_Strict):
    guess: str
    prob: float = Field(ge=0, le=1)


class AttackerResult(_Strict):
    name: str
    top5: list[AttackerGuess] = Field(max_length=5)
    true_rank: int | None  # 1-based rank of the real secret, null if outside top-5
    true_secret_rank: int  # 1-based rank among all secrets
    true_secret_prob: float = Field(ge=0, le=1)


class GeneratePanelResult(_Strict):
    base_model: str
    method: Method
    replies: list[str]
    n_numbers_expected: int  # per reply, from the prompt
    n_numbers_parsed: list[int]  # per reply
    numbers: list[int]  # all replies, flattened
    digit_counts: list[int] = Field(min_length=10, max_length=10)
    expected_uniform_per_digit: float
    chi2: float
    chi2_p: float = Field(ge=0, le=1)
    attacker: AttackerResult


class GenerateResponse(_Strict):
    generation_id: str  # pass to POST /api/playground/save to keep this result
    secret: str
    prompt: str
    n_samples: int
    seed: int
    temperature: float
    backend: Literal["mock", "real"]
    is_mock: bool
    mock_note: str | None
    results: list[GeneratePanelResult]  # one per panel, in request order


class SaveRequest(_Strict):
    """POST /api/playground/save: keep a recent Generate result as playground runs."""

    generation_id: str = Field(min_length=1)


class SaveResponse(_Strict):
    session_id: str
    run_ids: list[str]  # one per panel, in panel order
    already_saved: bool  # true if this generation had been saved before (nothing new written)
    is_mock: bool


# --------------------------------------------------------------------------- reanalysis.json


class ReanalysisDiagnostics(_Strict):
    expected: str  # how "expected" is defined (leave-one-out)
    pool: str  # which secrets' replies formed the leave-one-out baseline
    rows: list[dict[str, Any]]
    summary: dict[str, float | None]


class Reanalysis(_Strict):
    """reanalysis.json: recomputed from a run's saved replies (no model); originals untouched."""

    reanalysis_version: Literal[1] = 1
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    created_at: datetime
    is_mock: bool
    attacker_source: Literal["refit_on_train_samples", "stored_params"]
    attacker_params: dict[str, float]
    leakage: LeakageMetrics  # includes the significance section
    diagnostics: ReanalysisDiagnostics
    original: dict[str, float | None]  # accuracies / bits from the run's own metrics.json
    matches_original: bool | None  # stored_params only: accuracies reproduce metrics.json
    notes: list[str]


# --------------------------------------------------------------------------- likelihood_attack.json


# What the attacker is assumed to have:
# grey_box: the deployed model itself (base + the run's adapter, or the base model when the run
#   used no adapter); public_weights: only the public base model, without the run's adapter
#   (--scorer-no-adapter); surrogate: a different model altogether (--scorer-model).
ThreatModel = Literal["grey_box", "public_weights", "surrogate"]


class ScorerInfo(_Strict):
    model: str  # registry id or path/HF id as given
    hf_id: str
    same_as_generator: bool  # False = surrogate scorer
    method: str  # whose system prompt the candidate prompts use
    device: str
    dtype: str
    temperature: float = Field(gt=0)
    prompt_cache: bool = True  # False = --no-cache (prompt re-run with every reply)
    adapter: str | None = None  # LoRA adapter on the scorer (the run's own, for fsa/b3/...)
    threat_model: ThreatModel | None = None  # None in files written before it was recorded


def threat_model_of(scorer: ScorerInfo) -> ThreatModel:
    """The recorded threat model, or the one implied by an older file's scorer fields."""
    if scorer.threat_model is not None:
        return scorer.threat_model
    return "grey_box" if scorer.same_as_generator else "surrogate"


class TokenizationReport(_Strict):
    n_replies: int = Field(ge=0)
    n_stored_ids_used: int = Field(ge=0)  # generator's own sampled IDs (same tokenizer)
    n_retokenized: int = Field(ge=0)  # rebuilt from text (older runs, or a surrogate scorer)
    n_roundtrip_exact: int = Field(ge=0)  # replies with decode(encode(text)) == text (all)
    n_stored_match_retokenized: int | None  # stored IDs identical to a fresh tokenisation


class SamplingSettings(_Strict):
    """Next-token distribution settings, at generation time or as the scorer applies them."""

    temperature: float | None
    top_k: int | None  # 0 = no top-k truncation
    top_p: float | None  # 1.0 = no nucleus truncation
    repetition_penalty: float | None  # 1.0 = none
    padding_side: str | None  # tokenizer padding side
    prompt_padding: str | None  # whether the prompt itself was padded (shifts positions)
    dtype: str | None
    source: str  # where these values come from


class SamplingCheck(_Strict):
    generation: SamplingSettings
    scorer: SamplingSettings
    matches: bool  # scorer computes exactly the distribution the replies were sampled from
    mismatches: list[str]
    notes: list[str]


class RankSummary(_Strict):
    """Rank of the true secret among all candidates (1 = top guess), per held-out secret."""

    ranks: list[int]
    histogram: dict[str, int]  # "1", "2-5", "6-10", "11-25", "26-50", "51-100" -> count
    n_first: int = Field(ge=0)
    mean_log2_p_true_first: float | None  # raw posterior, secrets ranked first
    mean_log2_p_true_not_first: float | None
    mean_log2_p_true_first_calibrated: float | None = None
    mean_log2_p_true_not_first_calibrated: float | None = None

    @model_validator(mode="after")
    def _counts(self) -> RankSummary:
        if sum(self.histogram.values()) != len(self.ranks):
            raise ValueError("histogram counts must sum to the number of ranks")
        if self.n_first != sum(r == 1 for r in self.ranks):
            raise ValueError("n_first must count the ranks equal to 1")
        return self


class Calibration(_Strict):
    """One temperature tau fitted on the train secrets; held-out posterior = softmax(s / tau)."""

    tau: float = Field(gt=0)
    tau_at_search_bound: bool
    fitted_on: Literal["train_samples"] = "train_samples"
    train_secrets: list[str]
    train_scores: list[list[float]]  # [train secret][candidate], kept so tau can be refitted
    train_tokenization: TokenizationReport | None
    train_seconds: float = Field(ge=0)
    train_mean_log2_p_raw: float
    train_mean_log2_p_calibrated: float
    leakage: LeakageMetrics  # held-out secrets under the calibrated posterior

    @model_validator(mode="after")
    def _shapes(self) -> Calibration:
        if len(self.train_scores) != len(self.train_secrets):
            raise ValueError("one train score row per train secret")
        return self


class LikelihoodAttack(_Strict):
    """likelihood_attack.json: Bayes attacker scored by teacher forcing on a run's replies.

    Version 2 adds sampling, ranks, calibration and rescored_from (all optional, so version-1
    files still validate).
    """

    likelihood_attack_version: Literal[1, 2] = 2
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    created_at: datetime
    is_mock: bool
    scorer: ScorerInfo
    tokenization: TokenizationReport
    max_batch_tokens: int = Field(ge=1)
    candidates_per_call: int = Field(ge=1)
    seconds_total: float = Field(ge=0)
    secrets: list[str]
    scores: list[list[float]]  # [held-out secret][candidate 00..99], summed log-probs
    leakage: LeakageMetrics  # posterior = softmax(scores); includes significance
    notes: list[str]
    sampling: SamplingCheck | None = None
    ranks: RankSummary | None = None
    calibration: Calibration | None = None  # only with --calibrate
    rescored_from: str | None = None  # earlier result file whose scores were reused
    threat_model: ThreatModel | None = None  # copied from scorer; see ThreatModel

    @model_validator(mode="after")
    def _shapes(self) -> LikelihoodAttack:
        if len(self.scores) != len(self.secrets):
            raise ValueError("one score row per held-out secret")
        if self.ranks is not None and len(self.ranks.ranks) != len(self.secrets):
            raise ValueError("one rank per held-out secret")
        if any(len(row) != self.leakage.secret_space_size for row in self.scores):
            raise ValueError("each score row must cover every candidate secret")
        return self


# --------------------------------------------------------------------------- utility_metrics.json
# scripts/utility_eval.py: outputs/utility/<run_id>/{config.yaml, items.jsonl, utility_metrics.json}.
# Separate from evaluation runs (outputs/runs); nothing here changes metrics.json.

# pe_access (digits4 runs only): compliant reveal rate and withhold leak rate under the
# PasswordEval-style pe_base prompt; like verification, only with a secret.
UtilityBenchmark = Literal["mmlu", "ifeval", "verification", "kl_to_base", "pe_access"]
# secret: the method's system template with a held-out secret; no_secret: the no_secret template.
UtilityCondition = Literal["secret", "no_secret"]
UTILITY_METRIC_KEYS: dict[str, set[str]] = {
    "mmlu": {"accuracy"},
    "verification": {"accuracy"},
    "ifeval": {"prompt_level_strict", "instruction_level_strict"},
    "kl_to_base": {"kl_per_token"},
    "pe_access": {"compliant_reveal_rate", "withhold_leak_rate"},
}
SECRET_ONLY_BENCHMARKS = ("verification", "pe_access")


class EstimateCI(_Strict):
    """A point estimate with a percentile bootstrap CI over items (n_resamples 0 = exact)."""

    estimate: float
    low: float | None
    high: float | None
    level: float = 0.95
    n_items: int = Field(ge=1)  # resampling units: questions, prompts or verification items
    n_resamples: int = Field(ge=0)
    seed: int

    @model_validator(mode="after")
    def _ordered(self) -> EstimateCI:
        if self.low is not None and self.high is not None and self.low > self.high + 1e-12:
            raise ValueError("CI low must be <= high")
        return self


class UtilityConditionResult(_Strict):
    n_items: int = Field(ge=1)
    metrics: dict[str, EstimateCI]
    diagnostics: dict[str, float] = Field(default_factory=dict)  # e.g. mean answer-token mass
    note: str | None = None


class UtilityBenchmarkResult(_Strict):
    conditions: dict[UtilityCondition, UtilityConditionResult]
    # Paired over the same items (same secret assignment): secret minus no_secret.
    secret_minus_no_secret: dict[str, EstimateCI] = Field(default_factory=dict)


class UtilityItem(_Strict):
    """One line of items.jsonl: one (benchmark, condition, item)."""

    benchmark: UtilityBenchmark
    condition: UtilityCondition
    item_id: str = Field(min_length=1)
    secret: str | None = Field(default=None, pattern=RUN_SECRET_PATTERN)  # None: no_secret
    gold: str | None = None  # MMLU letter; verification "yes"/"no"
    prediction: str | None = None
    correct: bool | None = None  # MMLU, verification; IFEval: all instructions followed
    logprobs: dict[str, float] | None = None  # answer tokens -> next-token log-prob
    answer_mass: float | None = Field(default=None, ge=0, le=1.0 + 1e-6)  # P(answer tokens)
    instruction_id_list: list[str] | None = None  # IFEval
    follow_instruction_list: list[bool] | None = None  # IFEval, strict
    response: str | None = None  # IFEval and general-prompt replies (greedy)
    n_reply_tokens: int | None = Field(default=None, ge=0)
    kl_sum: float | None = Field(default=None, ge=0)  # kl_to_base: sum of per-token KL (nats)
    n_tokens: int | None = Field(default=None, ge=0)  # kl_to_base: reply tokens scored


class UtilityEvalMetrics(_Strict):
    """utility_metrics.json."""

    utility_eval_version: Literal[1] = 1
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    created_at: datetime
    is_mock: bool
    model: str  # registry id
    hf_id: str
    method: str
    adapter: str | None
    adapter_run: str | None  # training run that produced the adapter
    benchmarks: dict[UtilityBenchmark, UtilityBenchmarkResult]
    seconds_total: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> UtilityEvalMetrics:
        for name, result in self.benchmarks.items():
            allowed = UTILITY_METRIC_KEYS[name]
            for cond, res in result.conditions.items():
                if set(res.metrics) != allowed:
                    raise ValueError(f"{name}/{cond}: metrics must be {sorted(allowed)}")
            if set(result.secret_minus_no_secret) - allowed:
                raise ValueError(f"{name}: unknown paired-difference metric")
        for name in SECRET_ONLY_BENCHMARKS:
            res = self.benchmarks.get(name)
            if res is not None and set(res.conditions) != {"secret"}:
                raise ValueError(f"{name} runs only with a secret in the system prompt")
        kl = self.benchmarks.get("kl_to_base")
        if (
            kl is not None
            and self.adapter is None
            and any(r.metrics["kl_per_token"].estimate != 0 for r in kl.conditions.values())
        ):
            raise ValueError("kl_to_base must be 0 for a model without an adapter")
        return self


# --------------------------------------------------------------------------- dashboard extras
# GET /api/likelihood-attacks and GET /api/utility (read-only summaries of the files above).

PosteriorKind = Literal["calibrated", "raw"]


class LikelihoodAttackSummary(_Strict):
    """One likelihood_attack*.json as the Results table shows it. bits, CI, exact, top5 and
    perm_p come from the calibrated posterior when the file has a usable one (posterior
    "calibrated"), otherwise from the raw one: no calibration, or a calibration whose tau hit its
    search bound (tau_at_search_bound true: degenerate, shown as raw†). bits_raw is always the
    raw posterior's lower bound."""

    file: str
    version: Literal[1, 2]
    created_at: datetime
    is_mock: bool
    threat_model: ThreatModel
    threat_model_inferred: bool  # not recorded in the file; derived from its scorer fields
    posterior: PosteriorKind
    bits: float = Field(ge=0)
    bits_ci_low: float | None
    bits_ci_high: float | None
    bits_raw: float = Field(ge=0)
    tau: float | None
    exact: float = Field(ge=0, le=1)
    top5: float = Field(ge=0, le=1)
    perm_p: float | None
    n_heldout: int = Field(ge=1)
    duplicate_of: str | None  # same secrets and scores as this earlier file of the run
    tau_at_search_bound: bool | None = None  # None: the file has no calibration


class RunAttacks(_Strict):
    run_id: str
    attacks: list[LikelihoodAttackSummary]  # every valid file, duplicates included


class LikelihoodAttacksList(_Strict):
    """GET /api/likelihood-attacks: runs listed by /api/runs that have valid attack files."""

    backend: Literal["mock", "real"]
    is_mock: bool
    runs: list[RunAttacks]
    warnings: list[str]  # files skipped as unreadable or invalid


class UtilityRunSummary(_Strict):
    """One outputs/utility/<run_id>/ folder (utility_metrics.json + config.yaml)."""

    run_id: str
    created_at: datetime
    is_mock: bool
    model: str
    hf_id: str
    method: str
    adapter_run: str | None
    seed: int | None
    n_mmlu: int | None
    n_ifeval: int | None
    benchmarks: dict[UtilityBenchmark, UtilityBenchmarkResult]


class UtilityRunsList(_Strict):
    """GET /api/utility. The mock backend lists none (no mock utility data exists)."""

    backend: Literal["mock", "real"]
    is_mock: bool
    runs: list[UtilityRunSummary]
    warnings: list[str]  # folders skipped as unreadable or invalid
    note: str | None = None
