"""Playground registry from configs/training/models.yaml: two independent axes, base models x methods.

A Playground panel picks one base model and one method; `Registry.spec` validates the pair
(both must exist and be enabled) and turns it into a ModelSpec for the generator. Disabled
entries are still listed so the UI can grey them out. The attacker is not part of the registry;
it is a separate setting (configs/training/app.yaml: attacker). Trained LoRA adapters live under
base_models[].adapters as {method: path}; the path is resolved against the experiment folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fsa_runtime.generators.base import ModelSpec
from fsa.contract.schemas import BaseModelInfo, Method, MethodInfo

ROOT = Path(__file__).resolve().parents[3]
MODELS_YAML = ROOT / "configs" / "training" / "models.yaml"


class RegistryError(ValueError):
    """Unknown or disabled base model / method."""


class BaseModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    hf_id: str
    n_layers: int | None = Field(default=None, ge=1)
    optional: bool = False
    enabled: bool = True
    adapters: dict[Method, str] = Field(default_factory=dict)  # method -> LoRA adapter path

    def info(self) -> BaseModelInfo:
        return BaseModelInfo(**self.model_dump(exclude={"adapters"}))

    def adapter_for(self, method: str) -> str | None:
        path = self.adapters.get(method)
        if not path:
            return None
        p = Path(path)
        return str(p if p.is_absolute() else (ROOT / p).resolve())


class MethodEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Method
    label: str
    enabled: bool = True
    k: int | None = Field(default=None, ge=1)  # fsa only; mock falls back to playground_fsa_k

    def info(self) -> MethodInfo:
        return MethodInfo(id=self.id, label=self.label, enabled=self.enabled)


def _unique(entries: list, kind: str) -> dict:
    by_id = {e.id: e for e in entries}
    if len(by_id) != len(entries):
        raise ValueError(f"duplicate {kind} ids in models.yaml")
    return by_id


@dataclass(frozen=True)
class Registry:
    base_models: dict[str, BaseModelEntry]
    methods: dict[str, MethodEntry]

    def spec(self, base_model: str, method: str) -> ModelSpec:
        bm = self.base_models.get(base_model)
        if bm is None:
            raise RegistryError(f"unknown base model {base_model!r}")
        if not bm.enabled:
            raise RegistryError(f"base model {base_model!r} is disabled")
        m = self.methods.get(method)
        if m is None:
            raise RegistryError(f"unknown method {method!r}")
        if not m.enabled:
            raise RegistryError(f"method {method!r} is disabled")
        return ModelSpec(
            method=m.id, base_model=bm.id, hf_id=bm.hf_id, k=m.k, adapter=bm.adapter_for(m.id)
        )


def load_registry(path: Path = MODELS_YAML) -> Registry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Registry(
        base_models=_unique(
            [BaseModelEntry.model_validate(b) for b in raw["base_models"]], "base model"
        ),
        methods=_unique([MethodEntry.model_validate(m) for m in raw["methods"]], "method"),
    )
