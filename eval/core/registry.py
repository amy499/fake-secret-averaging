"""Name -> class registry for every pluggable slot.

Status: ✅ implemented + tested (tests/test_registry.py).

    @register("defence", "my_defence")
    class MyDefence(Defence): ...

    cls = get("defence", "my_defence")

`discover()` imports every module that registers something:
  * all modules in models/, secret_types/, benchmarks/, attackers/, metrics/
  * experiments/<name>/defence.py for every approach folder (folders starting with "_", such as
    experiments/_template, are skipped)
  * every module in experiments/_bridges/ and in the repository's baselines/
Heavy dependencies (torch, transformers, peft, pyrit) must be imported lazily inside methods,
never at module import time, so discovery works on a CPU-only laptop.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

KINDS = ("model", "secret", "benchmark", "defence", "attacker", "metric")
ROOT = Path(__file__).resolve().parent.parent
_REGISTRY: dict[str, dict[str, Any]] = {k: {} for k in KINDS}
_PACKAGES = {
    "model": "models",
    "secret": "secret_types",
    "benchmark": "benchmarks",
    "attacker": "attackers",
    "metric": "metrics",
}
T = TypeVar("T")
_discovered = False


def register(kind: str, name: str) -> Callable[[T], T]:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown kind {kind!r}; kinds are {KINDS}")

    def deco(obj: T) -> T:
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing is not obj:
            same = getattr(existing, "__qualname__", None) == getattr(obj, "__qualname__", None)
            if not same:  # re-import of the same module is fine; a real clash is not
                raise ValueError(f"{kind} {name!r} already registered by {existing!r}")
        _REGISTRY[kind][name] = obj
        return obj

    return deco


def discover() -> None:
    global _discovered
    if _discovered:
        return
    for pkg in _PACKAGES.values():
        _import_package(pkg)
    from core.io import REPO

    for mod in sorted((REPO / "baselines").glob("*.py")):  # none, prompt_instruction, output_filter
        if mod.stem != "__init__":
            importlib.import_module(f"baselines.{mod.stem}")
    exp = ROOT / "experiments"
    if exp.is_dir():
        for sub in sorted(p for p in exp.iterdir() if p.is_dir()):
            if sub.name == "_bridges":
                for mod in sorted(sub.glob("*.py")):
                    if mod.stem != "__init__":
                        importlib.import_module(f"experiments.{sub.name}.{mod.stem}")
            elif not sub.name.startswith("_") and (sub / "defence.py").exists():  # skips _template
                importlib.import_module(f"experiments.{sub.name}.defence")
    _discovered = True


def _import_package(pkg: str) -> None:
    module = importlib.import_module(pkg)
    for info in pkgutil.iter_modules(module.__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{pkg}.{info.name}")


def get(kind: str, name: str) -> Any:
    discover()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "(none)"
        raise KeyError(f"no {kind} named {name!r}. Registered: {known}") from None


def names(kind: str) -> list[str]:
    discover()
    return sorted(_REGISTRY[kind])


def all_entries() -> dict[str, dict[str, Any]]:
    discover()
    return {k: dict(v) for k, v in _REGISTRY.items()}
