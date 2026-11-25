"""Where the LoRA adapter goes: which decoder layers, which modules.

`lora.layers` is "all", an explicit list, or "random:N": N distinct layers drawn with the run's
seed (the random-layer control), so the same seed always picks the same layers. The layers
and modules become one regex for PEFT's target_modules. It matches only the language model's
decoder layers: Gemma-3's vision tower also has q_proj/k_proj/... and must stay untouched.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np


def n_decoder_layers(model: Any) -> int:
    cfg = getattr(model, "config", model)
    text_cfg = getattr(cfg, "text_config", None) or cfg  # Gemma-3 nests the LM config
    return int(text_cfg.num_hidden_layers)


def resolve_layers(spec: str | list[int], n_layers: int, seed: int) -> list[int]:
    """Concrete sorted layer indices for "all" | [list] | "random:N"."""
    if spec == "all":
        return list(range(n_layers))
    if isinstance(spec, str):
        n = int(spec.split(":", 1)[1])
        if n > n_layers:
            raise ValueError(f"random:{n} asks for more layers than the model has ({n_layers})")
        # Own stream ([seed, 31]), so the choice does not shift if other random draws change.
        return sorted(int(i) for i in np.random.default_rng([seed, 31]).choice(n_layers, n, False))
    bad = [i for i in spec if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"layers {bad} out of range for a {n_layers}-layer model")
    return sorted(spec)


def target_regex(layers: Sequence[int], modules: Sequence[str]) -> str:
    """PEFT target_modules regex (full match) for these decoder layers and module names."""
    layer_alt = "|".join(str(i) for i in layers)
    mod_alt = "|".join(re.escape(m) for m in modules)
    return rf"^(?!.*vision).*\.layers\.({layer_alt})\.(self_attn|mlp)\.({mod_alt})$"


def adapted_layers(model: Any) -> set[int]:
    """Decoder-layer indices that actually carry LoRA weights (for checks and tests)."""
    found = set()
    for name, _ in model.named_parameters():
        m = re.search(r"\.layers\.(\d+)\..*lora_", name)
        if m and "vision" not in name:
            found.add(int(m.group(1)))
    return found
