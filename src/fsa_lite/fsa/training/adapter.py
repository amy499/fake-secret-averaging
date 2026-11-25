"""Trained adapters in evaluation runs: what an adapter was trained on, and is it safe to use.

Every saved adapter folder holds training_info.json (written by fsa.training.trainer). Before
generating with an adapter, an evaluation checks four things: same base model, same method,
none of its held-out secrets were trained on, and the system template it will use was a
training one. Paths go into config.yaml relative to the experiment folder when possible, so
a run copied between the laptop and the Slurm cluster still finds its adapter.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

TRAINING_INFO = "training_info.json"
ADAPTER_METHODS = ("fsa", "b3", "random_layer")


def read_training_info(adapter: str | Path) -> dict[str, Any] | None:
    path = Path(adapter) / TRAINING_INFO
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check_adapter(
    info: dict[str, Any],
    *,
    hf_id: str,
    method: str,
    heldout: Sequence[str],
    secret_type: str = "digits",
    secret_list_sha256: str | None = None,
) -> tuple[list[str], list[str]]:
    """(errors, warnings) for evaluating `method` on `heldout` secrets with this adapter.

    The adapter's secret type (training_info secret_type; adapters from before word secrets
    have none and are digit adapters) must match the evaluation's, and a word adapter must be
    evaluated on the same word list (same hash)."""
    errors, warnings = [], []
    trained_type = info.get("secret_type") or "digits"
    if trained_type != secret_type:
        errors.append(
            f"adapter was trained on {trained_type} secrets, evaluated on {secret_type} secrets"
        )
    elif (
        secret_type in ("words", "digits4") and info.get("secret_list_sha256") != secret_list_sha256
    ):
        errors.append(
            f"adapter was trained on a different "
            f"{'word' if secret_type == 'words' else 'digits4 code'} list (sha256 "
            f"{str(info.get('secret_list_sha256'))[:12]}..., evaluated with "
            f"{str(secret_list_sha256)[:12]}...)"
        )
    if info.get("hf_id") and info["hf_id"] != hf_id:
        errors.append(f"adapter was trained on {info['hf_id']}, not {hf_id}")
    if info.get("method") and info["method"] != method:
        errors.append(f"adapter was trained as {info['method']!r}, evaluated as {method!r}")
    leaked = sorted(set(heldout) & set(info.get("train_secrets") or []))
    if leaked:
        errors.append(
            f"{len(leaked)} held-out secrets were training secrets of this adapter ({leaked[:5]}"
            "...): use the training split (same --seed, --n-train >= its n_train)"
        )
    systems = info.get("train_system_methods") or ["base"]
    # digits4: pe_base is the same text as that space's base template (configs/training/prompts.yaml).
    if systems != ["base"] and not (secret_type == "digits4" and systems == ["pe_base"]):
        warnings.append(
            f"adapter trained with system templates {systems}; evaluation of {method!r} uses the "
            "base template"
        )
    return errors, warnings


def portable_path(path: str | Path, root: Path) -> str:
    """Relative to `root` when inside it (POSIX separators), otherwise absolute."""
    p = Path(path).resolve()
    try:
        return p.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(p)


def resolve_path(path: str | Path, root: Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else (root / p).resolve())
