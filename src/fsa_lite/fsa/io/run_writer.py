"""Writes one run folder: outputs/runs/<YYYYMMDD-HHMMSS>_<name>/{config.yaml, metrics.json, samples.jsonl}.

Every write is validated against fsa.contract.schemas, so a run that the writer accepts is a run
the dashboard can read. config.yaml is written as soon as the run starts (a crashed run still
records what it was), samples are appended as they are produced, and metrics.json is written
last and atomically; its presence marks the run as complete. All files are UTF-8 with LF endings.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fsa.contract.schemas import RunConfig, RunMetrics, Sample

CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.json"
SAMPLES_FILE = "samples.jsonl"
# Optional: replies of the attacker-train secrets (lets scripts/reanalyze_run.py refit the attacker).
TRAIN_SAMPLES_FILE = "train_samples.jsonl"

_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def sanitize_name(name: str) -> str:
    """Make a run name safe for folder names and run ids: [A-Za-z0-9_-] only."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    if not cleaned:
        raise ValueError(f"run name {name!r} has no usable characters")
    return cleaned


def make_run_id(name: str, now: datetime | None = None) -> str:
    return f"{(now or datetime.now()).strftime(_TIMESTAMP_FORMAT)}_{sanitize_name(name)}"


def parse_run_timestamp(run_id: str) -> datetime:
    return datetime.strptime(run_id.split("_", 1)[0], _TIMESTAMP_FORMAT)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


class RunWriter:
    """Create a run folder and write its files.

    Usage:
        run = RunWriter(runs_dir, config_dict)
        run.append_samples([...])
        run.write_metrics({...})        # run_id is filled in if missing
    """

    def __init__(
        self, runs_dir: Path | str, config: Mapping[str, Any], *, now: datetime | None = None
    ) -> None:
        self.config = RunConfig.model_validate(dict(config))
        # Dump the caller's dict, not the model, to keep key order; safe_dump rejects non-plain types.
        config_text = yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
        self.run_id = make_run_id(self.config.name, now)
        self.path = Path(runs_dir) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)  # never overwrite an existing run
        _write_text_atomic(self.path / CONFIG_FILE, config_text)

    def append_samples(
        self, samples: Iterable[Sample | Mapping[str, Any]], filename: str = SAMPLES_FILE
    ) -> int:
        """Validate and append samples (default samples.jsonl); returns how many were written."""
        if filename not in (SAMPLES_FILE, TRAIN_SAMPLES_FILE):
            raise ValueError(f"unknown samples file {filename!r}")
        lines = [
            (s if isinstance(s, Sample) else Sample.model_validate(dict(s))).model_dump_json()
            for s in samples
        ]
        with (self.path / filename).open("a", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")
        return len(lines)

    def write_metrics(self, metrics: RunMetrics | Mapping[str, Any]) -> RunMetrics:
        """Validate and write metrics.json (marks the run complete). run_id and run_type default
        to the folder and config values and must match them."""
        data = metrics.model_dump() if isinstance(metrics, RunMetrics) else dict(metrics)
        data.setdefault("run_id", self.run_id)
        data.setdefault("run_type", self.config.run_type)
        validated = RunMetrics.model_validate(data)
        if validated.run_id != self.run_id:
            raise ValueError(f"metrics run_id {validated.run_id!r} != folder {self.run_id!r}")
        if validated.run_type != self.config.run_type:
            raise ValueError(
                f"metrics run_type {validated.run_type!r} != config run_type {self.config.run_type!r}"
            )
        is_patching_run = self.config.method == "patching"
        if is_patching_run != (validated.patching is not None):
            raise ValueError(
                "patching runs (method: patching) must have a patching section; other runs must not"
            )
        _write_text_atomic(
            self.path / METRICS_FILE,
            json.dumps(validated.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        )
        return validated
