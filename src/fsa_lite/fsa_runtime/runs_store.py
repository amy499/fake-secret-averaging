"""Read access to runs behind a small interface (list_runs, get_run, get_samples).

FolderRunStore reads run folders directly; they are the single source of truth. A later
SQLite index can implement the same RunStore protocol without changing the API layer.
Folders without metrics.json (still running) or that fail validation are skipped from the
listing with a warning, so one broken run cannot take the dashboard down. list_attacks() reads
each listed run's likelihood_attack*.json files (fsa.eval.run_extras, shared with
scripts/collect_results.py); an invalid file becomes a warning string, never an error.
"""

from __future__ import annotations

import logging
import re
from itertools import islice
from pathlib import Path
from typing import Protocol

import yaml

from fsa.contract.schemas import (
    RUN_ID_PATTERN,
    PlaygroundSummary,
    RunAttacks,
    RunConfig,
    RunDetail,
    RunMetrics,
    RunSummary,
    Sample,
    SamplesPage,
)
from fsa.eval.run_extras import prompt_ids, prompt_label, read_attack_files, summarize_attack
from fsa.io.run_writer import CONFIG_FILE, METRICS_FILE, SAMPLES_FILE, parse_run_timestamp

log = logging.getLogger(__name__)

_RUN_ID_RE = re.compile(RUN_ID_PATTERN)


class RunNotFound(KeyError):
    """Unknown, malformed, incomplete or invalid run id."""


class RunStore(Protocol):
    def list_runs(self) -> list[RunSummary]: ...

    def get_run(self, run_id: str) -> RunDetail: ...

    def get_samples(self, run_id: str, offset: int = 0, limit: int = 100) -> SamplesPage: ...

    def list_attacks(self) -> tuple[list[RunAttacks], list[str]]: ...


def _playground_summary(config: RunConfig, metrics: RunMetrics) -> PlaygroundSummary | None:
    pc, pm = config.playground, metrics.playground
    if pc is None or pm is None:
        return None
    req = pc.request
    return PlaygroundSummary(
        session_id=pc.session_id,
        panel=pc.panel,
        n_panels=pc.n_panels,
        secret=req.secret,
        prompt=req.prompt,
        n_samples=req.n_samples,
        seed=req.seed,
        temperature=req.temperature,
        chi2=pm.chi2,
        chi2_p=pm.chi2_p,
        digit_kl_to_uniform_bits=pm.digit_kl_to_uniform_bits,
        true_secret_rank=pm.true_secret_rank,
        true_secret_prob=pm.true_secret_prob,
        n_replies_mismatched=pm.n_replies_mismatched,
        attacker_name=pm.attacker_name,
    )


def _summary(
    run_id: str, config: RunConfig, metrics: RunMetrics, raw_config: dict | None = None
) -> RunSummary:
    leak, util = metrics.leakage, metrics.utility
    if config.run_type != metrics.run_type:
        raise ValueError(
            f"run_type differs: config {config.run_type!r}, metrics {metrics.run_type!r}"
        )
    return RunSummary(
        run_type=metrics.run_type,
        playground=_playground_summary(config, metrics),
        run_id=run_id,
        name=config.name,
        created_at=parse_run_timestamp(run_id),
        is_mock=metrics.is_mock,
        base_model=config.model.id or config.model.base,
        hf_id=config.model.base,
        method=config.method,
        k=config.fsa.k if config.fsa else None,
        layers=config.lora.layers if config.lora else None,
        attacker_exact_match=leak.attacker.exact_match if leak else None,
        attacker_top5=leak.attacker.top5 if leak else None,
        attacker_per_digit=leak.attacker.per_digit if leak else None,
        bits_leaked_lower_bound=leak.bits_leaked_lower_bound if leak else None,
        kl_upper_bound_bits=leak.kl_upper_bound_bits if leak else None,
        n_replies_per_secret=leak.n_replies_per_secret if leak else None,
        eval_split=leak.eval_split if leak else None,
        attacker_trained_fresh=leak.attacker_trained_fresh if leak else None,
        kl_to_base_nonsecret=util.kl_to_base_nonsecret if util else None,
        task_score=util.task_score if util else None,
        has_patching=metrics.patching is not None,
        prompt=prompt_label(raw_config or {}),
        prompt_ids=prompt_ids(raw_config or {}),
    )


class FolderRunStore:
    """RunStore reading <runs_dir>/<run_id>/{config.yaml, metrics.json, samples.jsonl}."""

    def __init__(self, runs_dir: Path | str) -> None:
        self.runs_dir = Path(runs_dir)

    # -- helpers ---------------------------------------------------------------------------

    def _run_path(self, run_id: str) -> Path:
        # run_id comes from the URL: allow only well-formed ids that resolve inside runs_dir.
        if not _RUN_ID_RE.fullmatch(run_id):
            raise RunNotFound(run_id)
        path = self.runs_dir / run_id
        if path.resolve().parent != self.runs_dir.resolve() or not path.is_dir():
            raise RunNotFound(run_id)
        return path

    def _load(self, path: Path) -> tuple[dict, RunConfig, RunMetrics]:
        raw_config = yaml.safe_load((path / CONFIG_FILE).read_text(encoding="utf-8"))
        config = RunConfig.model_validate(raw_config)
        metrics = RunMetrics.model_validate_json((path / METRICS_FILE).read_text(encoding="utf-8"))
        if metrics.run_id != path.name:
            raise ValueError(f"metrics.run_id {metrics.run_id!r} != folder name")
        return raw_config, config, metrics

    def _load_or_not_found(self, run_id: str) -> tuple[Path, dict, RunConfig, RunMetrics]:
        path = self._run_path(run_id)
        try:
            return (path, *self._load(path))
        except (OSError, ValueError, yaml.YAMLError) as e:  # pydantic errors are ValueErrors
            raise RunNotFound(run_id) from e

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    # -- RunStore --------------------------------------------------------------------------

    def list_runs(self) -> list[RunSummary]:
        """All complete, valid runs, newest first."""
        if not self.runs_dir.is_dir():
            return []
        summaries = []
        for path in self.runs_dir.iterdir():
            if not (path.is_dir() and _RUN_ID_RE.fullmatch(path.name)):
                continue
            if not (path / METRICS_FILE).exists():
                continue  # still running
            try:
                raw_config, config, metrics = self._load(path)
            except (OSError, ValueError, yaml.YAMLError) as e:
                log.warning("skipping invalid run %s: %s", path.name, e)
                continue
            summaries.append(_summary(path.name, config, metrics, raw_config))
        summaries.sort(key=lambda s: (s.created_at, s.run_id), reverse=True)
        return summaries

    def get_run(self, run_id: str) -> RunDetail:
        path, raw_config, _, metrics = self._load_or_not_found(run_id)
        return RunDetail(
            run_id=run_id,
            created_at=parse_run_timestamp(run_id),
            is_mock=metrics.is_mock,
            config=raw_config,
            metrics=metrics,
            n_samples=self._count_lines(path / SAMPLES_FILE),
        )

    def get_samples(self, run_id: str, offset: int = 0, limit: int = 100) -> SamplesPage:
        if offset < 0 or limit < 1:
            raise ValueError("offset must be >= 0 and limit >= 1")
        path, _, _, metrics = self._load_or_not_found(run_id)
        samples_path = path / SAMPLES_FILE
        items: list[Sample] = []
        if samples_path.exists():
            with samples_path.open(encoding="utf-8") as f:
                lines = (line for line in f if line.strip())
                items = [
                    Sample.model_validate_json(x) for x in islice(lines, offset, offset + limit)
                ]
        return SamplesPage(
            run_id=run_id,
            is_mock=metrics.is_mock,
            offset=offset,
            limit=limit,
            total=self._count_lines(samples_path),
            items=items,
        )

    def list_attacks(self) -> tuple[list[RunAttacks], list[str]]:
        """(runs, warnings): likelihood-attack summaries of every listed evaluation run that has
        valid attack files, newest run first; warnings name each file skipped and why."""
        runs: list[RunAttacks] = []
        warnings: list[str] = []
        for s in self.list_runs():
            if s.run_type != "evaluation":
                continue
            path = self.runs_dir / s.run_id
            files = read_attack_files(path)
            if not files:
                continue
            try:
                raw_config = yaml.safe_load((path / CONFIG_FILE).read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as e:  # listed a moment ago; changed since
                warnings.append(f"{s.run_id}: config.yaml unreadable: {e}")
                continue
            attacks = []
            for f in files:
                if f.attack is None:
                    warnings.append(f"{s.run_id}/{f.path.name}: skipped ({f.error})")
                elif f.attack.run_id != s.run_id:
                    warnings.append(
                        f"{s.run_id}/{f.path.name}: skipped (run_id {f.attack.run_id!r} "
                        "does not match the folder)"
                    )
                else:
                    attacks.append(summarize_attack(f, raw_config or {}))
            if attacks:
                runs.append(RunAttacks(run_id=s.run_id, attacks=attacks))
        return runs, warnings
