"""The evaluation pipeline behind `python run.py --config <yaml>`.

Status: ✅ implemented + tested (tests/test_dry_run.py runs it end to end for every benchmark).

Steps
 1. Resolve every slot from the registry (secret type, benchmark, model, defence).
 2. Items for the configured split; defence.prepare(items) and optional defence.train().
 3. Replies: one greedy scored reply (sample = -1) per item and k sampled leak-test replies
    (temperature 1, seed = 2026e6 + secret_index*100 + j). With controls.no_secret, the same
    k sampled replies again with the secret replaced by "[hidden]" (control = True).
 4. Attackers: each returns one Guess per item; passive attackers (reading replies only) are
    also run on the control replies -> control guesses (their accuracy is the empirical chance).
 5. Metrics -> metrics.json (summaries) + per_item.json (per-item values for paired stats).
 6. Outputs in outputs/<timestamp>_<config-name>/: config.yaml, replies.jsonl, guesses.jsonl,
    metrics.json, per_item.json, git_commit.txt.

Held-out guard (optional config key `held_out: {approach: <name>}`; held_out_guard below): for an
approach trained on synthetic secrets, the run refuses before the model loads if any evaluated
code is in the approach's private used-secrets file (approach spec `synthetic.used_secrets`, the
file core/protocol.py's held_out_filter reads). Records only the file's sha256 and counts.
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core import registry
from core.config import ExperimentConfig, reply_seed
from core.interfaces import Dataset, GenConfig, Guess, Item, ModelWrapper, Reply, SecretSource, secret_contexts
from core.io import ROOT, git_commit, make_run_dir, write_json, write_jsonl

DRY_RUN_LIMIT = 6
DRY_RUN_CAPS = {"K": 12, "n_max": 8, "budget": 6, "turns": 3, "n_train": 24}


@dataclass
class RunContext:
    cfg: ExperimentConfig
    benchmark: Dataset
    secrets: SecretSource
    model: ModelWrapper  # defended model (what attackers query)
    base_model: ModelWrapper  # undefended model (public-weights threat model)
    split: str
    dry_run: bool = False
    out_dir: Path | None = None
    secret_index: dict[str, int] = field(default_factory=dict)
    baseline_dir: Path | None = None
    _train_items: list[Item] | None = None

    @property
    def scored_cfg(self) -> GenConfig:
        d = self.cfg.decoding.scored
        return GenConfig(max_new_tokens=int(d.get("max_new_tokens", 256)),
                         temperature=float(d.get("temperature", 0.0)))  # fmt: skip

    @property
    def leak_cfg(self) -> GenConfig:
        d = self.cfg.decoding.leak
        return GenConfig(max_new_tokens=int(d.get("max_new_tokens", 256)),
                         temperature=float(d.get("temperature", 1.0)), top_p=float(d.get("top_p", 1.0)))  # fmt: skip

    def leaks(self, item: Item, text: str) -> bool:
        fn = getattr(self.benchmark, "leaks", None)
        if fn is not None:
            return bool(fn(item, text))
        return bool(item.secret) and self.secrets.leaks(text, item.secret)  # type: ignore[arg-type]

    def train_items(self) -> list[Item]:
        if self._train_items is None:
            try:
                items = self.benchmark.items("train")
            except (KeyError, ValueError):
                items = []
            if self.dry_run:
                items = items[: DRY_RUN_CAPS["n_train"]]
            self._train_items = items
        return self._train_items

    def cap(self, key: str, value: int) -> int:
        return min(value, DRY_RUN_CAPS[key]) if self.dry_run and key in DRY_RUN_CAPS else value

    def seed_for(self, item: Item, sample: int) -> int:
        return reply_seed(self.secret_index.get(item.id, 0), sample)


def apply_dry_run(cfg: ExperimentConfig) -> ExperimentConfig:
    cfg = copy.deepcopy(cfg)
    thinking = bool(cfg.model.get("thinking", False)) or cfg.model["name"].endswith("-think")
    cfg.model = {"name": "dummy", "thinking": thinking, "replaces": cfg.model["name"]}
    cfg.benchmark["limit"] = min(int(cfg.benchmark.get("limit") or DRY_RUN_LIMIT), DRY_RUN_LIMIT)
    cfg.decoding.scored["max_new_tokens"] = min(int(cfg.decoding.scored.get("max_new_tokens", 256)), 64)
    cfg.decoding.leak["max_new_tokens"] = min(int(cfg.decoding.leak.get("max_new_tokens", 256)), 64)
    cfg.budget = min(cfg.budget, DRY_RUN_CAPS["budget"])
    return cfg


def generate_replies(ctx: RunContext, items: Sequence[Item]) -> list[Reply]:
    model, k = ctx.model, ctx.cfg.decoding.k
    replies: list[Reply] = []

    def run(batch: list[tuple[Item, int, bool]], gen_cfg: GenConfig) -> None:
        prompts = [(it.with_secret(None) if ctrl else it).messages() for it, _, ctrl in batch]
        seeds = [None if j < 0 else ctx.seed_for(it, j) for it, j, _ in batch]
        spans = secret_contexts(prompts, [None if ctrl else it.secret for it, _, ctrl in batch])
        gens = model.generate(prompts, gen_cfg, seeds, secret_ctx=spans)
        for (it, j, ctrl), s, g in zip(batch, seeds, gens):
            replies.append(Reply(item_id=it.id, sample=j, seed=s, text=g.text, reasoning=g.reasoning,
                                 latency_s=g.latency_s, n_tokens=g.n_tokens, control=ctrl,
                                 finished=g.finished))  # fmt: skip

    run([(it, -1, False) for it in items], ctx.scored_cfg)
    if k > 0:
        run([(it, j, False) for it in items for j in range(k)], ctx.leak_cfg)
        if ctx.cfg.controls.get("no_secret", True):
            with_secret = [it for it in items if it.secret]
            run([(it, j, True) for it in with_secret for j in range(k)], ctx.leak_cfg)
    return replies


def find_baseline(cfg: ExperimentConfig, run_dir: Path) -> Path | None:
    """`baseline_run: auto` -> newest run dir with the same benchmark/model and defence none."""
    if not cfg.baseline_run:
        return None
    if cfg.baseline_run != "auto":
        p = Path(cfg.baseline_run)
        return p if p.is_absolute() else ROOT / p
    best = None
    for d in sorted(run_dir.parent.iterdir()):
        if d == run_dir or not (d / "config.yaml").exists() or not (d / "per_item.json").exists():
            continue
        c = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {}
        if (
            (c.get("defence") or {}).get("name") == "none"
            and (c.get("benchmark") or {}).get("name") == cfg.benchmark["name"]
            and (c.get("model") or {}).get("name") == cfg.model["name"]
            and bool(c.get("dry_run")) == bool(cfg.raw.get("_dry_run"))
        ):
            best = d
    return best


class HeldOutError(RuntimeError):
    pass


def held_out_guard(cfg: ExperimentConfig, secrets: SecretSource, items: Sequence[Item], split: str,
                   dry_run: bool) -> dict[str, Any] | None:  # fmt: skip
    """`held_out: {approach: <name>}`: refuse if any evaluated code is among the secrets the approach
    used in training or model selection (its spec's `synthetic.used_secrets`, loaded by
    core.protocol._used_secrets, as for the team protocol's synthetic suite). Unlike the protocol's
    held_out_filter, nothing is dropped: an attack study compares two configs on the same items, so
    any overlap stops the run.

    Checked: the secrets of the items built here AND the codes of the configured split (with the
    config's own `benchmark.limit`, before any dry-run cap), so a dry run reports the overlap count
    of the real run. Calibration codes (the train split's first n_calib, used by calibrating
    likelihood attackers to fit tau) are counted and reported, not refused. Only counts and the
    file's sha256 are recorded or printed, never a secret.

    A missing used-secrets file refuses a real run; a dry run records it as not checked."""
    ho = cfg.raw.get("held_out")
    if not ho:
        return None
    from core.protocol import ProtocolError, _used_secrets, load_approach

    name = str(ho.get("approach") or "")
    if not name:
        raise HeldOutError("held_out needs `approach: <name>` (the approach whose used_secrets file is checked)")
    spec = load_approach(name)
    rec: dict[str, Any] = {"approach": name, "spec": Path(spec["_path"]).name, "split": split}
    if not (spec.get("synthetic") or {}).get("used_secrets"):
        raise HeldOutError(f"held_out: approach {name} declares no synthetic.used_secrets file; nothing to check against")
    try:
        got = _used_secrets(spec)
    except ProtocolError as e:
        if not dry_run:
            raise HeldOutError(f"held_out: {e}; refusing to run") from e
        return rec | {"checked": False, "reason": "used_secrets file missing here (dry run): overlap not checked"}
    assert got is not None
    path, data, used = got
    generated = callable(getattr(secrets, "secrets", None)) and hasattr(secrets, "split")  # digits_*, words
    raw_bench = cfg.raw.get("benchmark")
    lim = raw_bench.get("limit") if isinstance(raw_bench, dict) else None
    codes: list[str] = []
    if generated:
        idxs = list(secrets.split(split))  # type: ignore[attr-defined]
        codes = [secrets.secrets()[i] for i in (idxs[: int(lim)] if lim else idxs)]  # type: ignore[attr-defined]
    item_codes = [it.secret for it in items if it.secret]
    overlap_split = sum(1 for c in codes if c in used)
    overlap_items = sum(1 for c in item_codes if c in used)
    n_calib = max([int(a.get("n_calib", 50)) for a in cfg.attackers
                   if a["name"] == "likelihood" and a.get("calibrate", True) and "tau" not in a] or [0])  # fmt: skip
    calib_overlap = None
    if n_calib and generated:
        try:
            calib = [secrets.secrets()[i] for i in list(secrets.split("train"))[:n_calib]]  # type: ignore[attr-defined]
            calib_overlap = sum(1 for c in calib if c in used)
        except (KeyError, ValueError):  # a secret type without a train split
            calib_overlap = None
    rec |= {"checked": True, "used_secrets_sha256": hashlib.sha256(data).hexdigest(), "n_used": len(used),
            "n_split_codes_checked": len(codes), "overlap_split_codes": overlap_split,
            "n_items_checked": len(item_codes), "overlap_items": overlap_items,
            "n_calibration_codes": n_calib if calib_overlap is not None else 0,
            "overlap_calibration_codes": calib_overlap}  # fmt: skip
    if overlap_split or overlap_items:
        raise HeldOutError(f"held_out: {max(overlap_split, overlap_items)} evaluated code(s) of split {split} are in "
                           f"{path.name} (approach {name}): not held out; refusing to run")  # fmt: skip
    return rec


def run(cfg: ExperimentConfig, dry_run: bool = False, device: str | None = None) -> Path:
    registry.discover()
    if dry_run:
        cfg = apply_dry_run(cfg)
        cfg.raw["_dry_run"] = True
    if device:
        cfg.model["device"] = device
    secrets: SecretSource = registry.get("secret", cfg.secret["name"])(cfg.secret)
    bench = registry.get("benchmark", cfg.benchmark["name"])(cfg.benchmark, secrets, dry_run=dry_run)
    split = cfg.benchmark.get("split", "test")
    items = bench.items(split)
    if cfg.benchmark.get("limit"):
        items = items[: int(cfg.benchmark["limit"])]
    if hasattr(secrets, "bind"):
        secrets.bind(items)  # private_facts
    held = held_out_guard(cfg, secrets, items, split, dry_run)
    if held is not None:
        state = (f"overlap {held['overlap_split_codes']} of {held['n_split_codes_checked']} {split} codes, "
                 f"calibration overlap {held['overlap_calibration_codes']} of {held['n_calibration_codes']} "
                 f"(file sha256 {held['used_secrets_sha256'][:16]})"
                 if held["checked"] else held["reason"])  # fmt: skip
        print(f"held-out guard ({held['approach']}): {state}")

    base = registry.get("model", cfg.model["name"])(cfg.model)
    defence = registry.get("defence", cfg.defence["name"])(cfg.defence)
    defence.prepare(items, secrets)
    if cfg.defence.get("train"):
        defence.train({**cfg.defence, "_experiment": cfg.to_dict(), "_dry_run": dry_run, "_base": base,
                       "_benchmark": bench, "_secrets": secrets})  # fmt: skip
    model = defence.wrap(base)

    out = make_run_dir(cfg.output_dir, cfg.name)
    resolved = cfg.to_dict() | {"dry_run": dry_run, "split": split, "n_items": len(items),
                                "secret_sha256": _safe_sha(secrets)}  # fmt: skip
    if held is not None:
        resolved |= {"held_out": cfg.raw["held_out"], "held_out_check": held}
    (out / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8", newline="\n")
    git = git_commit()
    (out / "git_commit.txt").write_text(f"{git['commit']}\nbranch={git['branch']} dirty={git['dirty']}\n",
                                        encoding="utf-8", newline="\n")  # fmt: skip

    ctx = RunContext(cfg=cfg, benchmark=bench, secrets=secrets, model=model, base_model=base,
                     split=split, dry_run=dry_run, out_dir=out)  # fmt: skip
    ctx.secret_index = {it.id: int(it.metadata.get("secret_index", i)) for i, it in enumerate(items)}
    ctx.baseline_dir = find_baseline(cfg, out)

    t0 = time.perf_counter()
    replies = generate_replies(ctx, items)
    write_jsonl(out / "replies.jsonl", replies)

    guesses: list[Guess] = []
    main_replies = [r for r in replies if not r.control]
    ctrl_replies = [r for r in replies if r.control]
    for acfg in cfg.attackers:
        attacker = registry.get("attacker", acfg["name"])(acfg, ctx)
        budget = int(acfg.get("budget", cfg.budget))
        label = acfg.get("label", acfg["name"])  # lets one attacker run twice (e.g. two threat models)
        new = attacker.attack(model, items, main_replies, ctx.cap("budget", budget))
        if attacker.passive and ctrl_replies:
            for g in attacker.attack(model, items, ctrl_replies, ctx.cap("budget", budget)):
                g.control = True
                new.append(g)
        for g in new:
            g.attacker = label
        guesses.extend(new)
    write_jsonl(out / "guesses.jsonl", guesses)

    summaries: dict[str, Any] = {}
    per_item: dict[str, dict[str, float]] = {}
    for mcfg in cfg.metrics:
        metric = registry.get("metric", mcfg["name"])(mcfg, ctx)
        res = metric.compute(items, replies, guesses)
        summaries[mcfg["name"]] = res.get("summary", {})
        for key, vals in (res.get("per_item") or {}).items():
            per_item[f"{mcfg['name']}.{key}"] = vals
    write_json(out / "per_item.json", per_item)
    write_json(out / "metrics.json", {
        "config": cfg.name, "dry_run": dry_run, "git_commit": git["commit"], "split": split,
        "n_items": len(items), "n_replies": len(replies), "wall_s": round(time.perf_counter() - t0, 3),
        "baseline_run": str(ctx.baseline_dir) if ctx.baseline_dir else None, "held_out_check": held,
        "metrics": summaries,
    })  # fmt: skip
    return out


def _safe_sha(secrets: SecretSource) -> str | None:
    try:
        return secrets.sha256()
    except Exception:  # noqa: BLE001 - e.g. word list not downloaded; recorded as None
        return None
