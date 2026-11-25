"""Residual-stream features at the last prompt token (forward passes only, no generation).

Status: ✅ implemented + tested on the tiny random Gemma-3 and Qwen2 (tests/test_need_probe.py,
`--dry-run`); 🟡 on the real models (not yet run on a GPU).

For every item and condition (PRESENT: the code in the context; MASKED: the code replaced by
build.MASK) one teacher-forced pass over the chat-templated prompt (system + user, generation
prompt appended) captures the output of every decoder layer at two positions:
  user_end  the last token of the user's text (offset mapping: the token holding its last char)
  final     the last token of the templated prompt (end of the assistant header): exactly what the
            model sees before it generates its first token
Every item's positions are checked before any pass (PositionError, fail fast): user_end must
be the last token holding user text, final the last index. position_sanity() also returns a
startup log of both positions plus the 3 tokens before each, for 3 items per group, with the
code shown as <CODE> and the password as <PASSWORD>.
Reuses analysis/internals/backend.py: locate() (gemma / prenorm families), Capture (resid hook),
forward() (right padding; causal, so real tokens never see the padding), check_finite, max_abs.

Storage: one float32 array per (condition, position), shape [layers, items, hidden], in
features/<condition>_<position>.npy (memory-mapped). Passes run in length-sorted batches; each
batch is cached in cache/<condition>_<k>.npz first, so a stopped job resumes. Every stored slice
is checked for inf / NaN; the largest |activation| per layer is logged (massive activations).
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from analysis.internals import backend as B
from analysis.internals import measures as M
from analysis.need_probe.build import MARK, MASK

CONDITIONS = ("present", "masked")
POSITIONS = ("user_end", "final")


class BudgetExceeded(RuntimeError):
    """--max-hours reached; finished batches are cached, rerun the same command to resume."""


def system_text(item: dict[str, Any], condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    return item["system_template"].replace(MARK, item["code"] if condition == "present" else MASK)


def render(hf: Any, item: dict[str, Any], condition: str) -> str:
    return hf.template(B.messages(system_text(item, condition), item["user"]))


class PositionError(AssertionError):
    """A position does not land where it must (user_end in the user turn, final = last token)."""


def _tokenize(hf: Any, item: dict[str, Any], condition: str) -> dict[str, Any]:
    """Templated text, token ids, offsets, the user-content span and both positions (unchecked)."""
    text = render(hf, item, condition)
    tok, _ = hf.load()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    user = item["user"].strip()  # chat templates may trim the content (Gemma does)
    a = text.rfind(user)
    if not user or a < 0:
        raise PositionError(f"{item['id']} ({condition}): user text not found verbatim in the templated prompt")
    b = a + len(user)
    hits = [i for i, (s, e) in enumerate(offs) if s <= b - 1 < e]
    if not hits:
        raise PositionError(f"{item['id']} ({condition}): no token covers the end of the user text")
    return {"text": text, "ids": ids, "offs": offs, "user_span": (a, b),
            "pos": {"user_end": hits[-1], "final": len(ids) - 1}}  # fmt: skip


def check_positions(t: dict[str, Any], item_id: str, condition: str) -> None:
    """Fail fast unless user_end is the last token of the user's content (inside the user turn)
    and final is the last index of the templated prompt."""
    ids, offs, (a, b), pos = t["ids"], t["offs"], t["user_span"], t["pos"]
    where = f"{item_id} ({condition})"
    ue, fi = pos["user_end"], pos["final"]
    if fi != len(ids) - 1:
        raise PositionError(f"{where}: final = {fi}, but the templated prompt's last index is {len(ids) - 1}")
    if not 0 <= ue < fi:
        raise PositionError(f"{where}: user_end = {ue} is not before final = {fi}")
    s, e = offs[ue]
    if not (s < b and e > a and s < e):
        raise PositionError(f"{where}: user_end token {ue} spans chars [{s}, {e}), outside the user turn's "
                            f"content [{a}, {b})")  # fmt: skip
    later = [i for i in range(ue + 1, len(ids)) if offs[i][0] < b and offs[i][1] > a and offs[i][0] < offs[i][1]]
    if later:
        raise PositionError(f"{where}: user_end = {ue}, but token {later[-1]} still holds user text")


def encode(hf: Any, item: dict[str, Any], condition: str) -> tuple[list[int], dict[str, int]]:
    """Token ids of the templated prompt and the index of each position (checked)."""
    t = _tokenize(hf, item, condition)
    check_positions(t, item["id"], condition)
    return t["ids"], t["pos"]


SANITY_GROUPS = {"needed": ("needed",), "not_needed": ("not_needed",), "attack": ("attack",),
                 "hard": ("hard_a", "hard_b")}  # fmt: skip


def _shown_tokens(tok: Any, t: dict[str, Any], idx: int, secrets: list[tuple[str, str]], n_before: int) -> str:
    """Tokens idx-n_before .. idx, decoded; any token overlapping a secret's span becomes its tag."""
    text, ids, offs = t["text"], t["ids"], t["offs"]
    spans = [(m, m + len(v), tag) for v, tag in secrets if v
             for m in _find_all(text, v)]  # fmt: skip
    out = []
    for i in range(max(0, idx - n_before), idx + 1):
        s, e = offs[i]
        tag = next((tg for a, b, tg in spans if s < e and s < b and e > a), None)
        shown = tag if tag else tok.decode([ids[i]])
        for v, tg in secrets:  # belt and braces: never show a secret even if offsets were odd
            if v:
                shown = shown.replace(v, tg)
        out.append(repr(shown))
    return " ".join(out[:-1]) + (" | " if len(out) > 1 else "") + f"[{out[-1]}]"


def _find_all(text: str, v: str) -> list[int]:
    out, k = [], text.find(v)
    while k >= 0:
        out.append(k)
        k = text.find(v, k + 1)
    return out


def position_sanity(hf: Any, items: list[dict[str, Any]], conditions: list[str] | tuple[str, ...] = CONDITIONS,
                    per_group: int = 3, n_before: int = 3) -> list[str]:  # fmt: skip
    """Check every item's positions (raises PositionError on the first bad one) and return log
    lines for `per_group` items per group: the token at user_end and at final with the n_before
    tokens before each, in the PRESENT condition. The code is shown as <CODE> and the scenario
    password as <PASSWORD>; neither is ever printed."""
    tok, _ = hf.load()
    for it in items:
        for c in conditions:
            check_positions(_tokenize(hf, it, c), it["id"], c)
    lines = [f"position check: user_end inside the user turn and final = last index for all "
             f"{len(items)} items x {len(conditions)} conditions"]  # fmt: skip
    for name, groups in SANITY_GROUPS.items():
        picked = [it for it in items if it["group"] in groups][:per_group]
        for it in picked:
            t = _tokenize(hf, it, "present")
            secrets = [(it["code"], "<CODE>"), (it.get("password") or "", "<PASSWORD>")]
            n = len(t["ids"])
            for p in ("user_end", "final"):
                k = t["pos"][p]
                lines.append(f"  {name:<10} {p:<8} idx {k:>5}/{n - 1}: {_shown_tokens(tok, t, k, secrets, n_before)}")
    return lines


def batches(lengths: list[int], batch_size: int, max_tokens: int) -> list[list[int]]:
    """Length-sorted batches (deterministic): at most batch_size rows and max_tokens padded tokens."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    out: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (len(cur) + 1 > batch_size or (len(cur) + 1) * lengths[i] > max_tokens):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


class Extractor:
    def __init__(self, cfg: dict[str, Any], run_dir: Path, hf: Any, items: list[dict[str, Any]],
                 max_hours: float | None = None) -> None:  # fmt: skip
        self.cfg, self.run_dir, self.hf, self.items = cfg, run_dir, hf, items
        tok, model = hf.load()
        self.h = B.locate(model, tok)
        self.conditions = list(cfg.get("conditions", CONDITIONS))
        self.positions = list(cfg.get("positions", POSITIONS))
        for p in self.positions:
            if p not in POSITIONS:
                raise ValueError(f"unknown position {p!r}; use {POSITIONS}")
        self.t0 = time.time()
        self.max_s = max_hours * 3600 if max_hours is not None else None
        self.cache = run_dir / "cache"
        self.feat_dir = run_dir / "features"
        self.maxabs: dict[int, tuple[float, int]] = {}

    def _budget(self) -> None:
        if self.max_s is not None and time.time() - self.t0 > self.max_s:
            raise BudgetExceeded(f"stopped after {self.max_s / 3600:.2f} h; rerun with --run-dir {self.run_dir} to resume")

    def meta(self) -> dict[str, Any]:
        return {"n_items": len(self.items), "item_ids": [it["id"] for it in self.items],
                "n_layers": self.h.n_layers, "hidden_size": self.h.hidden_size, "family": self.h.family,
                "layer_types": self.h.layer_types, "sliding_window": self.h.sliding_window,
                "conditions": self.conditions, "positions": self.positions, "mask": MASK}  # fmt: skip

    def done(self) -> bool:
        """Features already assembled for exactly these items / layers."""
        p = self.feat_dir / "meta.json"
        if not p.exists():
            return False
        old = json.loads(p.read_text(encoding="utf-8"))
        same = {k: old.get(k) for k in ("item_ids", "n_layers", "hidden_size", "conditions", "positions")}
        new = self.meta()
        return same == {k: new[k] for k in same} and all(
            (self.feat_dir / f"{c}_{p}.npy").exists() for c in self.conditions for p in self.positions)

    def run(self) -> dict[str, Any]:
        if self.done():
            print("features: already assembled, skipping the forward passes", flush=True)
            return json.loads((self.feat_dir / "meta.json").read_text(encoding="utf-8"))
        self._check_cache_key()
        fw = self.cfg["forward"]
        plan: dict[str, list[list[int]]] = {}
        enc: dict[str, list[tuple[list[int], dict[str, int]]]] = {}
        for c in self.conditions:
            enc[c] = [encode(self.hf, it, c) for it in self.items]
            plan[c] = batches([len(e[0]) for e in enc[c]], int(fw["batch_size"]), int(fw["max_batch_tokens"]))
        n_tok = sum(len(e[0]) for c in self.conditions for e in enc[c])
        print(f"features: {len(self.items)} items x {len(self.conditions)} conditions, {n_tok:,} prompt tokens, "
              f"{sum(len(b) for b in plan.values())} batches", flush=True)  # fmt: skip
        for c in self.conditions:
            for k, idx in enumerate(plan[c]):
                path = self.cache / f"{c}_{k:05d}.npz"
                if self._valid(path, idx):
                    continue
                self._budget()
                x = self._batch([enc[c][i] for i in idx], f"{c} batch {k}")
                np.savez(path, x=x, idx=np.array(idx))
        self.report_maxabs()
        return self.assemble(plan)

    def _check_cache_key(self) -> None:
        """Batch caches belong to one (items, model, conditions, positions); a mismatch wipes them."""
        import hashlib

        blob = json.dumps([[it["id"], it["user"], it["system_template"], it["code"]] for it in self.items]
                          + [self.cfg["model"]["name"], self.h.n_layers, self.h.hidden_size,
                             self.conditions, self.positions])  # fmt: skip
        key = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        kp = self.cache / "key.txt"
        if self.cache.exists() and (not kp.exists() or kp.read_text(encoding="utf-8").strip() != key):
            print("features: cache/ was made for other items or another model: clearing it", flush=True)
            shutil.rmtree(self.cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        kp.write_text(key + "\n", encoding="utf-8")

    def _valid(self, path: Path, idx: list[int]) -> bool:
        if not path.exists():
            return False
        z = np.load(path)
        x = z["x"]
        return list(z["idx"]) == idx and x.dtype == np.float32 and bool(np.isfinite(x).all())

    def _batch(self, rows: list[tuple[list[int], dict[str, int]]], name: str) -> np.ndarray:
        """[B, positions, layers, hidden] float32."""
        import torch

        h = self.h
        out = np.zeros((len(rows), len(self.positions), h.n_layers, h.hidden_size), dtype=np.float32)
        lens = [len(ids) for ids, _ in rows]
        pos = {p: torch.tensor([r[1][p] for r in rows]) for p in self.positions}

        def fn(kind: str, i: int, t: Any) -> None:
            v, d = M.max_abs(t, lens)
            if not math.isfinite(v):
                raise M.NonFiniteError(f"part features, layer {i}, array 'resid' ({name}): non-finite activations")
            if i not in self.maxabs or v > self.maxabs[i][0]:
                self.maxabs[i] = (v, d)
            ar = torch.arange(t.shape[0], device=t.device)
            for k, p in enumerate(self.positions):
                out[:, k, i] = t[ar, pos[p].to(t.device)].float().cpu().numpy()
            M.check_finite(out[:, :, i], "features", i, name)

        with B.Capture(h, {"resid"}, fn):
            B.forward(h, [ids for ids, _ in rows])
        return out

    def report_maxabs(self) -> None:
        if not self.maxabs:
            return
        layers = sorted(self.maxabs)
        peak = max(layers, key=lambda i: self.maxabs[i][0])
        print("max|resid| per layer [features]: " + " ".join(f"{i}:{self.maxabs[i][0]:.3g}" for i in layers)
              + f" (peak {self.maxabs[peak][0]:.4g} at layer {peak}, dim {self.maxabs[peak][1]})", flush=True)  # fmt: skip
        with open(self.run_dir / "max_abs.csv", "w", encoding="utf-8", newline="\n") as f:
            f.write("part,kind,layer,layer_type,max_abs,dim\n")
            for i in layers:
                f.write(f"features,resid,{i},{self.h.layer_types[i]},{self.maxabs[i][0]:.6g},{self.maxabs[i][1]}\n")

    def assemble(self, plan: dict[str, list[list[int]]]) -> dict[str, Any]:
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        L, N, H = self.h.n_layers, len(self.items), self.h.hidden_size
        for c in self.conditions:
            arrs = {p: np.lib.format.open_memmap(self.feat_dir / f"{c}_{p}.npy", mode="w+", dtype=np.float32,
                                                 shape=(L, N, H)) for p in self.positions}  # fmt: skip
            filled = np.zeros(N, dtype=bool)
            for k, idx in enumerate(plan[c]):
                x = np.load(self.cache / f"{c}_{k:05d}.npz")["x"]  # [B, P, L, H]
                for j, p in enumerate(self.positions):
                    arrs[p][:, idx, :] = x[:, j].transpose(1, 0, 2)
                filled[idx] = True
            if not filled.all():
                raise AssertionError(f"{c}: {int((~filled).sum())} items without features")
            for p, a in arrs.items():
                for i in range(L):
                    M.check_finite(a[i], "features", i, f"features/{c}_{p}.npy")
                a.flush()
                del a
        meta = self.meta()
        (self.feat_dir / "meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
        print(f"features: wrote {len(self.conditions) * len(self.positions)} arrays [{L}, {N}, {H}] float32 "
              f"({len(self.conditions) * len(self.positions) * L * N * H * 4 / 1e9:.2f} GB) -> {self.feat_dir}",
              flush=True)  # fmt: skip
        return meta


def drop_cache(run_dir: Path) -> None:
    c = run_dir / "cache"
    if c.exists():
        shutil.rmtree(c)
