"""The measurements of analysis/internals (parts 1a, 1b, 1c, 2, 3a, 3b).

Status: ✅ implemented + tested end to end on the tiny random Gemma-3 and Qwen2 (`--dry-run`,
tests/test_internals.py); 🟡 on Gemma 3 12B and Qwen2.5-7B-Instruct (not yet run on a GPU).

Every reply is generated once (temperature 1, seed = harness reply_seed(secret_index, 0)),
cached in replies.jsonl, and reused by every part; internals come only from ONE teacher-forced
pass over prompt + reply, never from generation. Reply positions = the positions of the reply's
tokens in that pass. Only summaries are written (no full tensors); 1b keeps pooled features
(one [layers, hidden] vector per reply) in cache/ so a stopped job can resume.

Numerics: every stored array (1b features, final-layer probabilities for the JSD, neuron
activations, sensitivity accumulators) is float32 or wider, never float16, because Gemma 3's
massive activations exceed float16's max (65,504). Each array is checked as it is written and the
run stops with NonFiniteError (naming part, layer, array) on inf/NaN. A 1b cache that is not
float32 or not finite (e.g. from the old float16 code) is ignored and recomputed. The largest
|activation| per layer is printed and appended to max_abs.csv in the run folder.

Conditions come from the prompts file. The hand-written prompts.yaml gives A / B / C (+ D) with one
system prompt; the bench file (analysis/internals/bench_prompts.py) gives A / W / T / C / B (+ D)
with a system prompt per prompt and a `parts:` block: 1a / 2 / 3 run on every secret condition,
1b on the probe conditions (one probe per condition, pooled over its prompts), 1c on the lens
condition vs D. part1.fixed_replies gives a condition more than one fixed reply in 1a / 3 (rows
carry secret_index). Codes in secret.exclude_codes_file are dropped from the train / test index
ranges, and indexing one raises ExcludedCodeError (GuardedCodes). With the hand-written prompts and
no exclude file every number is the same as before these additions (checked on both dry runs).
Bench status: 🟡 dry-run tested on both tiny families (tests/test_internals_bench.py), not yet run
on a GPU.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from analysis.internals import backend as B
from analysis.internals import measures as M
from analysis.internals import probe as P
from core.config import reply_seed
from core.interfaces import GenConfig
from core.io import ROOT

CONDS = ["A_needed", "B_leak_prone", "C_low_leak"]
D = "D_no_secret"
# Which conditions each part uses when the prompts file has no `parts:` block (prompts.yaml).
DEFAULT_PARTS: dict[str, Any] = {"secret": CONDS, "probe": ["B_leak_prone"], "lens": "B_leak_prone",
                                 "neurons": "B_leak_prone", "contrast": ["B_leak_prone", "C_low_leak"]}  # fmt: skip


class BudgetExceeded(RuntimeError):
    """Raised when --max-hours is reached; everything done so far is saved, rerun to resume."""


class ExcludedCodeError(AssertionError):
    """A code listed in secret.exclude_codes_file was about to be used."""


class GuardedCodes(list):  # type: ignore[type-arg]
    """The code list; indexing an excluded code raises ExcludedCodeError (empty set: a plain list)."""

    def __init__(self, codes: list[str], excluded: set[str] | frozenset[str] = frozenset()) -> None:
        super().__init__(codes)
        self.excluded = frozenset(excluded)

    def __getitem__(self, i: Any) -> Any:
        v = super().__getitem__(i)
        if not isinstance(i, slice) and v in self.excluded:
            raise ExcludedCodeError(f"secret index {i} holds an excluded code (secret.exclude_codes_file)")
        return v


def load_prompts(src: str | Path | dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    """A prompts file (path) or its parsed dict. Conditions are the file's `conditions` keys, in
    order; optional per-prompt `system` / `system_no_secret` override the file-wide
    system_secret / system_no_secret, and an optional `parts:` block overrides DEFAULT_PARTS."""
    from benchmarks.hidden_code import DEFAULT_SEQ_LENGTH, USER_TEMPLATES

    if isinstance(src, dict):
        raw = src
    else:
        p = Path(src)
        raw = yaml.safe_load((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8"))
    conds: dict[str, list[tuple[str, str]]] = {}
    systems: dict[tuple[str, str], tuple[str, str]] = {}
    for cond, entries in raw["conditions"].items():
        rows = []
        for e in entries:
            if "template" in e:
                text = USER_TEMPLATES[e["template"]].format(n_digits=4, seq_length=DEFAULT_SEQ_LENGTH)
            else:
                text = e["text"]
            pid = str(e["id"])
            rows.append((pid, text))
            sys_s, sys_n = e.get("system") or raw.get("system_secret"), e.get("system_no_secret") or raw.get("system_no_secret")
            if not sys_s or sys_s.count("{code}") != 1:
                raise ValueError(f"{cond}/{pid}: the system prompt needs exactly one {{code}}")
            if not sys_n or "{code}" in sys_n:
                raise ValueError(f"{cond}/{pid}: needs a system prompt without {{code}} (system_no_secret)")
            systems[(cond, pid)] = (sys_s, sys_n)
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError(f"duplicate prompt ids in {cond}")
        conds[cond] = rows[:limit] if limit else rows
    parts = {**DEFAULT_PARTS, **(raw.get("parts") or {})}
    used = [*parts["secret"], *parts["probe"], parts["lens"], parts["neurons"], *parts["contrast"], *raw["no_secret_from"]]
    missing = sorted({c for c in used if c not in conds})
    if missing:
        raise ValueError(f"conditions {missing} are used by `parts` / no_secret_from but not defined")
    probe_ids = [pid for c in parts["probe"] for pid, _ in conds[c]]
    if len(set(probe_ids)) != len(probe_ids):
        raise ValueError("prompt ids of the probe conditions must be unique together (1b cache names)")
    d_rows = [(pid, text, src) for src in raw["no_secret_from"] for pid, text in conds[src]]
    if len({r[0] for r in d_rows}) != len(d_rows):
        raise ValueError("prompt ids of the no_secret_from conditions must be unique together")
    return {"system_secret": raw.get("system_secret"), "system_no_secret": raw.get("system_no_secret"),
            "conds": conds, "D": d_rows, "systems": systems, "parts": parts}  # fmt: skip


def prompt_source(cfg: dict[str, Any]) -> str | Path | dict[str, Any]:
    """The prompts of a config: an in-memory bench set (dry run) or the `prompts` file."""
    return cfg.get("_prompts_data") or cfg["prompts"]


class Study:
    def __init__(self, cfg: dict[str, Any], run_dir: Path, hf: Any, codes: list[str],
                 max_hours: float | None = None, exclude: set[str] | None = None) -> None:  # fmt: skip
        self.cfg, self.run_dir, self.hf = cfg, run_dir, hf
        self.codes = GuardedCodes(list(codes), exclude or set())
        tok, model = hf.load()
        self.h = B.locate(model, tok)
        self.prompts = load_prompts(prompt_source(cfg), cfg.get("prompts_per_condition"))
        self.sys = self.prompts["system_secret"]
        parts = self.prompts["parts"]
        self.conds: list[str] = list(parts["secret"])
        self.probe_conds: list[str] = list(parts["probe"])
        self.lens_cond: str = parts["lens"]
        self.d_src = {pid: src for pid, _, src in self.prompts["D"]}
        ex = self.codes.excluded
        raw = list.__iter__(self.codes)
        allowed = [c not in ex for c in raw]
        self.train_idx = [i for i in range(*cfg["splits"]["train"]) if allowed[i]]
        self.test_idx = [i for i in range(*cfg["splits"]["test"]) if allowed[i]]
        self.allowed_codes = [c for c, ok in zip(list.__iter__(self.codes), allowed) if ok]
        self.n_excluded = {"train": len(range(*cfg["splits"]["train"])) - len(self.train_idx),
                           "test": len(range(*cfg["splits"]["test"])) - len(self.test_idx),
                           "all": len(allowed) - sum(allowed)}  # fmt: skip
        g = cfg["generation"]
        self.gen = GenConfig(max_new_tokens=int(g["max_new_tokens"]), temperature=float(g["temperature"]),
                             top_p=float(g.get("top_p", 1.0)))  # fmt: skip
        self.min_new = int(g.get("min_new_tokens", 1))
        self.fwd_bs = int(cfg["forward"]["batch_size"])
        self.lens_chunk = int(cfg["forward"].get("lens_chunk", 64))
        self.t0 = time.time()
        self.max_s = max_hours * 3600 if max_hours is not None else None
        self.replies: dict[str, dict[str, Any]] = {}
        self.replies_path = run_dir / "replies.jsonl"
        if self.replies_path.exists():
            for line in self.replies_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.replies[r["key"]] = r
        (run_dir / "cache").mkdir(exist_ok=True)
        self.digit_ids = self._digit_ids()
        self.maxabs: dict[tuple[str, str], dict[int, tuple[float, int]]] = {}

    # -- prompts ---------------------------------------------------------------------------
    def system(self, cond: str, pid: str) -> str:
        """The system template (with {code}) of one secret prompt."""
        return self.prompts["systems"][(cond, pid)][0]

    def system_no_secret(self, pid: str) -> str:
        """The no-secret system prompt of one D prompt (from its source condition)."""
        return self.prompts["systems"][(self.d_src[pid], pid)][1]

    def fixed_secrets(self, cond: str) -> list[int]:
        """1a / 3: the test secrets whose replies are held fixed (part1.fixed_replies, default 1)."""
        k = int((self.cfg["part1"].get("fixed_replies") or {}).get(cond, 1))
        return self.test_idx[:k]

    def probe_prompts(self, cond: str) -> list[tuple[str, str]]:
        """1b: the condition's prompts, at most part1.probe.prompts_per_condition (default: all)."""
        cap = self.cfg["part1"]["probe"].get("prompts_per_condition")
        rows = self.prompts["conds"][cond]
        return rows[: int(cap)] if cap else rows

    # -- checks ----------------------------------------------------------------------------
    def check_tokenisation(self) -> dict[str, Any]:
        """All codes x all secret prompts: same length, same span, one token per digit. Stops at
        the first prompt that fails, naming it."""
        out = {}
        for cond in self.conds:
            for pid, user in self.prompts["conds"][cond]:
                try:
                    out[f"{cond}/{pid}"] = B.check_equal_tokenisation(self.hf, self.system(cond, pid), user,
                                                                      self.allowed_codes)  # fmt: skip
                except AssertionError as e:
                    raise AssertionError(f"equal tokenisation failed for prompt {cond}/{pid}: {e}") from e
        first = next(iter(out.values()))
        res = {"span_tokens": first["span_tokens"], "digit_pos": first["digit_pos"],
               "n_codes": len(self.allowed_codes), "prompts_checked": len(out)}  # fmt: skip
        if len({tuple(v["digit_pos"]) for v in out.values()}) > 1:  # per-prompt system prompts
            spans = {v["span_tokens"] for v in out.values()}
            pos = [v["digit_pos"][0] for v in out.values()]
            res.update({"span_tokens": sorted(spans) if len(spans) > 1 else first["span_tokens"],
                        "digit_pos": f"prompt-specific (first digit at token {min(pos)}-{max(pos)})",
                        "digit_pos_by_prompt": {k: v["digit_pos"] for k, v in out.items()}})  # fmt: skip
        return res

    def check_never_excluded(self) -> int:
        """Every cached reply's code is allowed; returns the number of replies checked."""
        ex = self.codes.excluded
        for r in self.replies.values():
            if r.get("code") in ex or (ex and list.__getitem__(self.codes, int(r["secret_index"])) in ex):
                raise ExcludedCodeError(f"reply {r['condition']}|{r['prompt_id']}|{r['secret_index']} used an excluded code")
        return len(self.replies)

    def leak_report(self) -> list[dict[str, Any]]:
        """reply_leaks.csv + leak_rate.csv from every reply so far; returns the rate table."""
        import pandas as pd

        from analysis.internals import leaks as LK

        rows = LK.reply_rows(self.replies.values(), self.codes)
        table = LK.rate_table(rows, [*self.conds, D])
        pd.DataFrame(rows, columns=["condition", "prompt_id", "secret_index", "code_shown", "n_tokens",
                                    "leak_exact", "leak_spaced", "leaked"]).to_csv(self.run_dir / "reply_leaks.csv", index=False)  # fmt: skip
        pd.DataFrame(table, columns=["condition", "n_replies", "leaked", "not_leaked", "leak_exact",
                                     "leak_spaced_only", "leak_rate", "note"]).to_csv(self.run_dir / "leak_rate.csv", index=False)  # fmt: skip
        return table

    def _digit_ids(self) -> list[int]:
        ids = []
        for d in "0123456789":
            enc = self.h.tok(d, add_special_tokens=False)["input_ids"]
            if len(enc) != 1:
                raise AssertionError(f"digit {d!r} is not a single token: {enc}")
            ids.append(int(enc[0]))
        return ids

    # -- numerics ----------------------------------------------------------------------------
    def _track(self, part: str, kind: str, layer: int, t: Any, lens: list[int] | None = None) -> None:
        """Running max |activation| (and its hidden dim) over real (non-padding) tokens."""
        v, d = M.max_abs(t, lens)
        if not math.isfinite(v):
            raise M.NonFiniteError(f"part {part}, layer {layer}, array {kind!r}: model activations are non-finite")
        cur = self.maxabs.setdefault((part, kind), {})
        if layer not in cur or v > cur[layer][0]:
            cur[layer] = (v, int(d))

    def report_maxabs(self, part: str) -> None:
        """One log line per (part, kind) + rows appended to max_abs.csv."""
        path = self.run_dir / "max_abs.csv"
        new = not path.exists()
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            if new:
                f.write("part,kind,layer,layer_type,max_abs,dim\n")
            for (p, kind), cur in sorted(self.maxabs.items()):
                if p != part or not cur:
                    continue
                layers = sorted(cur)
                peak = max(layers, key=lambda i: cur[i][0])
                body = " ".join(f"{i}:{cur[i][0]:.3g}" for i in layers)
                print(f"max|{kind}| per layer [{part}]: {body} (peak {cur[peak][0]:.4g} at layer {peak}, "
                      f"dim {cur[peak][1]})", flush=True)  # fmt: skip
                for i in layers:
                    f.write(f"{p},{kind},{i},{self.h.layer_types[i]},{cur[i][0]:.6g},{cur[i][1]}\n")
                cur.clear()

    def _load_cache(self, path: Path, sec: list[int]) -> np.ndarray | None:
        """A 1b cache is reused only if it has the same secrets, is float32 and is all finite."""
        if not path.exists():
            return None
        z = np.load(path)
        x = z["x"]
        if list(z["secret_index"]) != sec:
            return None
        if x.dtype != np.float32 or not np.isfinite(x).all():
            print(f"ignoring stale cache {path.name} (dtype {x.dtype}, "
                  f"{int((~np.isfinite(x)).sum())} non-finite values): recomputing", flush=True)  # fmt: skip
            return None
        return x

    def _budget(self) -> None:
        if self.max_s is not None and time.time() - self.t0 > self.max_s:
            raise BudgetExceeded(f"stopped after {self.max_s / 3600:.2f} h; rerun with --run-dir {self.run_dir} to resume")

    # -- replies ---------------------------------------------------------------------------
    def reply(self, cond: str, pid: str, user: str, secret_idx: list[int]) -> dict[int, dict[str, Any]]:
        """Replies for one prompt and these secret indices (D: no secret, seed still paired)."""
        with_secret = cond != D
        system = self.system(cond, pid) if with_secret else self.system_no_secret(pid)
        key = lambda i: f"{cond}|{pid}|{i}"  # noqa: E731
        todo = [i for i in secret_idx if key(i) not in self.replies]
        bs = int(self.cfg["generation"]["batch_size"])
        for a in range(0, len(todo), bs):
            self._budget()
            chunk = todo[a : a + bs]
            encs = [B.encode(self.hf, system, user, self.codes[i] if with_secret else None) for i in chunk]
            seeds = [reply_seed(i, 0) for i in chunk]
            t = time.time()
            outs = B.generate(self.h, [e.ids for e in encs], seeds, self.gen, self.min_new)
            dt = (time.time() - t) / len(chunk)
            with open(self.replies_path, "a", encoding="utf-8", newline="\n") as f:
                for i, s, ids in zip(chunk, seeds, outs):
                    rec = {"key": key(i), "condition": cond, "prompt_id": pid, "secret_index": i,
                           "code": self.codes[i] if with_secret else None, "seed": s,
                           "text": self.h.tok.decode(ids, skip_special_tokens=True),
                           "reply_ids": ids, "n_tokens": len(ids), "latency_s": dt}  # fmt: skip
                    self.replies[rec["key"]] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {i: self.replies[key(i)] for i in secret_idx}

    def distinct_train_codes(self, n: int) -> list[int]:
        seen, out = set(), []
        for i in self.train_idx:
            if self.codes[i] not in seen:
                seen.add(self.codes[i])
                out.append(i)
            if len(out) == n:
                return out
        raise ValueError(f"only {len(out)} distinct train codes")

    # -- 1a / 3a / 3b: one fixed reply, n codes swapped in ------------------------------------------
    def sweep(self, n_codes: int, hidden: bool, components: bool, neurons: bool) -> dict[str, list[dict[str, Any]]]:
        import torch

        h, L = self.h, self.h.n_layers
        code_idx = self.distinct_train_codes(n_codes)
        first_test = self.test_idx[0]
        kinds = ({"resid"} if hidden else set()) | ({"attn_out", "mlp_out"} if components else set()) \
            | ({"mlp_act"} if neurons else set())  # fmt: skip
        rows: dict[str, list[dict[str, Any]]] = {"hidden": [], "jsd": [], "components": [], "neurons": []}
        nstats = {c: M.NeuronStats() for c in [*self.conds, D]}
        fixed: dict[tuple[str, str], list[int]] = {}
        for cond, pid, user, si in [(c, pid, user, si) for c in self.conds for pid, user in self.prompts["conds"][c]
                                    for si in self.fixed_secrets(c)]:  # fmt: skip
                rec = self.reply(cond, pid, user, [si])[si]
                reply = rec["reply_ids"]
                if si == first_test:
                    fixed[(cond, pid)] = reply
                r = len(reply)
                if r == 0:
                    continue
                system = self.system(cond, pid)
                encs = [B.encode(self.hf, system, user, self.codes[i]) for i in code_idx]
                p_len = len(encs[0].ids)
                assert all(len(e.ids) == p_len for e in encs)
                sl = slice(p_len, p_len + r)
                acc = {k: M.Sensitivity(L, r, h.hidden_size, h.device) for k in ("resid", "attn_out", "mlp_out") if k in kinds}
                # final-layer next-token probabilities, float32 [codes, r, vocab] on CPU (for the JSD)
                probs: dict[str, Any] = {"x": None, "row": 0}
                buf: list[Any] = [None] * L
                part_name = "1a" if hidden else "3"

                def fn(kind: str, i: int, t: Any) -> None:
                    if kind in acc:
                        self._track(part_name, kind, i, t)
                        acc[kind].add(i, t[:, sl])
                    if kind == "resid" and hidden and i == L - 1:
                        row = probs["row"]
                        for a in range(0, r, self.lens_chunk):
                            p = B.lens_logprobs(h, t[:, sl][:, a : a + self.lens_chunk]).exp().float()
                            M.check_finite(p, "1a", i, f"{cond}/{pid} final-layer probabilities")
                            if probs["x"] is None:
                                probs["x"] = torch.zeros((len(encs), r, p.shape[-1]), dtype=torch.float32)
                            probs["x"][row : row + t.shape[0], a : a + p.shape[1]] = p.cpu()
                        probs["row"] = row + t.shape[0]
                    if kind == "mlp_act":
                        buf[i] = t[:, sl].float().mean(1)
                        M.check_finite(buf[i], "3b", i, f"{cond}/{pid} mean MLP activations")

                seqs = [e.ids + reply for e in encs]
                for a in range(0, len(seqs), self.fwd_bs):
                    with B.Capture(h, kinds, fn):
                        B.forward(h, seqs[a : a + self.fwd_bs])
                    if neurons:
                        nstats[cond].add(pid if si == first_test else f"{pid}#{si}", torch.stack(buf, 1).cpu())
                base = {"condition": cond, "prompt_id": pid, "secret_index": si, "n_codes": len(code_idx),
                        "reply_tokens": r}  # fmt: skip
                for k, s_acc in acc.items():
                    for i in range(L):
                        for nm, arr in (("sum", s_acc.s1[i]), ("sum_sq_diff", s_acc.s2d[i]), ("sum_sq", s_acc.s2[i])):
                            M.check_finite(arr, part_name, i, f"{cond}/{pid} {k} sensitivity {nm}")
                if hidden:
                    frac = acc["resid"].fraction()
                    for i in range(L):
                        for pos in range(r):
                            rows["hidden"].append({**base, "layer": i, "layer_type": h.layer_types[i],
                                                   "position": pos, "fraction": float(frac[i, pos])})  # fmt: skip
                    jsd = M.pairwise_jsd(probs["x"], device=h.device)
                    probs["x"] = None
                    for pos in range(r):
                        rows["jsd"].append({**base, "position": pos, "jsd_bits": float(jsd[pos])})
                if components:
                    fr = {k: acc[k].fraction().mean(1) for k in acc}
                    for i in range(L):
                        rows["components"].append({**base, "layer": i, "layer_type": h.layer_types[i],
                                                   **{f"{k}_fraction": float(v[i]) for k, v in fr.items()}})  # fmt: skip
        if neurons:  # D: the same fixed replies with the no-secret system prompt
            for pid, user, src in self.prompts["D"]:
                reply = fixed.get((src, pid), [])
                if not reply:
                    continue
                e = B.encode(self.hf, self.system_no_secret(pid), user, None)
                sl = slice(len(e.ids), len(e.ids) + len(reply))
                buf = [None] * L

                def fn_d(kind: str, i: int, t: Any, pid: str = pid) -> None:
                    buf[i] = t[:, sl].float().mean(1)
                    M.check_finite(buf[i], "3b", i, f"{D}/{pid} mean MLP activations")

                with B.Capture(h, {"mlp_act"}, fn_d):
                    B.forward(h, [e.ids + reply])
                nstats[D].add(pid, torch.stack(buf, 1).cpu())
            rows["neurons"] = self._top_neurons(nstats, int(self.cfg["part3"]["top_neurons"]))
        self.report_maxabs("1a" if hidden else "3")
        return rows

    def _top_neurons(self, nstats: dict[str, M.NeuronStats], k: int) -> list[dict[str, Any]]:
        import torch

        score = M.neuron_scores(nstats[self.prompts["parts"]["neurons"]])
        flat = torch.nan_to_num(score, nan=-1.0).flatten()
        top = torch.topk(flat, min(k, flat.numel())).indices.tolist()
        width = score.shape[1]
        means = {c: nstats[c].mean() for c in nstats if nstats[c].sum}
        sds = {c: nstats[c].within_var().sqrt() for c in self.conds if nstats[c].sum}
        out = []
        for rank, f in enumerate(top, 1):
            layer, n = divmod(f, width)
            row = {"rank": rank, "layer": layer, "layer_type": self.h.layer_types[layer], "neuron": n,
                   "score": float(score[layer, n])}  # fmt: skip
            for c in [*self.conds, D]:
                short = c[0]
                row[f"mean_{short}"] = float(means[c][layer, n]) if c in means else math.nan
                if c in sds:
                    row[f"sd_codes_{short}"] = float(sds[c][layer, n])
            out.append(row)
        return out

    # -- 1b: where does the secret live? ---------------------------------------------------
    def part1b(self) -> list[dict[str, Any]]:
        pc = self.cfg["part1"]["probe"]
        n_tr, n_te, first = int(pc["n_train"]), int(pc["n_test"]), int(pc["first_reply_tokens"])
        sec = self.train_idx[:n_tr] + self.test_idx[:n_te]
        feats: dict[str, dict[str, np.ndarray]] = {c: {} for c in self.probe_conds}
        for cond, pid, user in [(c, pid, u) for c in self.probe_conds for pid, u in self.probe_prompts(c)]:
            path = self.run_dir / "cache" / f"1b_{pid}.npz"
            cached = self._load_cache(path, sec)
            if cached is not None:
                feats[cond][pid] = cached
                continue
            recs = self.reply(cond, pid, user, sec)
            feats[cond][pid] = self._pooled(self.system(cond, pid), user, [recs[i] for i in sec], first, path.name)
            M.check_finite(feats[cond][pid], "1b", None, path.name)
            np.savez(path, x=feats[cond][pid], secret_index=np.array(sec))
        ctrl_path = self.run_dir / "cache" / "1b_control.npz"
        ctrl = self._load_cache(ctrl_path, sec)
        if ctrl is None:
            ctrl = self._at_secret(sec)
            M.check_finite(ctrl, "1b", None, ctrl_path.name)
            np.savez(ctrl_path, x=ctrl, secret_index=np.array(sec))
        self.report_maxabs("1b")
        y = P.digits_of([self.codes[i] for i in sec])
        ytr, yte = y[:n_tr], y[n_tr:]
        rng = np.random.default_rng(int(pc.get("shuffle_seed", 2026)))
        kw = {"l2": float(pc["l2"]), "iters": int(pc["iters"]), "lr": float(pc["lr"])}
        rows = []
        for layer in range(self.h.n_layers):
            # one probe per probe condition, pooled over its prompts (hand-written run: B only)
            for cond in self.probe_conds:
                pids = list(feats[cond])
                xtr = np.concatenate([feats[cond][p][:n_tr, layer] for p in pids]).astype(np.float32)
                xte = np.concatenate([feats[cond][p][n_tr:, layer] for p in pids]).astype(np.float32)
                ytr_all, yte_all = np.tile(ytr, (len(pids), 1)), np.tile(yte, (len(pids), 1))
                accs = {"reply": P.accuracy(P.fit(xtr, ytr_all, **kw), xte, yte_all)}
                accs["shuffled"] = P.accuracy(P.fit(xtr, ytr_all[rng.permutation(len(ytr_all))], **kw), xte, yte_all)
                for variant, acc in accs.items():
                    for k, a in enumerate(acc):
                        rows.append({"variant": variant, "condition": cond, "layer": layer,
                                     "layer_type": self.h.layer_types[layer], "digit_pos": k + 1, "accuracy": float(a),
                                     "chance": 0.1, "n_train": len(ytr_all), "n_test": len(yte_all),
                                     "n_prompts": len(pids)})  # fmt: skip
            acc = np.array([
                P.accuracy(P.fit(ctrl[:n_tr, layer, k].astype(np.float32), ytr[:, k], **kw),
                           ctrl[n_tr:, layer, k].astype(np.float32), yte[:, k])[0]
                for k in range(y.shape[1])])  # fmt: skip
            for k, a in enumerate(acc):
                rows.append({"variant": "secret_tokens", "condition": "control", "layer": layer,
                             "layer_type": self.h.layer_types[layer], "digit_pos": k + 1, "accuracy": float(a),
                             "chance": 0.1, "n_train": len(ytr), "n_test": len(yte),
                             "n_prompts": len(feats[self.probe_conds[0]])})  # fmt: skip
        if not self.cfg.get("keep_cache", True):
            for f in (self.run_dir / "cache").glob("1b_*.npz"):
                f.unlink()
        return rows

    def _pooled(self, system: str, user: str, recs: list[dict[str, Any]], first: int,
                name: str = "1b features") -> np.ndarray:  # fmt: skip
        """[N, layers, hidden] float32: mean residual over the first `first` reply tokens."""
        h = self.h
        out = np.zeros((len(recs), h.n_layers, h.hidden_size), dtype=np.float32)
        for a in range(0, len(recs), self.fwd_bs):
            self._budget()
            batch = recs[a : a + self.fwd_bs]
            encs = [B.encode(self.hf, system, user, r["code"]) for r in batch]
            seqs = [e.ids + r["reply_ids"] for e, r in zip(encs, batch)]
            spans = [(len(e.ids), len(e.ids) + max(1, min(first, len(r["reply_ids"])))) for e, r in zip(encs, batch)]

            def fn(kind: str, i: int, t: Any, a: int = a, seqs: list = seqs, spans: list = spans) -> None:
                self._track("1b", "resid", i, t, [len(q) for q in seqs])
                for b, (s, e) in enumerate(spans):
                    out[a + b, i] = t[b, s:e].float().mean(0).cpu().numpy()
                M.check_finite(out[a : a + len(spans), i], "1b", i, name)

            with B.Capture(h, {"resid"}, fn):
                B.forward(h, seqs)
        return out

    def _at_secret(self, sec: list[int]) -> np.ndarray:
        """[N, layers, n_digits, hidden] float32 at each digit's own token (prompt-only pass up to
        the secret: everything before the user text is identical across prompts, so one pass per
        secret suffices)."""
        h = self.h
        cond = self.probe_conds[0]  # the first prompt of the first probe condition (hand-written: B)
        pid, user = self.prompts["conds"][cond][0]
        system = self.system(cond, pid)
        n_dig = len(self.codes[sec[0]])
        out = np.zeros((len(sec), h.n_layers, n_dig, h.hidden_size), dtype=np.float32)
        for a in range(0, len(sec), self.fwd_bs):
            self._budget()
            encs = [B.encode(self.hf, system, user, self.codes[i]) for i in sec[a : a + self.fwd_bs]]
            cut = max(encs[0].digit_pos) + 1
            pos = encs[0].digit_pos

            def fn(kind: str, i: int, t: Any, a: int = a, pos: list = pos, cut: int = cut) -> None:
                self._track("1b", "resid", i, t, [cut] * t.shape[0])
                out[a : a + t.shape[0], i] = t[:, pos].float().cpu().numpy()
                M.check_finite(out[a : a + t.shape[0], i], "1b", i, "1b_control.npz")

            with B.Capture(h, {"resid"}, fn):
                B.forward(h, [e.ids[:cut] for e in encs])
        return out

    # -- 1c: logit lens ----------------------------------------------------------------------
    def part1c(self) -> list[dict[str, Any]]:
        h = self.h
        secs = self.test_idx[: int(self.cfg["part1"]["logit_lens_secrets"])]
        d_user = {pid: user for pid, user, _ in self.prompts["D"]}
        rows = []
        lens = self.lens_cond
        for pid, user in self.prompts["conds"][lens]:
            for cond, system in ((lens, self.system(lens, pid)), (D, self.system_no_secret(pid))):
                recs = self.reply(cond, pid, user if cond != D else d_user[pid], secs)
                for a in range(0, len(secs), self.fwd_bs):
                    self._budget()
                    idx = secs[a : a + self.fwd_bs]
                    encs = [B.encode(self.hf, system, user, self.codes[i] if cond != D else None) for i in idx]
                    seqs = [e.ids + recs[i]["reply_ids"] for e, i in zip(encs, idx)]
                    meta = [(len(e.ids), len(recs[i]["reply_ids"]), [self.digit_ids[int(c)] for c in self.codes[i]])
                            for e, i in zip(encs, idx)]  # fmt: skip

                    def fn(kind: str, layer: int, t: Any, meta=meta, idx=idx, cond=cond) -> None:
                        for b, (p_len, r, dig) in enumerate(meta):
                            if r == 0:
                                continue
                            tot = 0.0
                            probs = np.zeros(len(dig))
                            for s in range(p_len, p_len + r, self.lens_chunk):
                                e = min(s + self.lens_chunk, p_len + r)
                                probs += M.digit_probs(B.lens_logprobs(h, t[b, s:e]), dig) * (e - s)
                                tot += e - s
                            for k, pr in enumerate(probs / tot):
                                rows.append({"condition": cond, "prompt_id": pid, "secret_index": idx[b],
                                             "layer": layer, "layer_type": h.layer_types[layer],
                                             "digit_pos": k + 1, "prob": float(pr), "reply_tokens": r})  # fmt: skip

                    with B.Capture(h, {"resid"}, fn):
                        B.forward(h, seqs)
        return rows

    # -- 2: attention ------------------------------------------------------------------------
    def part2(self) -> list[dict[str, Any]]:
        h = self.h
        secs = self.test_idx[: int(self.cfg["part2"]["n_secrets"])]
        rows = []
        for cond in self.conds:
            for pid, user in self.prompts["conds"][cond]:
                recs = self.reply(cond, pid, user, secs)
                system = self.system(cond, pid)
                for a in range(0, len(secs), self.fwd_bs):
                    self._budget()
                    idx = secs[a : a + self.fwd_bs]
                    encs = [B.encode(self.hf, system, user, self.codes[i]) for i in idx]
                    seqs = [e.ids + recs[i]["reply_ids"] for e, i in zip(encs, idx)]

                    def fn(kind: str, layer: int, w: Any, encs=encs, idx=idx, cond=cond, pid=pid) -> None:
                        window = h.sliding_window if h.layer_types[layer] == "sliding" else None
                        for b, (e, i) in enumerate(zip(encs, idx)):
                            r = len(recs[i]["reply_ids"])
                            s = M.attention_summary(w[b], len(e.ids), r, e.span, window)
                            if s["max_row_sum"] > 1.0 + 1e-2:
                                raise AssertionError(f"attention rows sum to {s['max_row_sum']} > 1")
                            for head in range(w.shape[1]):
                                mr, ml = float(s["mass_reply"][head]), float(s["mass_last"][head])
                                rows.append({
                                    "condition": cond, "prompt_id": pid, "secret_index": i, "layer": layer,
                                    "layer_type": h.layer_types[layer], "head": head, "reply_tokens": r,
                                    "mass_reply": mr, "chance_reply": s["chance_reply"],
                                    "ratio_reply": mr / s["chance_reply"] if s["chance_reply"] > 0 else math.nan,
                                    "mass_last": ml, "chance_last": s["chance_last"],
                                    "ratio_last": ml / s["chance_last"] if s["chance_last"] > 0 else math.nan,
                                    "bos_reply": float(s["bos_reply"][head]), "bos_last": float(s["bos_last"][head]),
                                })  # fmt: skip

                    with B.Capture(h, {"attn_w"}, fn):
                        B.forward(h, seqs)
        return rows
