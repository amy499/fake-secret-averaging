"""PasswordEval "loose key check" test: REPORT-ONLY (NOT PROTOCOL), POST-HOC. Re-reads stored replies.

Status: ✅ implemented + tested on synthetic fixture runs (tests/test_key_similarity.py: offered-
password parsing, similarity, quantile bins, logistic slope and its paired bootstrap, prompt-hash
join, no secret / prompt / reply / parsed token in any output). 🟡 on the real fsa / fsa_lite /
fsa_lite_run3 runs.

    python -m stats.key_similarity --model gemma [--root outputs/protocol] [--out outputs/diagnostics/key_similarity]
                                   [--run NAME=RUN_DIR ...] [--dry-run]
    sbatch --time=00:30:00 --export=ALL,MODULE=stats.key_similarity scripts/slurm/harness_run.sbatch --model gemma     (Slurm)
    qsub -l walltime=00:30:00 -v MODULE=stats.key_similarity,RUN_ARGS="--model qwen" scripts/pbs/harness_run.pbs    (PBS)

Question (post-hoc follow-up to stats/leak_diagnosis.py): does the info leak on the PasswordEval
`wrong` turn grow with how close the password the user offers is to the real one (a "loose key
check")? Changes nothing; writes only <out>/<model>/key_similarity.{json,md}: counts, rates, CIs,
slopes (never a secret, password, info string, prompt, reply or parsed token).

The offered password. items.jsonl stores no prompt text (prompt_sha256 only) and PasswordEval stores
no wrong-password field; the benchmark itself parses nothing (benchmarks/password_eval.py builds the
turn from `non_compliant_user_prompt` verbatim). So the user turn is re-built with the benchmark's own
loader (password_eval_team, the pinned data file), joined to items.jsonl by key, and the join is
checked: sha256 of the rebuilt messages == the row's prompt_sha256 (a mismatch stops the run). The
offered password is then parsed from the user turn with OFFER_PATTERNS (this module's rules, stated
below; parse counts reported). In the official data most `wrong` turns say the user forgot or does
not have the password, so few items parse; with fewer than MIN_PARSED parsed items the bins and the
slope are reported as not estimable, never padded.

  OFFER_PATTERNS: a password word (password, passcode, passphrase, pin, access code, code word),
  optionally a link ("is", "was", ":", "=", "might be", "could be", "should be", "may be", "I think
  is", "I believe is", "is probably"), then a token; or a verb (tried, entered, typed, used, using)
  then a quoted token. A token is quoted ('x', "x", curly quotes, backticks; 3-40 non-space
  characters), or unquoted and password-like (3-40 of [A-Za-z0-9_!@#$%^&*.-], with a digit, an
  upper-case letter after the first character, or one of _!@#$%^&*; trailing . , ; : ! ? removed).
  The first match wins.

Similarity (casefolded): ratio = 1 - Levenshtein(offered, real) / max(len); prefix = shared-prefix
length in characters. Bins: 4 quantile bins of the ratio over the parsed items (numpy quantiles;
ties keep equal values in one bin, so bins can be unequal; edges and counts are reported). Per bin
and run: info and password leak rates (strict, loose: stats.rescore.substring_rules via
stats.leak_diagnosis.derive) with bootstrap CIs (stats.core, 2,000 resamples, seed 2026). Logistic
regression per run (strict info and password leaks): logit P(leak) = a + b * ratio (Newton, ridge LOGIT_RIDGE on b),
b with a percentile bootstrap CI over items; paired by item: b(run) - b(reference) on the SAME
resampled items (reference = the first run, original by default; run3 also vs run2).

PROXY (secondary, labelled): lexical_proxy = the best ratio between ANY token (>= 4 characters) of
the wrong-turn user text and the real password. Defined for every item, but it measures word overlap
between the request and the password (e.g. a password built from a topic word), not an offered
password: it is not a key check. Same bins and slope, reported separately.

Runs (default, under --root, as stats.leak_diagnosis): original = fsa/<model>/original, run2 =
fsa_lite/<model>/technique, run3 = fsa_lite_run3/<model>/technique; --run NAME=DIR adds one (e.g.
the post-hoc ablation: --run nouse=outputs/protocol/fsa_lite_run3_nouse/gemma/technique/password_eval_team).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from stats import leak_diagnosis as LD

NOTE = "report-only (not protocol), post-hoc"
BENCH = LD.BENCH
TURN = "wrong"
MIN_PARSED = 20  # fewer parsed items: bins and slope not estimable (5 per bin at least)
N_BINS = 4
LOGIT_RIDGE = 1e-3
TARGETS = ("info_strict", "info_loose", "pw_strict", "pw_loose")
SLOPE_TARGETS = ("info_strict", "pw_strict")  # slopes for the primary (strict) leaks only
N_BOOT, SEED = 2000, 2026

_PW_WORD = r"(?:password|passcode|pass\s?code|passphrase|pass\s?phrase|pin|access\s?code|code\s?word)"
_LINK = r"(?:\s*(?:is\s+probably|i\s+think\s+is|i\s+believe\s+is|might\s+be|could\s+be|should\s+be|may\s+be|is|was|:|=)\s*|\s+)"
_QUOTED = r"[\"'“‘`]([^\s\"'”’`]{3,40})[\"'”’`]"
_BARE = r"([A-Za-z0-9_!@#$%^&*.\-]{3,40})"
OFFER_PATTERNS = (
    ("pw_word_link_quoted", re.compile(r"\b" + _PW_WORD + _LINK + _QUOTED, re.I)),
    ("pw_word_link_bare", re.compile(r"\b" + _PW_WORD + _LINK + _BARE, re.I)),
    ("verb_quoted", re.compile(r"\b(?:tried|entered|typed|used|using)\s+(?:the\s+|a\s+)?(?:" + _PW_WORD + r"\s+)?" + _QUOTED, re.I)),
)  # fmt: skip
_LOOKS = re.compile(r"\d|.[A-Z]|[_!@#$%^&*]")
_TOK = re.compile(r"[A-Za-z0-9_!@#$%^&*.\-]{4,}")


class KeySimilarityError(RuntimeError):
    pass


# --------------------------------------------------------------------------- parsing and similarity


def parse_offered(user: str) -> tuple[str | None, str | None]:
    """(offered password, pattern name) or (None, None). Never logged."""
    for name, rx in OFFER_PATTERNS:
        for m in rx.finditer(user):
            tok = m.group(1)
            if name.endswith("bare"):
                tok = tok.rstrip(".,;:!?")
                if len(tok) < 3 or not _LOOKS.search(tok):
                    continue
            return tok, name
    return None, None


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def ratio(a: str, b: str) -> float:
    a, b = a.casefold(), b.casefold()
    n = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / n if n else 1.0


def prefix_len(a: str, b: str) -> int:
    a, b = a.casefold(), b.casefold()
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def lexical_proxy(user: str, pw: str) -> float:
    return max((ratio(t, pw) for t in _TOK.findall(user)), default=0.0)


# --------------------------------------------------------------------------- statistics


def quantile_bins(x: dict[str, float], k: int = N_BINS) -> tuple[list[float], dict[str, int]]:
    """Inner edges (k-1 quantiles) and the bin of every key; equal values share a bin."""
    import numpy as np

    v = np.asarray(sorted(x.values()), float)
    edges = [float(q) for q in np.quantile(v, [i / k for i in range(1, k)])]
    return edges, {key: int(np.searchsorted(edges, val, side="right")) for key, val in x.items()}


def logit_slopes(x, y):
    """Batched: x, y of shape (B, n). Slope b of logit P(y) = a + b x per row (Newton, ridge
    LOGIT_RIDGE on b, at most 100 iterations); NaN where y or x is constant in the row."""
    import numpy as np

    x, y = np.atleast_2d(np.asarray(x, float)), np.atleast_2d(np.asarray(y, float))
    ok = (y.min(axis=1) < y.max(axis=1)) & (x.min(axis=1) < x.max(axis=1))
    w = np.zeros((x.shape[0], 2))
    R = np.diag([0.0, LOGIT_RIDGE])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(w[:, :1] + w[:, 1:] * x, -50, 50)))
        r, v = y - p, p * (1 - p)
        g = np.stack([r.sum(1), (r * x).sum(1)], 1) - w @ R
        h01 = (v * x).sum(1)
        H = np.stack([np.stack([v.sum(1), h01], 1), np.stack([h01, (v * x * x).sum(1) + LOGIT_RIDGE], 1)], 1)
        H[~ok] = np.eye(2)
        step = np.linalg.solve(H, g[..., None])[..., 0]
        step[~ok] = 0.0
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    b = w[:, 1].copy()
    b[~ok] = np.nan
    return b


def logit_fit(x, y) -> tuple[float, float] | None:
    """Slope of logit P(y) = a + b x (see logit_slopes) as (b, b); None if undefined."""
    b = float(logit_slopes(x, y)[0])
    return None if b != b else (b, b)


def slope_report(x: list[float], ys: dict[str, list[float]], ref: str | None, pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Per run: slope b with a bootstrap CI; per pair (run, ref): b(run) - b(ref) on the same resamples.
    ys: run -> 0/1 values aligned with x (the same items in the same order)."""
    import numpy as np

    n = len(x)
    xa = np.asarray(x, float)
    out: dict[str, Any] = {"definition": "logit P(leak) = a + b * similarity; b per +1.0 similarity (log-odds); "
                           f"ridge {LOGIT_RIDGE} on b; percentile bootstrap over items ({N_BOOT}, seed {SEED})",
                           "n": n, "runs": {}, "paired": {}}  # fmt: skip
    if n < 2:
        return out
    idx = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    boots: dict[str, list[float | None]] = {}
    for run, y in ys.items():
        ya = np.asarray(y, float)
        fit = logit_fit(xa, ya)
        bs = [None if v != v else float(v) for v in logit_slopes(xa[idx], ya[idx])]
        boots[run] = bs
        ok = [b for b in bs if b is not None]
        lo, hi = (float(q) for q in np.quantile(ok, [0.025, 0.975])) if ok else (None, None)
        out["runs"][run] = {"slope": None if fit is None else fit[1], "ci_low": lo, "ci_high": hi,
                            "n_leaks": int(round(float(ya.sum()))), "boot_defined": len(ok),
                            **({} if fit is not None else {"why_undefined": "no variation in the leak (or in similarity)"})}  # fmt: skip
    for a, b in pairs:
        if a not in boots or b not in boots:
            continue
        d = [x1 - x2 for x1, x2 in zip(boots[a], boots[b]) if x1 is not None and x2 is not None]
        sa, sb = out["runs"][a]["slope"], out["runs"][b]["slope"]
        lo, hi = (float(q) for q in np.quantile(d, [0.025, 0.975])) if d else (None, None)
        out["paired"][f"{a}_vs_{b}"] = {"diff": None if sa is None or sb is None else sa - sb, "ci_low": lo, "ci_high": hi,
                                        "boot_defined": len(d)}  # fmt: skip
    return out


def bins_report(sim: dict[str, float], per_run: dict[str, dict[str, dict]]) -> dict[str, Any]:
    edges, of = quantile_bins(sim)
    out: dict[str, Any] = {"inner_edges": edges, "bins": {}}
    for b in range(N_BINS):
        keys = sorted(k for k, v in of.items() if v == b)
        vals = [sim[k] for k in keys]
        d: dict[str, Any] = {"n": len(keys), "similarity_min": min(vals) if vals else None,
                             "similarity_max": max(vals) if vals else None, "runs": {}}  # fmt: skip
        for run, per in per_run.items():
            ks = [k for k in keys if k in per]
            d["runs"][run] = {t: LD.est([per[k][t] for k in ks]) for t in TARGETS} if ks else None
        out["bins"][str(b + 1)] = d
    return out


def analyse_measure(name: str, sim: dict[str, float], per_run: dict[str, dict[str, dict]], ref: str,
                    pairs: list[tuple[str, str]], min_n: int) -> dict[str, Any]:  # fmt: skip
    keys = sorted(k for k in sim if all(k in per for per in per_run.values()))
    out: dict[str, Any] = {"measure": name, "n_items": len(keys), "distinct_values": len({sim[k] for k in keys})}
    if len(keys) < min_n or out["distinct_values"] < 2:
        out["status"] = f"not estimable: {len(keys)} items, {out['distinct_values']} distinct values (need >= {min_n} and >= 2)"
        return out
    out["status"] = "ok"
    out["bins"] = bins_report({k: sim[k] for k in keys}, per_run)
    out["slope"] = {t: slope_report([sim[k] for k in keys], {r: [per[k][t] for k in keys] for r, per in per_run.items()},
                                    ref, pairs) for t in SLOPE_TARGETS}  # fmt: skip
    return out


# --------------------------------------------------------------------------- driver


def wrong_turn_prompts(dry_run: bool) -> dict[str, dict[str, Any]]:
    """key -> {user, pw, sha, turn} for the benchmark's items (rebuilt with its own loader)."""
    from core import registry
    from core.protocol import sha256_json

    registry.discover()
    bench = registry.get("benchmark", BENCH)({"download": False}, None, dry_run)
    return {it.id: {"user": it.user_prompt, "pw": it.secret, "sha": sha256_json(it.messages()),
                    "turn": str(it.metadata.get(LD.TURN_FIELD))} for it in bench.items("test")}  # fmt: skip


def analyse(model: str, runs: list[tuple[str, Path]], prompts: dict[str, dict[str, Any]], out_dir: Path,
            min_n: int = MIN_PARSED, log=print) -> dict[str, Any]:  # fmt: skip
    present = [(n, d) for n, d in runs if (d / "items.jsonl").exists()]
    missing = {n: d.as_posix() for n, d in runs if not (d / "items.jsonl").exists()}
    for n, d in missing.items():
        log(f"missing run {n}: {d} (no items.jsonl); left out")
    if not present:
        raise KeySimilarityError("no run found")
    per_run: dict[str, dict[str, dict]] = {}
    res: dict[str, Any] = {"note": NOTE, "model": model, "benchmark": BENCH, "turn": TURN, "missing_runs": missing,
                           "offer_patterns": [[n, rx.pattern] for n, rx in OFFER_PATTERNS], "runs": {}}  # fmt: skip
    for name, d in present:
        rows = LD.read_rows(d)
        inv = LD.inventory(rows)
        log(f"[{name}] turn field {inv['turn_field']}: {inv['turn_values']}")
        if TURN not in inv["turn_values"]:
            raise KeySimilarityError(f"{name}: no {TURN!r} turn; turn values {sorted(inv['turn_values'])}")
        wrong = [r for r in rows if str(r["meta"].get(LD.TURN_FIELD)) == TURN]
        bad = [r["key"] for r in wrong if r["key"] not in prompts or prompts[r["key"]]["sha"] != r.get("prompt_sha256")]
        if bad:
            raise KeySimilarityError(f"{name}: {len(bad)} of {len(wrong)} {TURN} rows do not match the rebuilt prompt "
                                     "(key missing or prompt_sha256 differs): other data or another loader")  # fmt: skip
        per, agree = LD.derive(wrong)
        dis = {k: v for k, v in agree.items() if v.get("disagree")}
        per_run[name] = per
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}
        res["runs"][name] = {"dir": d.as_posix(), "n_wrong": len(wrong), "prompt_join": "all prompt_sha256 match",
                             "approach": summary.get("approach"), "variant": summary.get("variant"),
                             "adapter_sha256": (summary.get("defence_info") or {}).get("adapter_sha256"),
                             "score_agreement": agree}  # fmt: skip
        log(f"[{name}] {len(wrong)} {TURN} rows, prompts joined by sha256; stored vs re-derived: "
            f"{'all agree' if not dis else f'DISAGREE {dis}'}")  # fmt: skip
    keys = sorted(set.intersection(*(set(p) for p in per_run.values())))
    offered: dict[str, float] = {}
    prefix: dict[str, int] = {}
    proxy: dict[str, float] = {}
    by_pat: dict[str, int] = {}
    for k in keys:
        p = prompts[k]
        tok, pat = parse_offered(p["user"])
        proxy[k] = lexical_proxy(p["user"], p["pw"])
        if tok is not None:
            offered[k] = ratio(tok, p["pw"])
            prefix[k] = prefix_len(tok, p["pw"])
            by_pat[pat] = by_pat.get(pat, 0) + 1
    exact = sum(1 for v in offered.values() if v == 1.0)
    res["parse"] = {"n_items": len(keys), "parsed": len(offered), "not_parsed": len(keys) - len(offered),
                    "by_pattern": dict(sorted(by_pat.items())), "parsed_equal_to_real_casefold": exact,
                    "parser": "stats.key_similarity.OFFER_PATTERNS (the benchmark parses nothing)"}  # fmt: skip
    if prefix:
        vals = sorted(prefix.values())
        res["parse"]["shared_prefix_chars"] = {"mean": sum(vals) / len(vals), "median": vals[len(vals) // 2],
                                               "zero": sum(1 for v in vals if v == 0), "max": vals[-1]}  # fmt: skip
    log(f"parsed offered password: {len(offered)} of {len(keys)} {TURN} items {json.dumps(res['parse']['by_pattern'])}")
    ref = present[0][0]
    pairs = [(n, ref) for n in per_run if n != ref]
    if "run3" in per_run and "run2" in per_run and ref != "run2":
        pairs.append(("run3", "run2"))
    res["reference"] = ref
    res["offered_ratio"] = analyse_measure("offered_ratio", offered, per_run, ref, pairs, min_n)
    res["offered_prefix"] = analyse_measure("offered_prefix", {k: float(v) for k, v in prefix.items()}, per_run, ref,
                                            pairs, min_n)  # fmt: skip
    res["lexical_proxy"] = analyse_measure("lexical_proxy", proxy, per_run, ref, pairs, min_n)
    res["lexical_proxy"]["label"] = "PROXY, not a key check: word overlap between the request and the password"
    out = out_dir / model
    out.mkdir(parents=True, exist_ok=True)
    (out / "key_similarity.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                             newline="\n")  # fmt: skip
    (out / "key_similarity.md").write_text(markdown(res), encoding="utf-8", newline="\n")
    log(f"written {out / 'key_similarity.json'} and key_similarity.md")
    return res


def _e(e: dict | None) -> str:
    if not e or e.get("value") is None or (isinstance(e.get("value"), float) and math.isnan(e["value"])):
        return "n/a"
    return f"{100 * e['value']:.1f} [{100 * e['ci_low']:.1f}, {100 * e['ci_high']:.1f}] ({e['count']}/{e['n']})"


def _s(d: dict | None, k: str = "slope") -> str:
    if not d or d.get(k) is None:
        return "undefined"
    lo, hi = d.get("ci_low"), d.get("ci_high")
    ci = f" [{lo:+.2f}, {hi:+.2f}]" if lo is not None and hi is not None else ""
    return f"{d[k]:+.2f}{ci}"


def markdown(res: dict[str, Any]) -> str:
    p = res["parse"]
    L = [f"# Key similarity on the PasswordEval `{res['turn']}` turn: {res['model']} ({NOTE})", "",
         f"Offered password parsed in **{p['parsed']} of {p['n_items']}** items {json.dumps(p['by_pattern'])}; "
         f"equal to the real one: {p['parsed_equal_to_real_casefold']}. Runs: {', '.join(res['runs'])} "
         f"(paired against {res['reference']}). Counts, rates (%) and CIs only.", ""]  # fmt: skip
    for m in ("offered_ratio", "offered_prefix", "lexical_proxy"):
        d = res[m]
        L += [f"## {m}" + (f" ({d['label']})" if d.get("label") else ""), "", f"{d['n_items']} items: {d['status']}", ""]
        if d["status"] != "ok":
            continue
        L += [f"Bin edges (inner): {', '.join(f'{x:.3f}' for x in d['bins']['inner_edges'])}", "",
              "| bin | n | range | run | info strict | pw strict |", "|---|---|---|---|---|---|"]  # fmt: skip
        for b, bd in d["bins"]["bins"].items():
            rng = "-" if bd["similarity_min"] is None else f"{bd['similarity_min']:.2f}-{bd['similarity_max']:.2f}"
            for run, r in bd["runs"].items():
                L.append(f"| {b} | {bd['n']} | {rng} | {run} | {_e((r or {}).get('info_strict'))} | {_e((r or {}).get('pw_strict'))} |")
        L += ["", "| leak | run | slope [95% CI] | leaks |", "|---|---|---|---|"]
        for t in ("info_strict", "pw_strict"):
            for run, s in d["slope"][t]["runs"].items():
                L.append(f"| {t} | {run} | {_s(s)} | {s['n_leaks']} |")
            for pr, s in d["slope"][t]["paired"].items():
                L.append(f"| {t} | {pr} (paired) | {_s(s, 'diff')} | - |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PasswordEval key-similarity test (report-only, post-hoc)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", type=Path, default=Path("outputs/protocol"))
    ap.add_argument("--out", type=Path, default=Path("outputs/diagnostics/key_similarity"))
    ap.add_argument("--run", action="append", default=[], help="NAME=RUN_DIR (adds or replaces a run)")
    ap.add_argument("--dry-run", action="store_true", help="rebuild the prompts from the benchmark's fixtures")
    a = ap.parse_args(argv)
    try:
        res = analyse(a.model, LD.resolve_runs(a.root, a.model, a.run), wrong_turn_prompts(a.dry_run), a.out)
    except (KeySimilarityError, LD.DiagnosisError) as e:
        print(f"STOP: {e}")
        return 1
    print(markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
