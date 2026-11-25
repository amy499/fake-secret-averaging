"""Held-out secret guard for the run3 data (build_lite_run3.py --exclude-secrets <harness held_out_secrets.txt>).

Status: ✅ implemented + tested on CPU with synthetic held-out lists (selftest_run3.py); 🟡 not yet run with
the harness file (cluster rebuild jobs).

Why: the harness's secret-in-context suite evaluates on experiments/_bridges/private/held_out_secrets.txt and
refuses an approach whose used-secrets file shares a line with it. contam_check compares eval n-grams and
eval-file secrets, never this file, so a short random value (4 digits) of the first run3 build collided
with a held-out secret by chance.

A value is REJECTED when, after strip(), it
  exact      equals a held-out secret,
  casefold   equals one ignoring case (str.casefold),
  contains   contains a held-out secret of >= MIN_SUB characters as a substring (casefolded), or
  contained  is itself >= MIN_SUB characters and a substring of a held-out secret (casefolded).
A fragment of a value (the FSA token weights' partial matches, >= lite_common.PARTIAL_MIN = 4 characters) is
a substring of that value, so "contains" on the value covers every fragment that could equal a held-out
secret. Near-misses derived at TRAINING time from fake passwords (lite_source: the same kgp_data.derive edit
of each fake) are values too: derived_near_misses() lists them so the build can check them.

Nothing here prints or stores a held-out secret: only the file's sha256, counts and reasons.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MIN_SUB = 4


class HeldOutOverlap(RuntimeError):
    """The built data shares a value with the held-out secrets (counts by category in .counts)."""

    def __init__(self, counts: dict[str, dict[str, int]]) -> None:
        self.counts = counts
        super().__init__(f"held-out secrets overlap the run3 data (counts only, by category and reason): {counts}")


class HeldOut:
    def __init__(self, values: Iterable[str], sha256: str | None = None) -> None:
        vals = sorted({str(v).strip() for v in values if v is not None and str(v).strip()})
        self.n, self.sha256 = len(vals), sha256
        self._exact = set(vals)
        self._cf = {v.casefold() for v in vals}
        self._long = sorted({v.casefold() for v in vals if len(v) >= MIN_SUB}, key=len, reverse=True)
        self._rx = re.compile("|".join(map(re.escape, self._long))) if self._long else None
        self.rejected: dict[str, dict[str, int]] = {}  # kind -> reason -> count (filled by keep())

    @classmethod
    def from_file(cls, path: Path) -> HeldOut:
        data = Path(path).read_bytes()
        lines = data.decode("utf-8").splitlines()
        h = cls(lines, hashlib.sha256(data).hexdigest())
        if h.n == 0:
            raise ValueError(f"{path}: no held-out secrets in the file")
        return h

    def reason(self, value: Any) -> str | None:
        s = str(value or "").strip()
        if not s:
            return None
        if s in self._exact:
            return "exact"
        c = s.casefold()
        if c in self._cf:
            return "casefold"
        if self._rx is not None and self._rx.search(c):
            return "contains"
        if len(c) >= MIN_SUB and any(c in h for h in self._long):
            return "contained"
        return None

    def in_text(self, text: str) -> bool:
        """text holds a held-out secret (casefolded substring; secrets shorter than MIN_SUB: word-bounded)."""
        c = str(text).casefold()
        if self._rx is not None and self._rx.search(c):
            return True
        short = [s for s in self._cf if len(s) < MIN_SUB]
        return any(re.search(rf"(?<![a-z0-9_]){re.escape(s)}(?![a-z0-9_])", c) for s in short)

    def keep(self, value: Any, kind: str) -> bool:
        """True if value may be used; a rejection is counted under kind (counts only)."""
        r = self.reason(value)
        if r is not None:
            k = self.rejected.setdefault(kind, {})
            k[r] = k.get(r, 0) + 1
        return r is None

    def record(self) -> dict:
        return {"file_sha256": self.sha256, "n": self.n, "min_substring": MIN_SUB,
                "rejected": {k: dict(sorted(v.items())) for k, v in sorted(self.rejected.items())}}  # fmt: skip


def unit_values(u: dict) -> list[str]:
    """Every secret-like value a unit carries: secret(s), password, payload, slot values (near-misses)."""
    out = [u.get("secret"), u.get("password"), u.get("payload"), *(u.get("secrets") or [])]
    out += [sl.get("value") for sl in u.get("slots") or []]
    return [str(v) for v in out if v]


def derived_near_misses(pool: Iterable[str]) -> set[str]:
    """The near-miss of every pool password under every kgp_data.derive edit (fake near-misses at training time)."""
    from kgp_data import DERIVES, derive

    return {derive(k, p) for p in pool for k in DERIVES}


def overlap_counts(held: HeldOut, units: list[dict], pools: dict[str, list[str]], used: Iterable[str],
                   derived: Iterable[str] = ()) -> dict[str, dict[str, int]]:  # fmt: skip
    """{category: {reason: count}} of values that a held-out secret matches (empty dict: clean)."""
    out: dict[str, dict[str, int]] = {}

    def add(cat: str, vals: Iterable[str]) -> None:
        for v in vals:
            r = held.reason(v)
            if r is not None:
                out.setdefault(cat, {}).setdefault(r, 0)
                out[cat][r] += 1

    add("unit_values", {v for u in units for v in unit_values(u)})
    add("fake_pools", {v for vs in pools.values() for v in vs})
    add("used_secrets", set(used))
    add("derived_fake_near_misses", set(derived))
    return out


def assert_clean(held: HeldOut, units: list[dict], pools: dict[str, list[str]], used: Iterable[str],
                 derived: Iterable[str] = ()) -> dict:  # fmt: skip
    c = overlap_counts(held, units, pools, used, derived)
    if c:
        raise HeldOutOverlap(c)
    return {"result": "PASS", "categories_checked": ["unit_values", "fake_pools", "used_secrets", "derived_fake_near_misses"]}


def wrong_turn_problems(units: list[dict]) -> dict[str, int]:
    """kgp_wrong turns that hold the real password: word-bounded (any wrong kind), or as a plain casefolded
    substring for every kind except add_digit (whose near-miss is the password plus a digit by design, so the
    password is its prefix but never a word-bounded match). Counts by wrong kind; empty: clean."""
    import lite_common as LC

    out: dict[str, int] = {}
    for u in units:
        if u.get("turn_type") != "kgp_wrong":
            continue
        pw, user = u["password"], u["messages"][-1]["content"]
        kind = (u.get("meta") or {}).get("wrong") or "?"
        if LC.contains_wb(user, pw) or (kind != "add_digit" and pw.casefold() in user.casefold()):
            out[kind] = out.get(kind, 0) + 1
    return out


def check_dir(data: Path, held: HeldOut) -> dict:
    """The build's final checks on an already built folder (units.jsonl, pools.json, used_secrets.txt)."""
    import json

    units = [json.loads(x) for x in (Path(data) / "units.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    pools = json.loads((Path(data) / "pools.json").read_text(encoding="utf-8"))
    used = [x for x in (Path(data) / "used_secrets.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"overlaps": overlap_counts(held, units, pools, used, derived_near_misses(pools.get("kgp_pw", []))),
            "wrong_turns": wrong_turn_problems(units), "n_used": len(used)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    """python src/fsa_lite/lite/heldout.py check --data <built_run3> --exclude-secrets <held_out_secrets.txt>
    Exit 0 and "HELD-OUT CHECK: PASS" when clean; exit 1 otherwise (counts only, never a value)."""
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--exclude-secrets", type=Path, required=True)
    a = ap.parse_args(argv)
    held = HeldOut.from_file(a.exclude_secrets)
    r = check_dir(a.data, held)
    ok = not r["overlaps"] and not r["wrong_turns"]
    print(f"HELD-OUT CHECK: {'PASS' if ok else 'FAIL'} ({a.data}: {r['n_used']} used secrets vs {held.n} held-out, "
          f"file sha256 {held.sha256[:16]}; overlaps {r['overlaps'] or 0}; wrong-password turns holding the "
          f"password {r['wrong_turns'] or 0})")  # fmt: skip
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
