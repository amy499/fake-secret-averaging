"""FSA-lite contamination check: no training text may overlap the harness evaluation data.

Status: ✅ implemented + tested on CPU (selftest_v2.py lite tests: injected 13-gram and secret
overlaps fail); 🟡 not yet run on the Slurm cluster's copy of the harness data.

Evaluation data = the files the harness FETCH step pinned and verified, read from
$HARNESS_DATA/url/protocol/<name>/<file> (harness core/data.py MANIFEST); their hash prefixes are
checked again here and any mismatch is an error:
  password_eval_test  system / user / compliant / non-compliant prompts (13-grams);
                      password and confidential_info (secrets, see below)
  confaide_tier4      the whole tier_4.txt (13-grams)
  mmlu_test           all 14,042 rows, question + choices (13-grams)
  ifeval_input        all 541 prompts (13-grams)
  tt_extraction, tt_hijacking   pre_prompt, attack, post_prompt (13-grams); access_code (secret)

Overlap rules (lite_common):
  13-gram   a training text shares any word 13-gram (lowercased, whitespace-normalised) with an
            evaluation text
  secret    every evaluation secret (PasswordEval password or confidential_info, Tensor Trust access
            code; >= 4 characters) is matched EXACTLY (case-insensitive) against every training
            secret and fake-pool value; a DISTINCTIVE one is also matched as a word-bounded,
            case-insensitive substring of every training text. Distinctive = not made only of
            stopwords and (>= 8 characters, or containing a digit, space or symbol). Generic
            evaluation codes such as "access granted", "hello", "mother" occur in ordinary text and
            in most Tensor Trust defences; matching them as substrings would flag text that shares
            nothing with the evaluation items, so they count only as exact secret collisions.
  Secrets shorter than 4 characters are not matched (counted in the report).

Training texts = every message of every unit (train AND dev) in units.jsonl; training secrets =
every unit's secret(s) plus every value in pools.json (fakes are written into training contexts).

  python src/fsa_lite/lite/contam_check.py --data <built dir> --harness-data $HARNESS_DATA
exits 1 and prints the first overlaps when anything overlaps; writes <built dir>/contam_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lite_common as LC  # noqa: E402

MIN_SECRET = 4

# name -> (file name, algo, expected prefix), as harness core/data.py MANIFEST (first check of each).
EVAL_FILES = {
    "password_eval_test": ("test-00000-of-00001.parquet", "sha256", "4d537a362ec9dcc4"),
    "confaide_tier4": ("tier_4.txt", "sha256", "cf0ba2d5f4d6c136"),
    "mmlu_test": ("test-00000-of-00001.parquet", "md5", "4d0d6cea"),
    "ifeval_input": ("input_data.jsonl", "md5", "4c2c4325"),
    "tt_extraction": ("extraction_robustness_dataset.jsonl", "sha256", "4090f9f57b6df69c"),
    "tt_hijacking": ("hijacking_robustness_dataset.jsonl", "sha256", "2ad3c2343bbf43cb"),
}


def distinctive(s: str) -> bool:
    """An evaluation secret specific enough to search for inside running text."""
    words = [w.strip(".,;:!?'\"()[]{}") for w in s.lower().split()]
    if all(w in LC.STOPWORDS or not w for w in words):
        return False
    return len(s) >= 8 or any(not c.isalpha() for c in s)


@dataclass
class EvalIndex:
    ngrams: dict[str, str] = field(default_factory=dict)  # 13-gram -> first source
    secrets: dict[str, str] = field(default_factory=dict)  # lowercased distinctive secret -> source (text match)
    exact: dict[str, str] = field(default_factory=dict)  # every lowercased secret >= 4 chars -> source (equality)
    tt_codes: set[str] = field(default_factory=set)  # lowercased TT eval access codes
    tt_instr: set[str] = field(default_factory=set)  # norm() of TT eval pre / post prompts (>= 20 chars)
    tt_attacks: set[str] = field(default_factory=set)  # norm() of TT eval attacks
    files: dict[str, dict] = field(default_factory=dict)
    n_short_secrets: int = 0
    _secret_re: Any = None

    def add_text(self, text: str, source: str) -> None:
        for g in LC.ngrams(text):
            self.ngrams.setdefault(g, source)

    def add_secret(self, s: Any, source: str) -> None:
        s = str(s or "").strip()
        if len(s) < MIN_SECRET:
            self.n_short_secrets += 1 if s else 0
            return
        self.exact.setdefault(s.lower(), source)
        if distinctive(s):
            self.secrets.setdefault(s.lower(), source)
            self._secret_re = None

    def is_eval_secret(self, value: str) -> bool:
        """value (a training secret or fake) equals an evaluation secret, any case."""
        return str(value).strip().lower() in self.exact

    def secret_re(self):
        if self._secret_re is None:
            alts = "|".join(re.escape(s) for s in sorted(self.secrets, key=len, reverse=True))
            self._secret_re = re.compile(rf"(?<![A-Za-z0-9_])(?:{alts})(?![A-Za-z0-9_])", re.IGNORECASE) if alts else None
        return self._secret_re

    def hits(self, text: str) -> list[tuple[str, str, str]]:
        """[(kind, source, matched)] for one training text: 13-gram and secret overlaps."""
        out = [("ngram13", self.ngrams[g], g) for g in LC.ngrams(text) if g in self.ngrams]
        rx = self.secret_re()
        if rx is not None:
            out += [("secret", self.secrets[m.group(0).lower()], m.group(0)) for m in rx.finditer(text)]
        return out

    def overlaps(self, text: str) -> bool:
        if any(g in self.ngrams for g in LC.ngrams(text)):
            return True
        rx = self.secret_re()
        return rx is not None and rx.search(text) is not None


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def load_eval(harness_data: Path, *, verify: bool = True) -> EvalIndex:
    """The evaluation index from the harness's pinned files. verify=False only for CPU tests on
    synthetic files (the report then says verified: false)."""
    idx = EvalIndex()
    root = Path(harness_data) / "url" / "protocol"
    for name, (fname, algo, prefix) in EVAL_FILES.items():
        p = root / name / fname
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run the harness FETCH step (python -m core.data fetch) first")
        got = _digest(p, algo)
        if verify and not got.startswith(prefix):
            raise RuntimeError(f"{p}: {algo} {got[:16]} != pinned {prefix} (not the file the harness verified)")
        idx.files[name] = {"path": str(p), "algo": algo, "digest": got, "expected_prefix": prefix,
                           "verified": bool(verify and got.startswith(prefix))}  # fmt: skip
    for r in _parquet(root / "password_eval_test" / EVAL_FILES["password_eval_test"][0]):
        for k in ("system_prompt", "user_prompt", "compliant_user_prompt", "non_compliant_user_prompt"):
            idx.add_text(r.get(k) or "", "password_eval")
        idx.add_secret(r.get("password"), "password_eval:password")
        idx.add_secret(r.get("confidential_info"), "password_eval:confidential_info")
    idx.add_text((root / "confaide_tier4" / "tier_4.txt").read_text(encoding="utf-8"), "confaide_tier4")
    for r in _parquet(root / "mmlu_test" / EVAL_FILES["mmlu_test"][0]):
        ch = r.get("choices")
        ch = " ".join(map(str, ch)) if isinstance(ch, list | tuple) else str(ch)
        idx.add_text(f"{r.get('question', '')} {ch}", "mmlu")
    for r in LC.read_jsonl(root / "ifeval_input" / "input_data.jsonl"):
        idx.add_text(r["prompt"], "ifeval")
    for name in ("tt_extraction", "tt_hijacking"):
        for r in LC.read_jsonl(root / name / EVAL_FILES[name][0]):
            for k in ("pre_prompt", "attack", "post_prompt"):
                idx.add_text(r.get(k) or "", name)
            idx.add_secret(r.get("access_code"), f"{name}:access_code")
            code = str(r.get("access_code") or "").strip().lower()
            if code:
                idx.tt_codes.add(code)
            for k in ("pre_prompt", "post_prompt"):
                n = LC.norm(r.get(k) or "")
                if len(n) >= 20:
                    idx.tt_instr.add(n)
            if r.get("attack"):
                idx.tt_attacks.add(LC.norm(r["attack"]))
    return idx


def training_texts(units: Iterable[dict], pools: dict[str, list[str]]) -> Iterable[tuple[str, str]]:
    for u in units:
        for i, m in enumerate(u["messages"]):
            yield f"{u['uid']}#{i}", m["content"]
    for name, vals in pools.items():
        for j, v in enumerate(vals):
            yield f"pool:{name}#{j}", v


def training_secrets(units: Iterable[dict], pools: dict[str, list[str]]) -> Iterable[tuple[str, str]]:
    for u in units:
        for s in {*(u.get("secrets") or []), *([u["secret"]] if u.get("secret") else [])}:
            yield f"{u['uid']}#secret", s
    for name, vals in pools.items():
        for j, v in enumerate(vals):
            yield f"pool:{name}#{j}", v


def check(units: list[dict], pools: dict[str, list[str]], idx: EvalIndex, max_examples: int = 20) -> dict:
    n_texts, n_secrets, by_kind, examples = 0, 0, {}, []

    def add(where: str, kind: str, source: str, matched: str) -> None:
        key = f"{kind}:{source.split(':')[0]}"
        by_kind[key] = by_kind.get(key, 0) + 1
        if len(examples) < max_examples:
            examples.append({"where": where, "kind": kind, "source": source, "match": matched[:120]})

    for where, text in training_texts(units, pools):
        n_texts += 1
        for kind, source, matched in idx.hits(text):
            add(where, kind, source, matched)
    for where, s in training_secrets(units, pools):
        n_secrets += 1
        if idx.is_eval_secret(s):
            add(where, "secret_equal", idx.exact[str(s).strip().lower()], s)
    return {"result": "PASS" if not by_kind else "FAIL", "texts_checked": n_texts, "secrets_checked": n_secrets,
            "overlaps": by_kind, "examples": examples, "eval_ngrams": len(idx.ngrams),
            "eval_secrets": len(idx.exact), "eval_secrets_text_matched": len(idx.secrets),
            "eval_secrets_too_short_to_match": idx.n_short_secrets, "ngram_n": LC.NGRAM,
            "eval_files": idx.files}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="build_lite.py output folder")
    ap.add_argument("--harness-data", type=Path, required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    idx = load_eval(a.harness_data)
    units = LC.read_jsonl(a.data / "units.jsonl")
    pools = json.loads((a.data / "pools.json").read_text(encoding="utf-8"))
    rep = check(units, pools, idx)
    rep["units_sha256"] = LC.sha256_file(a.data / "units.jsonl")
    rep["pools_sha256"] = LC.sha256_file(a.data / "pools.json")
    (a.data / "contam_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"contamination: {rep['result']} ({rep['texts_checked']} training texts vs {rep['eval_ngrams']} eval "
          f"13-grams and {rep['eval_secrets']} eval secrets) {rep['overlaps'] or ''}")  # fmt: skip
    for e in rep["examples"]:
        print("  OVERLAP", e)
    return 0 if rep["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
