"""Pinned data files for the team protocol (docs/eval_protocol.md): where each file comes
from, at which upstream revision, and the hashes it must match.

Status: 🟡 implemented; hashing, the canonical-jsonl rule and the checks are unit-tested on
generated files (tests/test_data.py); the real files have not been downloaded from this repo yet,
so no real hash has been checked here.

    python -m core.data fetch [NAME ...]        # download (all protocol files by default)
    python -m core.data verify [NAME ...]       # hash table: MATCH / MISMATCH / MISSING

Files live under $HARNESS_DATA/url/protocol/<name>/ (core.io.url_file; never committed).

Hashes and where the expected values come from:
  * "protocol": docs/eval_protocol.md (first 16 hex characters of sha256):
      ConfAIde tier_4.txt cf0ba2d5f4d6c136, GSM8K test ca6e7c5108b7ab7b, ARC-Challenge test
      b90e27562e002b98. The GSM8K and ARC values are of a *converted* file, not of the published
      parquet: one json.dumps(row, ensure_ascii=False) per row, CRLF line endings, final CRLF (the
      conversion the team protocol's reference files were made with). `canonical_sha256` computes
      those bytes in memory from the parquet rows, so no converted file is written.
  * "team": pins of the AI6130 Group 11 team protocol (Kaur et al., 2026): md5 prefixes of the
      MMLU and IFEval inputs and raw-file sha256 prefixes of the other downloads.
  * "own": values recorded for this repository: the converted PasswordEval jsonl and the
      nltk_data punkt_tab zip.
Upstream revisions are the ones that gave these hashes when the files were fetched on 2026-11-14.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HF = "https://huggingface.co/datasets"
GH = "https://raw.githubusercontent.com"


@dataclass(frozen=True)
class Check:
    """One expected hash prefix of a data file."""

    kind: str  # "sha256" (raw bytes), "md5" (raw bytes) or "canonical_sha256" (CRLF jsonl of the parquet rows)
    prefix: str
    source: str  # "protocol" | "team" | "own"
    where: str  # file:line or document the expected value comes from


@dataclass(frozen=True)
class DataFile:
    name: str
    url: str
    revision: str
    upstream: str  # repo the revision belongs to
    checks: tuple[Check, ...]
    rows: int | None = None  # expected data rows (jsonl lines or parquet rows); None = not a table
    licence: str = ""

    @property
    def basename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def _hf(repo: str, rev: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/{rev}/{path}"


def _gh(repo: str, rev: str, path: str) -> str:
    return f"{GH}/{repo}/{rev}/{path}"


MMLU_REV = "c30699e8356da336a370243923dbaf21066bb9fe"  # HF cais/mmlu
GR_REV = "e6890f85757dd84e27ca6df2dd30651dafad28e0"  # github google-research (IFEval)
CONFAIDE_REV = "3174c63e689226189675a740b7bf9053f10bae50"  # github skywalker023/confAIde
PE_REV = "9f7613f40399228011d40049d10c8c2cec8255ed"  # HF locuslab/password_eval
TT_REV = "747a75e096761ebc01bd3970158827326b4add23"  # github HumanCompatibleAI/tensor-trust-data
GSM_REV = "740312add88f781978c0658806c59bc2815b9866"  # HF openai/gsm8k
ARC_REV = "210d026faf9955653af8916fad021475a3f00453"  # HF allenai/ai2_arc
NLTK_REV = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"  # github nltk/nltk_data

_TEAM_PIN = "the AI6130 Group 11 team protocol (Kaur et al., 2026): md5 pins of the inputs"
_TEAM_RAW = "the AI6130 Group 11 team protocol (Kaur et al., 2026): sha256 of the downloaded files"
_PROTOCOL = "docs/eval_protocol.md"
_OWN = "recorded for this repository ({})"

MANIFEST: dict[str, DataFile] = {f.name: f for f in [
    DataFile("mmlu_test", _hf("cais/mmlu", MMLU_REV, "all/test-00000-of-00001.parquet"), MMLU_REV, "HF cais/mmlu",
             (Check("md5", "4d0d6cea", "team", _TEAM_PIN),), rows=14042, licence="MIT"),
    DataFile("ifeval_input", _gh("google-research/google-research", GR_REV,
                                 "instruction_following_eval/data/input_data.jsonl"), GR_REV,
             "github google-research/google-research",
             (Check("md5", "4c2c4325", "team", _TEAM_PIN),), rows=541, licence="Apache-2.0"),
    DataFile("confaide_tier4", _gh("skywalker023/confAIde", CONFAIDE_REV, "benchmark/tier_4.txt"), CONFAIDE_REV,
             "github skywalker023/confAIde",
             (Check("sha256", "cf0ba2d5f4d6c136", "protocol", _PROTOCOL),), licence="MIT"),
    DataFile("password_eval_test", _hf("locuslab/password_eval", PE_REV, "data/test-00000-of-00001.parquet"), PE_REV,
             "HF locuslab/password_eval",
             (Check("sha256", "4d537a362ec9dcc4", "team", _TEAM_RAW),
              Check("canonical_sha256", "47a10c684ff7f8dd", "own", _OWN.format("converted jsonl"))),
             rows=500),
    DataFile("tt_extraction", _gh("HumanCompatibleAI/tensor-trust-data", TT_REV,
                                  "benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl"), TT_REV,
             "github HumanCompatibleAI/tensor-trust-data",
             (Check("sha256", "4090f9f57b6df69c", "team", _TEAM_RAW),), rows=570),
    DataFile("tt_hijacking", _gh("HumanCompatibleAI/tensor-trust-data", TT_REV,
                                 "benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"), TT_REV,
             "github HumanCompatibleAI/tensor-trust-data",
             (Check("sha256", "2ad3c2343bbf43cb", "team", _TEAM_RAW),), rows=776),
    DataFile("gsm8k_test", _hf("openai/gsm8k", GSM_REV, "main/test-00000-of-00001.parquet"), GSM_REV, "HF openai/gsm8k",
             (Check("sha256", "ee7b8da9e381df27", "team", _TEAM_RAW),
              Check("canonical_sha256", "ca6e7c5108b7ab7b", "protocol", _PROTOCOL)), rows=1319, licence="MIT"),
    DataFile("arc_challenge_test", _hf("allenai/ai2_arc", ARC_REV, "ARC-Challenge/test-00000-of-00001.parquet"),
             ARC_REV, "HF allenai/ai2_arc",
             (Check("sha256", "62f03257e737aed2", "team", _TEAM_RAW),
              Check("canonical_sha256", "b90e27562e002b98", "protocol", _PROTOCOL)), rows=1172,
             licence="CC-BY-SA-4.0"),
    DataFile("nltk_punkt_tab", _gh("nltk/nltk_data", NLTK_REV, "packages/tokenizers/punkt_tab.zip"), NLTK_REV,
             "github nltk/nltk_data",
             (Check("sha256", "e57f641879742777", "own", _OWN.format("fetched 2026-11-14")),),
             licence="Apache-2.0"),
]}  # fmt: skip

PROTOCOL_PREFIXES = {"confaide_tier4": "cf0ba2d5f4d6c136", "gsm8k_test": "ca6e7c5108b7ab7b",
                     "arc_challenge_test": "b90e27562e002b98"}  # fmt: skip


class DataHashError(RuntimeError):
    pass


# --------------------------------------------------------------------------- hashing


def file_digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """The converted-file bytes behind the protocol's GSM8K / ARC hashes (see module docstring)."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\r\n" for r in rows).encode("utf-8")


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - depends on env
        raise RuntimeError("pip install pyarrow to read the protocol's parquet files") from e
    return pq.read_table(path).to_pylist()


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def table_rows(path: Path) -> list[dict[str, Any]] | None:
    if path.suffix == ".parquet":
        return parquet_rows(path)
    if path.suffix == ".jsonl":
        return jsonl_rows(path)
    return None


# --------------------------------------------------------------------------- checks


@dataclass
class Record:
    """What summary.json stores for one data file."""

    name: str
    url: str
    revision: str
    path: str
    sha256: str | None = None
    md5: str | None = None
    canonical_sha256: str | None = None
    rows: int | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    match: bool = False
    protocol_prefix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_file(entry: DataFile, path: Path) -> Record:
    """Hash `path` and compare with every expected prefix of `entry`. Never raises on mismatch."""
    rec = Record(name=entry.name, url=entry.url, revision=entry.revision, path=str(path),
                 protocol_prefix=PROTOCOL_PREFIXES.get(entry.name))  # fmt: skip
    if not path.exists():
        rec.checks = [{"kind": c.kind, "expected": c.prefix, "got": None, "status": "MISSING",
                       "source": c.source, "where": c.where} for c in entry.checks]  # fmt: skip
        return rec
    rec.sha256 = file_digest(path, "sha256")
    rec.md5 = file_digest(path, "md5")
    rows = table_rows(path) if (entry.rows is not None or any(c.kind == "canonical_sha256" for c in entry.checks)) else None
    if rows is not None:
        rec.rows = len(rows)
        if any(c.kind == "canonical_sha256" for c in entry.checks):
            rec.canonical_sha256 = hashlib.sha256(canonical_jsonl_bytes(rows)).hexdigest()
    ok = True
    for c in entry.checks:
        got = {"sha256": rec.sha256, "md5": rec.md5, "canonical_sha256": rec.canonical_sha256}[c.kind]
        status = "MATCH" if got is not None and got.startswith(c.prefix) else "MISMATCH"
        ok &= status == "MATCH"
        rec.checks.append({"kind": c.kind, "expected": c.prefix, "got": got[: len(c.prefix)] if got else None,
                           "status": status, "source": c.source, "where": c.where})  # fmt: skip
    if entry.rows is not None:
        status = "MATCH" if rec.rows == entry.rows else "MISMATCH"
        ok &= status == "MATCH"
        rec.checks.append({"kind": "rows", "expected": entry.rows, "got": rec.rows, "status": status,
                           "source": "count", "where": "data rows in the file"})  # fmt: skip
    rec.match = ok
    return rec


def local_path(name: str) -> Path:
    from core.io import data_dir

    entry = MANIFEST[name]
    return data_dir() / "url" / "protocol" / name / entry.basename


def fetch(name: str) -> Path:
    """Download one manifest file (once) and return its local path."""
    from core.io import url_file

    entry = MANIFEST[name]
    return url_file(entry.url, f"protocol/{name}/{entry.basename}")


def get(name: str, download: bool = True, strict: bool = True) -> tuple[Path, Record]:
    """Local path + hash record of a manifest file. Downloads it when missing (if allowed).
    strict: a MISMATCH or MISSING raises DataHashError (protocol runs must use the pinned bytes)."""
    path = local_path(name)
    if not path.exists() and download:
        path = fetch(name)
    rec = verify_file(MANIFEST[name], path)
    if strict and not rec.match:
        bad = [f"{c['kind']} {c['status']} (expected {c['expected']}, got {c['got']})" for c in rec.checks
               if c["status"] != "MATCH"]  # fmt: skip
        raise DataHashError(f"{name}: {'; '.join(bad)} at {path}")
    return path, rec


# --------------------------------------------------------------------------- CLI


def _table(records: list[Record]) -> str:
    lines = [f"{'STATUS':9} {'FILE':20} {'KIND':17} {'EXPECTED':17} {'GOT':17} SOURCE"]
    for r in records:
        for c in r.checks:
            lines.append(f"{c['status']:9} {r.name:20} {c['kind']:17} {str(c['expected']):17} "
                         f"{str(c['got']):17} {c['source']}: {c['where']}")  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pinned protocol data files: fetch and verify")
    ap.add_argument("action", choices=("fetch", "verify", "list"))
    ap.add_argument("names", nargs="*", help=f"default: all of {', '.join(MANIFEST)}")
    args = ap.parse_args(argv)
    names = args.names or list(MANIFEST)
    unknown = [n for n in names if n not in MANIFEST]
    if unknown:
        ap.error(f"unknown data files {unknown}")
    if args.action == "list":
        for n in names:
            e = MANIFEST[n]
            print(f"{n:20} {e.upstream} @ {e.revision[:12]}  {e.url}")
        return 0
    if args.action == "fetch":
        for n in names:
            print(f"{n}: {fetch(n)}", flush=True)
    records = [verify_file(MANIFEST[n], local_path(n)) for n in names]
    print(_table(records))
    ok = all(r.match for r in records)
    print("RESULT:", "OK (every file MATCH)" if ok else "MISMATCH or MISSING above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
