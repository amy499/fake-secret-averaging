"""Offline dataset cache, seeded benchmark subsets and secret assignment for utility evals.

prepare() (login node, internet) downloads the pinned dataset revisions with huggingface_hub,
converts MMLU's parquet to JSONL (so the GPU job and the tests never need pyarrow), fetches
NLTK's punkt_tab for the IFEval checker, and writes manifest.json with repos, revisions,
licences and sha256 hashes. load_cache() (GPU job) only reads that cache and fails with the
--prepare command when anything is missing or changed. Subsets and secret assignments depend
only on the seed and the data, never on the model, so every model sees the same items.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fsa.secrets import DIGITS, SecretSpace

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CACHE_DIR = ROOT / "outputs" / "utility_cache"
MANIFEST = "manifest.json"
NLTK_DIR = "nltk_data"
NLTK_RESOURCES = ("punkt_tab",)  # Punkt sentence/word tokenizer data (NLTK >= 3.9)
PREPARE_CMD = "uv run --group ml python -m scripts.utility_eval --prepare"

DATASETS: dict[str, dict[str, str]] = {
    "mmlu": {
        "repo": "cais/mmlu",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "source_file": "all/test-00000-of-00001.parquet",
        "file": "mmlu_test.jsonl",
        "licence": "MIT",
    },
    "ifeval": {
        "repo": "google/IFEval",
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "source_file": "ifeval_input_data.jsonl",
        "file": "ifeval_input_data.jsonl",
        "licence": "Apache-2.0",
    },
}
CHECKER = {
    "name": "google-research instruction_following_eval (vendored: fsa.utility.ifeval_checker)",
    "url": "https://github.com/google-research/google-research/tree/"
    "e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval",
    "commit": "e6890f85757dd84e27ca6df2dd30651dafad28e0",
    "licence": "Apache-2.0",
    "patch": "sentence tokenizer from NLTK punkt_tab (PunktTokenizer) instead of the pickled "
    "punkt/english.pickle, which NLTK >= 3.9 refuses to load",
    "langdetect_seed": 0,
    "mode": "strict",
}

# Stream ids for numpy seed sequences [seed, stream, ...]: one per independent random choice.
_MMLU_STREAM, _IFEVAL_STREAM, _ASSIGN_STREAM, _VERIFY_STREAM = 31, 32, 21, 41


class CacheError(RuntimeError):
    """The offline cache is missing, incomplete or changed since --prepare."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mmlu_rows_from_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """cais/mmlu rows (question, subject, choices, answer) -> JSONL rows with their index."""
    out = []
    for i, r in enumerate(rows):
        choices = list(r["choices"])
        if len(choices) != 4 or int(r["answer"]) not in range(4):
            raise ValueError(f"MMLU row {i}: expected 4 choices and answer 0-3")
        out.append(
            {
                "index": i,
                "subject": r["subject"],
                "question": r["question"],
                "choices": choices,
                "answer": int(r["answer"]),
            }
        )
    return out


def _hf_download(repo: str, filename: str, revision: str, cache_dir: Path) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(cache_dir),
    )


def _read_parquet(path: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _nltk_download(resource: str, target: Path) -> None:
    import nltk

    if not nltk.download(resource, download_dir=str(target), quiet=True, raise_on_error=True):
        raise CacheError(f"nltk.download({resource!r}) failed")


def prepare(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    download: Callable[[str, str, str, Path], str] = _hf_download,
    read_parquet: Callable[[str], list[dict[str, Any]]] = _read_parquet,
    nltk_download: Callable[[str, Path], None] = _nltk_download,
    log: Callable[[str], None] = print,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Download, convert and hash everything the offline evaluation needs; write manifest.json."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "created_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "datasets": {},
        "nltk": {"dir": NLTK_DIR, "resources": list(NLTK_RESOURCES)},
        "ifeval_checker": CHECKER,
    }
    for name, d in DATASETS.items():
        log(f"downloading {d['repo']}@{d['revision'][:12]} {d['source_file']} ...")
        src = download(d["repo"], d["source_file"], d["revision"], cache_dir / "hf")
        dest = cache_dir / d["file"]
        if name == "mmlu":
            rows = mmlu_rows_from_table(read_parquet(src))
            _write_jsonl(rows, dest)
        else:
            shutil.copyfile(src, dest)
            rows = _read_jsonl(dest)
        manifest["datasets"][name] = {
            **d,
            "source_sha256": sha256_file(Path(src)),
            "sha256": sha256_file(dest),
            "n_rows": len(rows),
        }
        log(
            f"  {name}: {len(rows)} rows -> {dest.name} (sha256 {manifest['datasets'][name]['sha256'][:12]})"
        )
    for resource in NLTK_RESOURCES:
        log(f"downloading NLTK {resource} ...")
        nltk_download(resource, cache_dir / NLTK_DIR)
    (cache_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {cache_dir / MANIFEST}")
    return manifest


def load_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    """The manifest, after checking every cached file is present and unchanged (no network)."""
    cache_dir = Path(cache_dir)
    hint = f"run `{PREPARE_CMD}` on the login node (it needs internet) first"
    path = cache_dir / MANIFEST
    if not path.exists():
        raise CacheError(f"no utility dataset cache at {cache_dir} ({MANIFEST} missing): {hint}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, d in DATASETS.items():
        rec = (manifest.get("datasets") or {}).get(name)
        if not rec or rec.get("revision") != d["revision"]:
            raise CacheError(f"cache has no {name} at revision {d['revision'][:12]}: {hint}")
        f = cache_dir / rec["file"]
        if not f.exists():
            raise CacheError(f"cached file {f} is missing: {hint}")
        if sha256_file(f) != rec["sha256"]:
            raise CacheError(f"cached file {f} changed since --prepare (sha256 mismatch): {hint}")
    nltk_dir = cache_dir / NLTK_DIR
    for resource in NLTK_RESOURCES:
        if not any(nltk_dir.glob(f"*/{resource}")) and not any(nltk_dir.glob(f"*/{resource}.zip")):
            raise CacheError(f"NLTK {resource} missing under {nltk_dir}: {hint}")
    return manifest


def load_dataset(cache_dir: Path, manifest: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    return _read_jsonl(Path(cache_dir) / manifest["datasets"][name]["file"])


# --------------------------------------------------------------------------- subsets


def mmlu_subset(rows: Sequence[Mapping[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """n questions stratified by subject: floor(n / n_subjects) from every subject, then one more
    from each of (n mod n_subjects) seeded subjects. Returned in dataset order."""
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    subjects = sorted(by_subject)
    if not 1 <= n <= len(rows):
        raise ValueError(f"MMLU subset size must be 1..{len(rows)}")
    rng = np.random.default_rng([seed, _MMLU_STREAM])
    per, extra = divmod(n, len(subjects))
    picked: list[Mapping[str, Any]] = []
    remaining: dict[str, list[Mapping[str, Any]]] = {}
    for s in subjects:
        group = by_subject[s]
        order = rng.permutation(len(group))
        k = min(per, len(group))
        picked += [group[i] for i in order[:k]]
        remaining[s] = [group[i] for i in order[k:]]
    candidates = [s for s in subjects if remaining[s]]
    if extra > len(candidates):
        raise ValueError("not enough questions left for a stratified subset of this size")
    for j in sorted(rng.choice(len(candidates), size=extra, replace=False)):
        picked.append(remaining[candidates[j]][0])
    return sorted((dict(r) for r in picked), key=lambda r: r["index"])


def ifeval_subset(rows: Sequence[Mapping[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """n seeded prompts, in dataset order."""
    if not 1 <= n <= len(rows):
        raise ValueError(f"IFEval subset size must be 1..{len(rows)}")
    idx = np.random.default_rng([seed, _IFEVAL_STREAM]).choice(len(rows), size=n, replace=False)
    return [dict(rows[i]) for i in sorted(idx)]


def assign_secrets(n_items: int, heldout: Sequence[str], seed: int, stream: int) -> list[str]:
    """One held-out secret per item: a seeded order over the items, cycling through the held-out
    list, so every secret is used equally often (+-1). Depends only on (seed, stream, sizes)."""
    order = np.random.default_rng([seed, _ASSIGN_STREAM, stream]).permutation(n_items)
    out = [""] * n_items
    for j, i in enumerate(order):
        out[i] = heldout[j % len(heldout)]
    return out


def verification_items(
    heldout: Sequence[str], seed: int, space: SecretSpace = DIGITS
) -> list[dict[str, str]]:
    """Two items per held-out secret: the real code (gold "yes") and a seeded wrong code from the
    other 99 codes of the run's secret space (gold "no"; DIGITS by default, as before; the
    digits4 codes for digits4 runs, so a wrong code looks like a real one)."""
    rng = np.random.default_rng([seed, _VERIFY_STREAM])
    items = []
    for s in heldout:
        others = [c for c in space.secrets if c != s]
        wrong = others[int(rng.integers(len(others)))]
        items.append({"item_id": f"verify:{s}:real", "secret": s, "code": s, "gold": "yes"})
        items.append({"item_id": f"verify:{s}:wrong", "secret": s, "code": wrong, "gold": "no"})
    return items


def pe_access_items(
    heldout: Sequence[str], compliant: Sequence[str], withhold: Sequence[str]
) -> list[dict[str, str]]:
    """digits4 pe_access: every held-out secret x every evaluation-only compliant prompt (gold
    "reveal": the user gives the right password) and withhold prompt (gold "withhold": no
    password). Prompt ids only; the text is filled per secret (its password) by the caller."""
    items = []
    for s in heldout:
        for pid in compliant:
            items.append({"item_id": f"pe_access:{s}:{pid}", "secret": s, "prompt_id": pid,
                          "gold": "reveal"})  # fmt: skip
        for pid in withhold:
            items.append({"item_id": f"pe_access:{s}:{pid}", "secret": s, "prompt_id": pid,
                          "gold": "withhold"})  # fmt: skip
    return items
