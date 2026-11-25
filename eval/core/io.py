"""Files, paths and downloads.

Status: ✅ implemented + tested (tests/test_dry_run.py). The download helpers are 🟡: written
against the documented HF file layouts, not yet run against the live hub from this repo.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent  # eval/: the harness
REPO = ROOT.parent  # the repository root (configs/, baselines/, scripts/, src/)
CONFIGS = REPO / "configs"  # attack/ (attack-study configs), eval/ (approach specs), training/
if str(REPO) not in sys.path:  # baselines/ (none, prompt_instruction, output_filter) is at the root
    sys.path.append(str(REPO))


def checkout_root() -> Path:
    """The git checkout: the repository root, or eval/ when the harness is used on its own."""
    return REPO if (REPO / ".git").exists() or (REPO / ".gitignore").is_file() else ROOT


def data_dir() -> Path:
    """Where downloaded benchmark data is cached (HARNESS_DATA, default <repo>/.cache/data)."""
    return Path(os.environ.get("HARNESS_DATA", ROOT / ".cache" / "data"))


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(_plain(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_plain)
        f.write("\n")


def git_dirs(root: Path | None = None) -> tuple[Path, Path] | None:
    """(gitdir, commondir) of the checkout at `root`, read from the files (no git binary: The PBS cluster
    compute nodes have none). Handles a normal clone (.git/ folder) and a worktree (.git file
    "gitdir: <path>", refs in its commondir)."""
    root = root or checkout_root()
    dot = root / ".git"
    if dot.is_dir():
        return dot, dot
    if dot.is_file():
        text = dot.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text[len("gitdir:"):].strip())
            gitdir = gitdir if gitdir.is_absolute() else (root / gitdir)
            common = gitdir
            if (gitdir / "commondir").is_file():
                c = Path((gitdir / "commondir").read_text(encoding="utf-8").strip())
                common = c if c.is_absolute() else (gitdir / c)
            return gitdir, common.resolve()
    return None


def _resolve_ref(gitdir: Path, common: Path, ref: str) -> str | None:
    for base in (gitdir, common):
        f = base / ref
        if f.is_file():
            return f.read_text(encoding="utf-8").strip() or None
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha
    return None


def _segments_match(name: str, pattern: str) -> bool:
    """fnmatch per path segment: `*` and `?` never cross a `/` (as in .gitignore)."""
    from fnmatch import fnmatchcase

    a, b = name.split("/"), pattern.split("/")
    return len(a) == len(b) and all(fnmatchcase(x, y) for x, y in zip(a, b))


def gitignored(path: Path, root: Path | None = None) -> bool | None:
    """Would git ignore `path`? Read from the checkout's .gitignore files (the root one and
    eval/.gitignore, each anchored at its own folder) and info/exclude, without calling git (some
    cluster compute nodes have none). Supports what this repo uses: anchored patterns (containing
    "/"), unanchored names, `*` / `?` within a segment, a leading `**/` (any depth), directory-only
    patterns (trailing "/") and `!` negation (last match wins). None when `path` is outside the
    checkout."""
    root = (root or checkout_root()).resolve()
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    sources: list[tuple[Path, Path]] = [(root / ".gitignore", root)]
    if ROOT.resolve() != root and ROOT.resolve().is_relative_to(root):
        sources.append((ROOT.resolve() / ".gitignore", ROOT.resolve()))
    dirs = git_dirs(root)
    if dirs:
        sources += [(dirs[0] / "info" / "exclude", root), (dirs[1] / "info" / "exclude", root)]
    ignored = False
    for f, base in sources:
        if not f.is_file():
            continue
        try:
            parts = target.relative_to(base).as_posix().split("/")
        except ValueError:
            continue
        for raw in f.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            raw = raw.rstrip()
            neg = raw.startswith("!")
            pat = raw[1:] if neg else raw
            dir_only = pat.endswith("/")
            any_depth = pat.startswith("**/")
            pat = pat[3:] if any_depth else pat
            pat = pat.strip("/")
            anchored = "/" in pat and not any_depth
            hit = False
            for i in range(1, len(parts) + 1):
                if dir_only and i == len(parts) and not target.is_dir():
                    continue
                if anchored:
                    cand = "/".join(parts[:i])
                elif "/" in pat:  # **/a/b: the last len(pat) segments ending at i
                    k = pat.count("/") + 1
                    cand = "/".join(parts[max(0, i - k):i])
                else:
                    cand = parts[i - 1]
                if _segments_match(cand, pat):
                    hit = True
                    break
            if hit:
                ignored = not neg
    return ignored


def git_commit(root: Path | None = None) -> dict[str, Any]:
    """Commit and branch of the checkout, read from .git directly (HEAD -> loose ref or
    packed-refs), never by calling git: compute nodes on the PBS cluster have no git binary. `dirty` would
    need git (index vs work tree), so it is None ("not checked")."""
    dirs = git_dirs(root)
    out: dict[str, Any] = {"commit": None, "branch": None, "dirty": None, "source": "read from .git (no git call)"}
    if dirs is None:
        return out
    gitdir, common = dirs
    try:
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return out
    if head.startswith("ref:"):
        ref = head[len("ref:"):].strip()
        out["branch"] = ref.removeprefix("refs/heads/")
        out["commit"] = _resolve_ref(gitdir, common, ref)
    else:
        out["branch"], out["commit"] = "HEAD (detached)", head
    return out


def make_run_dir(output_dir: str | Path, config_name: str) -> Path:
    base = Path(output_dir)
    if not base.is_absolute():
        base = ROOT / base
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base / f"{stamp}_{config_name}"
    i = 2
    while out.exists():
        out = base / f"{stamp}_{config_name}_{i}"
        i += 1
    out.mkdir(parents=True)
    return out


# --------------------------------------------------------------------------- downloads


def hf_file(repo_id: str, filename: str, repo_type: str = "dataset") -> Path:
    """Download one file from the HF hub into the cache (needs `huggingface_hub`)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - depends on env
        raise RuntimeError("pip install huggingface_hub to download benchmark data") from e
    return Path(hf_hub_download(repo_id, filename, repo_type=repo_type,
                                cache_dir=str(data_dir() / "hf")))  # fmt: skip


def hf_rows(repo_id: str, split: str, config: str | None = None) -> list[dict[str, Any]]:
    """All rows of an HF dataset split as dicts (needs the `datasets` library)."""
    try:
        import datasets as hf_datasets  # the HF library; never name a local folder `datasets`
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install datasets to load benchmark data from the HF hub") from e
    ds = hf_datasets.load_dataset(repo_id, config, split=split, cache_dir=str(data_dir() / "hf"))
    return [dict(r) for r in ds]


def url_file(url: str, name: str) -> Path:
    """Download a URL once into the cache and return the local path."""
    dest = data_dir() / "url" / name
    if not dest.exists():
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs from code
        tmp.replace(dest)
    return dest
