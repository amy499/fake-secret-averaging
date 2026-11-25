"""Google's IFEval checker for the re-scorer (stats/rescore.py), loaded on its own.

Status: ✅ implemented + tested (tests/test_rescore.py: the same replies scored twice give identical
results; equal to the run's scorer on every reply tried).

The re-scorer must not import the producing code (benchmarks/, metrics/, core/, models/; a test
checks that). Google's checker is neither: it is third-party code, vendored unmodified in
benchmarks/_vendor/instruction_following_eval (google-research @ e6890f85757d). This module puts
that folder on sys.path and loads it directly, with the same pinned set-up the run used
(benchmarks/_ifeval_google.py), re-done here rather than imported:
  * the vendored files' md5 prefixes are checked (refuses a modified checker);
  * the sentence tokenizer is nltk.tokenize.PunktTokenizer("english") from the pinned punkt_tab
    zip (sha256 prefix e57f641879742777, checked here), unpacked under $HARNESS_DATA/nltk_data and
    put first on nltk.data.path; never the network;
  * langdetect's DetectorFactory.seed = 0 and random.seed(0) before EVERY strict and every loose
    call, so the language checks (and any random default) give the same answer every time.
Same checker, same data, same seeds as the run: re-scoring a stored reply must give the stored
score exactly; any difference means the stored score is wrong or the checker set-up drifted.
"""

from __future__ import annotations

import functools
import hashlib
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "benchmarks" / "_vendor"
VENDORED_MD5 = {"instructions.py": "ff24576b", "instructions_registry.py": "08e30048",
                "evaluation_lib.py": "0f0d52f1", "instructions_util.py": "c2c50a67"}  # fmt: skip
PUNKT_TAB_SHA256 = "e57f641879742777"
SEED = 0

_state: dict[str, Any] = {}


class CheckerUnavailable(RuntimeError):
    pass


def _data_dir() -> Path:
    return Path(os.environ.get("HARNESS_DATA", ROOT / ".cache" / "data"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_vendored() -> dict[str, str]:
    got = {}
    for name, want in VENDORED_MD5.items():
        data = (VENDOR / "instruction_following_eval" / name).read_bytes().replace(b"\r\n", b"\n")
        got[name] = hashlib.md5(data).hexdigest()
        if not got[name].startswith(want):
            raise CheckerUnavailable(f"vendored {name} md5 {got[name][:8]} != pinned {want}: the checker was modified")
    return got


def _punkt_root() -> tuple[Path, str]:
    """$HARNESS_DATA/nltk_data with the pinned punkt_tab unpacked from its hash-checked zip."""
    data = _data_dir()
    zpath = data / "url" / "protocol" / "nltk_punkt_tab" / "punkt_tab.zip"
    if not zpath.exists():
        raise CheckerUnavailable(f"{zpath} missing (python -m core.data fetch nltk_punkt_tab)")
    sha = _sha256(zpath)
    if not sha.startswith(PUNKT_TAB_SHA256):
        raise CheckerUnavailable(f"{zpath} sha256 {sha[:16]} != pinned {PUNKT_TAB_SHA256}")
    root = data / "nltk_data"
    if not (root / "tokenizers" / "punkt_tab" / "english").is_dir():
        with zipfile.ZipFile(zpath) as zf:
            members = [n for n in zf.namelist() if n.startswith("punkt_tab/") and ".." not in n]
            zf.extractall(root / "tokenizers", members=members)
    return root, sha


def load() -> tuple[Any, Any]:
    """(evaluation_lib, langdetect), set up as the run's scorer. Raises CheckerUnavailable."""
    if "ev" in _state:
        return _state["ev"], _state["langdetect"]
    try:
        import langdetect
        import nltk
    except ImportError as e:
        raise CheckerUnavailable(f"{e} (install requirements.txt)") from e
    md5 = _check_vendored()
    root, sha = _punkt_root()
    if str(root) not in nltk.data.path:
        nltk.data.path.insert(0, str(root))
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from instruction_following_eval import evaluation_lib, instructions_util

    instructions_util._get_sentence_tokenizer = functools.lru_cache(maxsize=None)(
        lambda: nltk.tokenize.PunktTokenizer("english"))  # fmt: skip
    _state.update(ev=evaluation_lib, langdetect=langdetect,
                  info={"checker": "google-research instruction_following_eval @ e6890f85757d (vendored)",
                        "vendored_md5": md5, "punkt_tab_sha256": sha, "nltk": nltk.__version__,
                        "langdetect_seed": SEED, "random_seed": SEED})  # fmt: skip
    return evaluation_lib, langdetect


def info() -> dict[str, Any]:
    load()
    return dict(_state["info"])


def _seed(langdetect: Any) -> None:
    langdetect.DetectorFactory.seed = SEED
    random.seed(SEED)


def score(prompt: str, response: str, ids: list[str], kwargs: list[dict], key: Any = 0) -> dict[str, Any]:
    """Strict and loose, prompt level and per instruction, for one stored reply."""
    ev, ld = load()
    inp = ev.InputExample(key=key, instruction_id_list=list(ids), prompt=prompt, kwargs=[dict(k) for k in kwargs])
    _seed(ld)
    strict = ev.test_instruction_following_strict(inp, {prompt: response})
    _seed(ld)
    loose = ev.test_instruction_following_loose(inp, {prompt: response})
    return {"prompt_strict": bool(strict.follow_all_instructions), "prompt_loose": bool(loose.follow_all_instructions),
            "instructions_strict": [bool(x) for x in strict.follow_instruction_list],
            "instructions_loose": [bool(x) for x in loose.follow_instruction_list]}  # fmt: skip
