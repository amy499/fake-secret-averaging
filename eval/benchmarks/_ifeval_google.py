"""Adapter for Google's IFEval checker, vendored unmodified in benchmarks/_vendor/ (see NOTICE.md).

Status: ✅ implemented + tested (tests/test_ifeval_google.py with the pinned nltk==3.9.1,
langdetect==1.0.9, absl-py==2.1.0, immutabledict==4.2.1 from requirements.txt). The nltk_data
punkt_tab zip (pinned, hash-checked) is downloaded on first use, or ahead of time with
`python -m core.data fetch nltk_punkt_tab`. A real protocol run calls IFEvalTeam.preflight before
the model loads and stops if the checker cannot load; dry runs and fixture runs skip it.

What the adapter does (nothing in the vendored files changes):
  * puts benchmarks/_vendor on sys.path, so the upstream absolute imports
    (`from instruction_following_eval import ...`) resolve to the vendored copy;
  * replaces instructions_util._get_sentence_tokenizer with nltk.tokenize.PunktTokenizer("english")
    from the pickle-free punkt_tab data (the team protocol's documented one-line patch
    docs/prereg/PREREG_general_ability.md; nltk 3.9 no longer loads punkt/english.pickle); the
    pinned punkt_tab zip is unpacked under $HARNESS_DATA/nltk_data and put first on nltk.data.path;
  * sets langdetect's DetectorFactory.seed = 0 and random.seed(0) before every strict and every
    loose call, so the language checks (and any instruction that draws a random default) are
    repeatable run to run;
  * scores one reply with evaluation_lib.test_instruction_following_strict / _loose, as the team
    protocol does (prompt level = follow_all_instructions,
    instruction level = follow_instruction_list).
"""

from __future__ import annotations

import functools
import random
import sys
import zipfile
from pathlib import Path
from typing import Any

VENDOR = Path(__file__).resolve().parent / "_vendor"
CHECKER_COMMIT = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
PINNED = {"nltk": "3.9.1", "langdetect": "1.0.9", "absl-py": "2.1.0", "immutabledict": "4.2.1"}
VENDORED_MD5 = {"instructions.py": "ff24576b", "instructions_registry.py": "08e30048",
                "evaluation_lib.py": "0f0d52f1", "instructions_util.py": "c2c50a67"}  # fmt: skip
SEED = 0

_state: dict[str, Any] = {}


def vendored_md5() -> dict[str, str]:
    """md5 of each vendored file's LF bytes (CRLF from a Windows checkout is normalised)."""
    import hashlib

    out = {}
    for name in VENDORED_MD5:
        data = (VENDOR / "instruction_following_eval" / name).read_bytes().replace(b"\r\n", b"\n")
        out[name] = hashlib.md5(data).hexdigest()
    return out


def punkt_tab_dir(download: bool = True) -> Path:
    """Unpack the pinned punkt_tab zip (hash-checked) under $HARNESS_DATA/nltk_data; return that root."""
    from core import data as D
    from core.io import data_dir

    root = data_dir() / "nltk_data"
    english = root / "tokenizers" / "punkt_tab" / "english"
    if not english.is_dir():
        path, _ = D.get("nltk_punkt_tab", download=download, strict=True)
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.startswith("punkt_tab/") and ".." not in n]
            zf.extractall(root / "tokenizers", members=members)
    return root


def _load(download: bool = True) -> tuple[Any, Any]:
    """(evaluation_lib, langdetect) with the tokenizer replaced. Raises ImportError / LookupError /
    core.data.DataHashError when a dependency or the punkt_tab data is missing."""
    if "ev" in _state:
        return _state["ev"], _state["langdetect"]
    import langdetect
    import nltk

    root = punkt_tab_dir(download=download)
    if str(root) not in nltk.data.path:
        nltk.data.path.insert(0, str(root))
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from instruction_following_eval import evaluation_lib, instructions_util

    instructions_util._get_sentence_tokenizer = functools.lru_cache(maxsize=None)(
        lambda: nltk.tokenize.PunktTokenizer("english"))  # fmt: skip
    _state.update(ev=evaluation_lib, langdetect=langdetect, nltk_version=nltk.__version__)
    return evaluation_lib, langdetect


def _seed(langdetect: Any) -> None:
    langdetect.DetectorFactory.seed = SEED
    random.seed(SEED)


def score(key: Any, ids: list[str], kwargs: list[dict], prompt: str, response: str,
          download: bool = True) -> dict[str, Any]:
    """Strict and loose results for one reply: prompt level (all instructions followed) and the
    per-instruction list."""
    ev, ld = _load(download)
    inp = ev.InputExample(key=key, instruction_id_list=list(ids), prompt=prompt, kwargs=[dict(k) for k in kwargs])
    _seed(ld)
    strict = ev.test_instruction_following_strict(inp, {prompt: response})
    _seed(ld)
    loose = ev.test_instruction_following_loose(inp, {prompt: response})
    return {"prompt_strict": bool(strict.follow_all_instructions), "prompt_loose": bool(loose.follow_all_instructions),
            "instructions_strict": [bool(x) for x in strict.follow_instruction_list],
            "instructions_loose": [bool(x) for x in loose.follow_instruction_list]}  # fmt: skip


def info() -> dict[str, Any]:
    """What summary.json records about the checker."""
    from importlib import metadata

    versions = {}
    for dist in PINNED:
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    md5 = vendored_md5()
    return {"commit": CHECKER_COMMIT, "versions": versions, "pinned": PINNED,
            "versions_match_pins": all(versions[k] == v for k, v in PINNED.items()),
            "vendored_md5_match": all(md5[k].startswith(v) for k, v in VENDORED_MD5.items()),
            "sentence_tokenizer": 'nltk.tokenize.PunktTokenizer("english") (punkt_tab)',
            "langdetect_seed": SEED, "random_seed": SEED}  # fmt: skip
