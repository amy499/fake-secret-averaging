"""Command-line tools of FSA-lite (python -m scripts.<name>, run from the repository root)."""

import sys as _sys
from pathlib import Path as _Path

# The fsa and fsa_runtime packages live in src/fsa_lite; make them importable without installing.
_SRC = str(_Path(__file__).resolve().parents[1] / "src" / "fsa_lite")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
