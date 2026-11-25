from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.io  # noqa: E402,F401  (puts the repository root on sys.path: baselines/ lives there)
