"""Write the evaluation secret lists that are never committed, from their fixed seeds, to their
git-ignored paths, and print only their sha256 (docs/eval_protocol.md records the same values).

    python scripts/gen_eval_secrets.py

  digits4  the 100 FSA 4-digit codes: numpy.random.default_rng(4004), sorted (fsa.secrets.digits4_codes)
           -> .cache/secrets/digits4_100.txt
The evaluation harness generates its own lists (eval/secret_types, seeds in each module) into
eval/secret_types/generated/ (git-ignored) and checks them against eval/secret_types/hashes.json.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "fsa_lite"))

from fsa.secrets import DIGITS4_FILE, write_digits4_file  # noqa: E402


def main() -> int:
    path = ROOT / DIGITS4_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    list_sha = write_digits4_file(path)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"digits4: {DIGITS4_FILE} (100 codes) list sha256 {list_sha} file sha256 {file_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
