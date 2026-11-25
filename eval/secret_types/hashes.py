"""Print or (re)write the committed SHA-256 of each generated secret list.

    python -m secret_types.hashes            # print current hashes vs committed
    python -m secret_types.hashes --write digits_4 api_key ...

Only use --write when a generator change is intended; the new hash must then be committed.
"""

from __future__ import annotations

import argparse
import json

from core import registry
from secret_types.base import HASH_FILE, committed_hashes

FIXED = ["digits_4", "digits_2", "api_key", "access_token", "cloud_key", "conn_string", "random_password",
         "human_password", "secret_words"]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", nargs="*", default=None)
    args = ap.parse_args(argv)
    committed = committed_hashes()
    for name in [n for n in FIXED if n in registry.names("secret")]:
        src = registry.get("secret", name)()
        try:
            h = src.sha256()
        except Exception as e:  # noqa: BLE001 - e.g. word list not downloadable
            print(f"{name:16s} unavailable: {e}")
            continue
        if committed.get(name) is None:
            mark = "MISSING (no committed hash yet)"
        else:
            mark = "ok" if committed[name] == h else f"DIFFERS from committed {committed[name][:12]}..."
        print(f"{name:16s} {h} {mark}")
        if args.write is not None and (not args.write or name in args.write):
            committed[name] = h
    if args.write is not None:
        HASH_FILE.write_text(json.dumps(committed, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {HASH_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
