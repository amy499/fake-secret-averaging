"""Carry cached base replies over to a rebuilt data folder, but ONLY for units whose prompt is byte-identical.

Status: ✅ tested on CPU (selftest_run3.py); 🟡 not yet run on the Slurm cluster / the PBS cluster.

Why: v2_train's ReplyCache (<run>/replies.jsonl) is keyed by "<reply_key>|<reply no>" only, not by
content. After a rebuild a reply_key can name a DIFFERENT prompt (KGP context ids are reassigned), so
copying the old file would silently train on replies to other prompts. This keeps an entry only when
every unit that shares its reply_key has exactly the same messages (JSON, sorted keys) in the old and
the new units.jsonl, and the same number of such units; everything else is left out, and the train or
pilot job generates it (make_replies only generates missing keys).

  python src/fsa_lite/lite/reuse_replies.py --config <run3 config> --old-data <old built_run3> \\
      --new-data <new built_run3> --replies <old run>/replies.jsonl --out <new run>/replies.jsonl
Refuses to overwrite --out. Prints counts per turn type (kept, to generate): never a prompt or reply.
Note: a kept reply is the one the old run generated (its batch and seed came from the old job list); a
fresh run on the new data alone would sample it with another batch seed. Same model, prompt, sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def signatures(units: list[dict]) -> dict[str, list[str]]:
    """reply_key -> sorted sha256 of the messages of every unit with that key."""
    out: dict[str, list[str]] = {}
    for u in units:
        s = hashlib.sha256(json.dumps(u["messages"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        out.setdefault(u["reply_key"], []).append(s)
    return {k: sorted(v) for k, v in out.items()}


def read_units(d: Path) -> list[dict]:
    return [json.loads(x) for x in (Path(d) / "units.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def plan(cfg: dict, old_units: list[dict], new_units: list[dict], cached: set[str]) -> dict:
    """Which needed train reply keys of the new data can be kept (content-identical and cached)."""
    import lite_source as LS

    src = LS.LiteSource(cfg, pilot=False, units=new_units, pools={}, manifest={})
    old_sig, new_sig = signatures(old_units), signatures(new_units)
    keep: set[str] = set()
    per: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for u in src.train:
        if u["turn_type"] not in src.weights:
            continue
        for r in range(src.n_replies(u)):
            key = LS.reply_cache_key(u, r)
            if key in seen:
                continue
            seen.add(key)
            c = per.setdefault(u["turn_type"], {"kept": 0, "to_generate": 0})
            if key in cached and old_sig.get(u["reply_key"]) == new_sig[u["reply_key"]]:
                keep.add(key)
                c["kept"] += 1
            else:
                c["to_generate"] += 1
    return {"keep": keep, "per_type": per, "needed": len(seen)}


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--old-data", type=Path, required=True)
    ap.add_argument("--new-data", type=Path, required=True)
    ap.add_argument("--replies", type=Path, required=True, help="the old run's replies.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="the new run's replies.jsonl (must not exist)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if a.out.exists():
        print(f"{a.out} exists: refusing to overwrite (use a fresh run folder)")
        return 1
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    lines = [x for x in a.replies.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_key = {json.loads(x)["key"]: x for x in lines}
    p = plan(cfg, read_units(a.old_data), read_units(a.new_data), set(by_key))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="\n") as f:
        for k in sorted(p["keep"]):
            f.write(by_key[k] + "\n")
    gen = sum(v["to_generate"] for v in p["per_type"].values())
    print(f"replies: {len(p['keep'])} kept (prompt byte-identical), {gen} to generate, of {p['needed']} needed "
          f"train replies; old cache {len(by_key)} entries")  # fmt: skip
    print("per turn type:", json.dumps(dict(sorted(p["per_type"].items()))))
    print(f"REUSE_TO_GENERATE {gen}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
