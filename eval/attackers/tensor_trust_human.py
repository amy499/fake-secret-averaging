"""tensor_trust_human: replay human-written Tensor Trust attacks.

Status: ✅ implemented + tested (unit + dry-run on fixture attacks); real attack pool from HF
`qxcv/tensor-trust` (extraction + hijacking robustness files).

* On Tensor Trust items the item's own attack (metadata.attack) is replayed first.
* Then `budget` attacks sampled (rng seeded by secret index) from the pool of benchmark attack
  strings (`pool: extraction | hijacking | both`, default extraction), each sent as the user
  turn under the item's own system prompt.
"""

from __future__ import annotations

import numpy as np

from attackers.base import ActiveAttacker, replace_user
from core.interfaces import Item, Message
from core.registry import register

_POOL_CACHE: dict[tuple[str, bool], list[str]] = {}


def attack_pool(which: str, dry_run: bool) -> list[str]:
    key = (which, dry_run)
    if key not in _POOL_CACHE:
        from benchmarks.tensor_trust import fixture_rows, load_rows

        kinds = ["extraction", "hijacking"] if which == "both" else [which]
        attacks: list[str] = []
        for k in kinds:
            rows = fixture_rows() if dry_run else load_rows(k)
            attacks += [r["attack"] for r in rows]
        _POOL_CACHE[key] = sorted(set(attacks))
    return _POOL_CACHE[key]


@register("attacker", "tensor_trust_human")
class TensorTrustHuman(ActiveAttacker):
    name = "tensor_trust_human"

    def queries(self, item: Item, budget: int) -> list[list[Message]]:
        qs: list[list[Message]] = []
        if item.metadata.get("attack") and item.metadata.get("variant", "attack") == "attack":
            qs.append(item.messages())
        pool = attack_pool(self.cfg.get("pool", "extraction"), self.ctx.dry_run)
        rng = np.random.default_rng(2026 + self.ctx.secret_index.get(item.id, 0))
        for j in rng.permutation(len(pool))[: max(budget - len(qs), 0)]:
            qs.append(replace_user(item, pool[int(j)]))
        return qs
