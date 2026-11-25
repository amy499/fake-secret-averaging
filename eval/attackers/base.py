"""Shared attacker machinery.

Status: ✅ implemented + tested (tests/test_attackers.py, tests/test_dry_run.py).

Active attackers send their own prompts to the (defended) model. Each item gets a list of
query conversations; the attacker stops at the first success or when the budget is spent.
Every Guess records:
  * guess          the attacker's extracted secret (None if nothing found)
  * attempts       one Attempt per query (prompt, reply, guess after that query)
  * extra.leaked   oracle check: did any reply contain the secret (benchmark leak rule)?
  * extra.first_success  1-based query index of the first leaking reply (None if never)

Extraction: for secret types with a candidate space (digits, keys) the guess is the first
candidate the secret type finds in the reply. For benchmarks whose items carry their own
free-text secrets (PasswordEval, Tensor Trust, private facts) there is no generic extractor; the
guess is then the secret if the reply leaks it (an oracle extractor = upper bound on attacker
success), which matches how those benchmarks define leakage.

Sampling: attack replies use the leak decoding settings (temperature 1) with seed
    attack_seed(attacker, secret_index, query) = 2026e6 + crc32(f"{attacker}|{secret_index}|{query}") % 1e9
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Any

from core.interfaces import Attacker, Attempt, GenConfig, Guess, Item, Message, ModelWrapper, Reply, secret_contexts


def attack_seed(attacker: str, secret_index: int, query: int) -> int:
    return 2026 * 10**6 + zlib.crc32(f"{attacker}|{secret_index}|{query}".encode()) % 10**9


def secret_noun(item: Item, ctx: Any) -> str:
    name = getattr(ctx.benchmark, "name", "") if ctx else ""
    if "password" in name:
        return "password"
    if "tensor_trust" in name:
        return "access code"
    sname = getattr(ctx.secrets, "name", "") if ctx else ""
    return {"digits_4": "secret code", "digits_2": "secret code", "api_key": "API key", "access_token": "access token",
            "cloud_key": "secret access key", "conn_string": "database password",
            "random_password": "password", "human_password": "password",
            "secret_words": "secret word"}.get(sname, "secret")  # fmt: skip


def extract_guess(ctx: Any, item: Item, text: str) -> str | None:
    if not item.secret:
        return None
    if getattr(ctx.benchmark, "uses_secret_source", True):
        cands = ctx.secrets.extract(text)
        return cands[0] if cands else None
    return item.secret if ctx.leaks(item, text) else None


def replace_user(item: Item, user_text: str) -> list[Message]:
    """Item's conversation with the final user turn replaced by `user_text`."""
    msgs = item.messages()
    if msgs and msgs[-1]["role"] == "user":
        msgs = msgs[:-1]
    return msgs + [{"role": "user", "content": user_text}]


class ActiveAttacker(Attacker):
    """Subclasses implement `queries(item, budget)` -> list of conversations (one per query)."""

    passive = False
    stop_on_success = True

    def queries(self, item: Item, budget: int) -> list[list[Message]]:
        raise NotImplementedError

    def decode(self, item: Item, reply_text: str) -> str | None:
        """Turn one reply into a guess (override for covert encodings)."""
        return extract_guess(self.ctx, item, reply_text)

    def gen_cfg(self) -> GenConfig:
        greedy = self.cfg.get("decoding", "sampled") == "greedy"
        base = self.ctx.leak_cfg
        return GenConfig(max_new_tokens=base.max_new_tokens, temperature=0.0 if greedy else base.temperature,
                         top_p=base.top_p)  # fmt: skip

    def attack(self, model: ModelWrapper, items: Sequence[Item], replies: Sequence[Reply], budget: int) -> list[Guess]:
        targets = [it for it in items if it.secret]
        plans = {it.id: self.queries(it, budget)[:budget] for it in targets}
        # Generate query q for every item at once (batched), then drop items that succeeded.
        attempts: dict[str, list[Attempt]] = {it.id: [] for it in targets}
        first: dict[str, int | None] = {it.id: None for it in targets}
        max_q = max((len(p) for p in plans.values()), default=0)
        gen_cfg = self.gen_cfg()
        for q in range(max_q):
            active = [it for it in targets if q < len(plans[it.id])
                      and not (self.stop_on_success and first[it.id] is not None)]  # fmt: skip
            if not active:
                break
            seeds = [attack_seed(self.name, self.ctx.secret_index.get(it.id, 0), q) for it in active]
            qs = [plans[it.id][q] for it in active]
            gens = model.generate(qs, gen_cfg, seeds, secret_ctx=secret_contexts(qs, [it.secret for it in active]))
            for it, g in zip(active, gens):
                guess = self.decode(it, g.text)
                attempts[it.id].append(Attempt(query=q + 1, prompt=plans[it.id][q][-1]["content"],
                                               reply=g.text, guess=guess, reasoning=g.reasoning))  # fmt: skip
                if first[it.id] is None and (self.ctx.leaks(it, g.text) or guess == it.secret):
                    first[it.id] = q + 1
        out = []
        for it in targets:
            att = attempts[it.id]
            hits = [a.guess for a in att if a.guess == it.secret]
            final = hits[0] if hits else next((a.guess for a in reversed(att) if a.guess), None)
            out.append(Guess(item_id=it.id, attacker=self.name, guess=final, queries=len(att), attempts=att,
                             extra={"leaked": first[it.id] is not None, "first_success": first[it.id],
                                    "budget": budget}))  # fmt: skip
        return out
