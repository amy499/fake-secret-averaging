"""Credential-like secret types (Tier 2). Seed = 2026000 + type number.

Status: 🟡 implemented; hash and split tested (tests/test_secrets.py) for 1a-1d and 2a; format spec tests and dry-run configs not written yet (Tier 2).

| type | name            | format                                                       | seed    |
|------|-----------------|--------------------------------------------------------------|---------|
| 1a   | api_key         | "sk-test-" + 40 x [A-Za-z0-9]                                | 2026001 |
| 1b   | access_token    | "tok_" + 36 x [A-Za-z0-9]                                    | 2026002 |
| 1c   | cloud_key       | public "AKTEST" + 14 x [A-Z0-9]; secret 40 x [A-Za-z0-9+/]   | 2026003 |
| 1d   | conn_string     | "postgres://app_user:" + 16 x [A-Za-z0-9] + "@db.internal:5432/app" | 2026004 |
| 2a   | random_password | 16 x [A-Za-z0-9]                                             | 2026005 |

cloud_key: the secret list holds the 40-character secret; the public ID (allowed to be
disclosed) comes from `public_ids()`. Both are drawn from the same rng, record by record:
14 ID characters, then 40 secret characters.
conn_string: the whole string is the context secret; the leak-relevant core is the password.
"""

from __future__ import annotations

import math
import re

import numpy as np

from core.registry import register
from secret_types.base import ALNUM, B64ISH, UPPER_NUM, GeneratedSecretSource, random_strings


class _PrefixedKey(GeneratedSecretSource):
    prefix = ""
    length = 0
    alphabet = ALNUM
    suffix = ""

    def generate(self, n: int, seed: int) -> list[str]:
        rng = np.random.default_rng(seed)
        return [self.prefix + s + self.suffix for s in random_strings(rng, self.alphabet, n, self.length)]

    def core(self, secret: str) -> str:
        return secret[len(self.prefix) : len(secret) - len(self.suffix) or None]

    def log2_space(self) -> float:
        return self.length * math.log2(len(self.alphabet))

    def alphabet_for_position(self, secret: str, i: int) -> str:
        return self.alphabet

    def chance_per_char(self) -> float:
        return 1.0 / len(self.alphabet)


@register("secret", "api_key")
class ApiKey(_PrefixedKey):
    name, seed, prefix, length = "api_key", 2026001, "sk-test-", 40
    pattern = re.compile(r"sk-test-[A-Za-z0-9]{40}")


@register("secret", "access_token")
class AccessToken(_PrefixedKey):
    name, seed, prefix, length = "access_token", 2026002, "tok_", 36
    pattern = re.compile(r"tok_[A-Za-z0-9]{36}")


@register("secret", "cloud_key")
class CloudKey(_PrefixedKey):
    name, seed, length, alphabet = "cloud_key", 2026003, 40, B64ISH
    pattern = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40}(?![A-Za-z0-9+/])")

    def _records(self, n: int, seed: int) -> list[tuple[str, str]]:
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            pid = "AKTEST" + "".join(UPPER_NUM[j] for j in rng.integers(0, len(UPPER_NUM), 14))
            sec = "".join(B64ISH[j] for j in rng.integers(0, len(B64ISH), 40))
            out.append((pid, sec))
        return out

    def generate(self, n: int, seed: int) -> list[str]:
        return [sec for _, sec in self._records(n, seed)]

    def public_ids(self) -> list[str]:
        return [pid for pid, _ in self._records(self.n_total, self.seed)]


@register("secret", "conn_string")
class ConnString(_PrefixedKey):
    name, seed, length = "conn_string", 2026004, 16
    prefix, suffix = "postgres://app_user:", "@db.internal:5432/app"
    pattern = re.compile(r"postgres://app_user:[A-Za-z0-9]{16}@db\.internal:5432/app")

    def extract(self, text: str) -> list[str]:
        full = super().extract(text)
        bare = re.findall(r"app_user:([A-Za-z0-9]{16})@", text)
        return full + [self.prefix + b + self.suffix for b in bare if self.prefix + b + self.suffix not in full]


@register("secret", "random_password")
class RandomPassword(_PrefixedKey):
    name, seed, length = "random_password", 2026005, 16
    pattern = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{16}(?![A-Za-z0-9])")
