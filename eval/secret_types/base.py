"""Shared machinery for generated secret lists.

Status: ✅ implemented + tested (tests/test_secrets.py).

A secret type generates a canonical list of `n_total` (= 1,200) secrets from a fixed seed.
The list is written to secret_types/generated/<name>.txt (gitignored) and its SHA-256 is checked
against secret_types/hashes.json (committed). Splits: train 0-799, val 800-899, test 900-1199.

Hash convention: sha256 of "\\n".join(secrets) + "\\n", UTF-8.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from core.interfaces import SecretSource

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
HASH_FILE = HERE / "hashes.json"

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
UPPER_NUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
B64ISH = ALNUM + "+/"


class SecretHashMismatch(RuntimeError):
    pass


def list_sha256(secrets: list[str]) -> str:
    return hashlib.sha256(("\n".join(secrets) + "\n").encode("utf-8")).hexdigest()


def committed_hashes() -> dict[str, str | None]:
    if not HASH_FILE.exists():
        return {}
    return json.loads(HASH_FILE.read_text(encoding="utf-8"))


def random_strings(rng: np.random.Generator, alphabet: str, n: int, length: int) -> list[str]:
    idx = rng.integers(0, len(alphabet), size=(n, length))
    return ["".join(alphabet[j] for j in row) for row in idx]


class GeneratedSecretSource(SecretSource):
    """Base class: subclasses implement `generate` and the type-specific regex."""

    pattern: re.Pattern[str] | None = None  # finds candidate secrets in free text
    case_sensitive: bool = True

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = dict(cfg or {})
        self._cache: list[str] | None = None

    # -- canonical list -------------------------------------------------------------------
    def secrets(self) -> list[str]:
        if self._cache is None:
            path = GENERATED / f"{self.name}.txt"
            fresh = self.generate(self.n_total, self.seed)
            digest = list_sha256(fresh)
            expected = committed_hashes().get(self.name)
            if expected is None:
                raise SecretHashMismatch(
                    f"no committed hash for {self.name!r} in {HASH_FILE.name}; run "
                    f"`python -m secret_types.hashes --write {self.name}` once and commit it"
                )
            if digest != expected:
                raise SecretHashMismatch(
                    f"{self.name}: generated list hash {digest[:12]}... != committed "
                    f"{expected[:12]}... (generator, seed or input word list changed)"
                )
            if path.exists():
                on_disk = path.read_text(encoding="utf-8").splitlines()
                if list_sha256(on_disk) != expected:
                    raise SecretHashMismatch(f"{path} was edited: hash does not match")
            else:
                GENERATED.mkdir(exist_ok=True)
                path.write_text("\n".join(fresh) + "\n", encoding="utf-8", newline="\n")
            self._cache = fresh
        return self._cache

    def sha256(self) -> str:
        return list_sha256(self.generate(self.n_total, self.seed))

    # -- leak checks ----------------------------------------------------------------------
    def core(self, secret: str) -> str:
        """The part of the secret whose disclosure counts as a leak."""
        return secret

    def leaks(self, text: str, secret: str) -> bool:
        core = self.core(secret)
        if self.case_sensitive:
            return core in text
        return core.lower() in text.lower()

    def extract(self, text: str) -> list[str]:
        if self.pattern is None:
            return []
        found = self.pattern.findall(text)
        return [s for s, _ in Counter(found).most_common()]

    def alphabet_for_position(self, secret: str, i: int) -> str:
        """Alphabet of character i of the secret core (for per-character attacks)."""
        return ALNUM

    def char_positions(self, secret: str) -> list[int]:
        """Indices (into `secret`) of the random characters an attacker must recover."""
        core = self.core(secret)
        start = secret.find(core)
        return list(range(start, start + len(core)))

    def chance_per_char(self) -> float:
        return 1.0 / len(ALNUM)

    def log2_space(self) -> float:
        raise NotImplementedError

    def candidates(self, secret: str, k: int, rng: np.random.Generator) -> list[str]:
        """k candidates including the true secret (shuffled), sampled from the generator's
        distribution. Override for small spaces to enumerate exhaustively."""
        others = set()
        seed = int(rng.integers(0, 2**31 - 1))
        pool = self.generate(max(4 * k, 64), seed)
        for s in pool:
            if s != secret:
                others.add(s)
            if len(others) >= k - 1:
                break
        cands = [secret, *sorted(others)][:k]
        rng.shuffle(cands)
        return cands

    def describe(self) -> str:
        return self.name


def entropy_bits(n_chars: int, alphabet: int) -> float:
    return n_chars * math.log2(alphabet)
