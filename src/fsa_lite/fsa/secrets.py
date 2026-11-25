"""The secret space and the one train / held-out split used everywhere.

baseline_leakage.py splits secrets into attacker-train and held-out with split_secrets(seed,
n_train, n_heldout). FSA training draws real and fake secrets only from the same "train" part,
so evaluating with the same seed and sizes never touches a secret the adapter was trained on.

Three secret types share that machinery (SecretSpace): "digits", the 100 codes "00".."99";
"words", a fixed list of 100 words read from a file (load_word_list); and "digits4", a fixed
set of 100 distinct 4-digit codes ("0000".."9999", leading zeros kept) drawn once with
numpy.random.default_rng(DIGITS4_SEED) (digits4_codes). The list is never committed:
load_digits4 reads it from DIGITS4_FILE (git-ignored) and writes it there first if it is missing
(also: python scripts/gen_eval_secrets.py); docs/eval_protocol.md records its sha256. A space's secrets are in a canonical order (digits and digits4:
numeric; words: alphabetical), so the split depends only on the seed and on which secrets the
space holds, never on how a file happens to be ordered. Word and digits4 runs record the list,
its file and its hash (secret_space_record, the same keys for both: "words" holds the codes for
digits4); digit runs record nothing extra, so their outputs stay exactly as before.

digits4 threat model: the attacker knows the code is one of these 100 (the file is public), so
the secret space is N = 100 exactly as for 2-digit codes and every bits / accuracy number is
directly comparable. digits4 runs also give every code a deterministic PasswordEval-style
password (pe_password), used by the pe_base system template and the compliant prompts.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

SECRET_SPACE = 100  # "00".."99"; a word list must have exactly this many words too
SecretType = Literal["digits", "words", "digits4"]
WORD_PATTERN = re.compile(r"[a-z]+")  # one lowercase ASCII word: no spaces, digits, punctuation
CODE4_PATTERN = re.compile(r"[0-9]{4}")  # one 4-digit code, leading zeros kept
# The digits4 codes are numpy.random.default_rng(DIGITS4_SEED).choice(10_000, SECRET_SPACE,
# replace=False), sorted. Changing the seed changes every digits4 split: never change it.
DIGITS4_SEED = 4004
DIGITS4_FILE = ".cache/secrets/digits4_100.txt"  # relative to the repository root; git-ignored, generated


@dataclass(frozen=True)
class SecretSpace:
    type: SecretType
    secrets: tuple[str, ...]  # canonical order; the split permutes positions in it
    source: str | None = None  # the word list file as given (None for digits)
    sha256: str | None = None  # word_list_hash(secrets) (None for digits)

    def __len__(self) -> int:
        return len(self.secrets)

    def index(self, secret: str) -> int:
        return self.secrets.index(secret)

    @property
    def n_digits(self) -> int | None:
        """Digits per code (2 for digits, 4 for digits4); None for words."""
        return {"digits": 2, "digits4": 4}.get(self.type)


DIGITS = SecretSpace("digits", tuple(f"{i:02d}" for i in range(SECRET_SPACE)))


def word_list_hash(words: Sequence[str]) -> str:
    """sha256 of the sorted words, one per line: ignores file order and line endings."""
    return hashlib.sha256(("\n".join(sorted(words)) + "\n").encode("utf-8")).hexdigest()


def parse_word_list(text: str, source: str | None = None) -> SecretSpace:
    """A word SecretSpace from file text: blank lines ignored, entries stripped, then exactly
    SECRET_SPACE distinct lowercase single words of ASCII letters. Errors name the lines."""
    entries = [(n, line.strip()) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    where = f"word list {source}" if source else "word list"
    problems = []
    bad = [f"line {n}: {w!r}" for n, w in entries if not WORD_PATTERN.fullmatch(w)]
    if bad:
        problems.append(
            "entries must be single lowercase words of letters a-z only (no spaces, digits, "
            f"punctuation or capitals): {', '.join(bad[:10])}" + (" ..." if len(bad) > 10 else "")
        )
    seen: dict[str, int] = {}
    dupes = []
    for n, w in entries:
        if w in seen:
            dupes.append(f"{w!r} (lines {seen[w]} and {n})")
        seen.setdefault(w, n)
    if dupes:
        problems.append(f"duplicate words: {', '.join(dupes[:10])}")
    if len(entries) != SECRET_SPACE:
        problems.append(f"needs exactly {SECRET_SPACE} words, found {len(entries)}")
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems))
    words = tuple(sorted(w for _, w in entries))
    return SecretSpace("words", words, source, word_list_hash(words))


def load_word_list(path: str | Path) -> SecretSpace:
    path = Path(path)
    return parse_word_list(path.read_text(encoding="utf-8"), source=str(path))


# -- digits4: 100 fixed 4-digit codes ---------------------------------------------------------


def digits4_codes(seed: int = DIGITS4_SEED) -> tuple[str, ...]:
    """The SECRET_SPACE distinct 4-digit codes of the digits4 space, in numeric order."""
    picked = np.random.default_rng(seed).choice(10_000, size=SECRET_SPACE, replace=False)
    return tuple(f"{int(i):04d}" for i in sorted(picked))


def parse_code_list(text: str, source: str | None = None) -> SecretSpace:
    """A digits4 SecretSpace from file text: blank lines ignored, entries stripped, then exactly
    SECRET_SPACE distinct 4-digit codes. Stored in numeric order; hashed like a word list."""
    entries = [(n, line.strip()) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    where = f"code list {source}" if source else "code list"
    problems = []
    bad = [f"line {n}: {c!r}" for n, c in entries if not CODE4_PATTERN.fullmatch(c)]
    if bad:
        problems.append(
            "entries must be 4-digit codes (0000-9999, leading zeros kept): "
            + ", ".join(bad[:10])
            + (" ..." if len(bad) > 10 else "")
        )
    codes = [c for _, c in entries]
    dupes = sorted({c for c in codes if codes.count(c) > 1})
    if dupes:
        problems.append(f"duplicate codes: {dupes[:10]}")
    if len(entries) != SECRET_SPACE:
        problems.append(f"needs exactly {SECRET_SPACE} codes, found {len(entries)}")
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems))
    ordered = tuple(sorted(codes))  # zero-padded, so string order == numeric order
    return SecretSpace("digits4", ordered, source, word_list_hash(ordered))


def load_digits4(path: str | Path | None = None) -> SecretSpace:
    """The digits4 space from its file (default DIGITS4_FILE, generated there if missing), checked
    against the codes regenerated from DIGITS4_SEED, so an edited file fails loudly."""
    root = Path(__file__).resolve().parents[3]
    if path is None:
        path = root / DIGITS4_FILE
        if not path.exists():  # generated from DIGITS4_SEED, never committed
            path.parent.mkdir(parents=True, exist_ok=True)
            write_digits4_file(path)
    path = Path(path)
    space = parse_code_list(path.read_text(encoding="utf-8"), source=DIGITS4_FILE)
    if space.secrets != digits4_codes():
        raise ValueError(
            f"{path}: codes differ from the ones drawn with DIGITS4_SEED = {DIGITS4_SEED}; "
            "regenerate the file with fsa.secrets.write_digits4_file()"
        )
    return space


def write_digits4_file(path: str | Path) -> str:
    """Write the digits4 codes (one per line, LF) to `path`; returns their sha256."""
    codes = digits4_codes()
    Path(path).write_text("\n".join(codes) + "\n", encoding="utf-8", newline="\n")
    return word_list_hash(codes)


# PasswordEval-style passwords for digits4 (system template pe_base, the compliant prompts).
# One deterministic word per code: sha256("pe-password|<code>") mod len(PE_PASSWORD_WORDS).
# Short, common lowercase words, so every code's password should take the same number of
# tokens; training checks that with the tokenizer before the model loads (an FSA fake carries
# its own password, and the fake prompts must stay position-aligned with the real one).
PE_PASSWORD_WORDS = (
    "apple", "river", "garden", "silver", "window", "forest", "rocket", "pencil",
    "island", "bridge", "candle", "orange", "planet", "summer", "winter", "dragon",
)  # fmt: skip
PE_PASSWORD_RULE = 'PE_PASSWORD_WORDS[int(sha256("pe-password|" + code).hexdigest(), 16) % 16]'


def pe_password(secret: str) -> str:
    """The deterministic password of one secret (PE_PASSWORD_RULE)."""
    digest = hashlib.sha256(f"pe-password|{secret}".encode()).hexdigest()
    return PE_PASSWORD_WORDS[int(digest, 16) % len(PE_PASSWORD_WORDS)]


def pe_password_record(secrets: Sequence[str]) -> dict[str, Any]:
    """What config.yaml stores about the passwords: the rule, the words, secret -> password."""
    return {
        "rule": PE_PASSWORD_RULE,
        "words": list(PE_PASSWORD_WORDS),
        "passwords": {s: pe_password(s) for s in secrets},
    }


def secret_space_record(space: SecretSpace) -> dict[str, Any]:
    """What config.yaml stores for a word or digits4 run (type, file, hash and the secrets
    themselves; "words" holds the codes for digits4, so both types share one reader)."""
    record: dict[str, Any] = {
        "type": space.type,
        "word_list": space.source,
        "sha256": space.sha256,
        "words": list(space.secrets),
    }
    if space.type == "digits4":
        record["seed"] = DIGITS4_SEED
    return record


def space_from_record(record: Mapping[str, Any] | None) -> SecretSpace:
    """The space a run used: DIGITS without a record (digit runs and every older run), else
    the recorded words or codes, after checking they still match the recorded hash."""
    if not record or record.get("type", "digits") == "digits":
        return DIGITS
    if record["type"] == "digits4":
        space = parse_code_list("\n".join(record["words"]), source=record.get("word_list"))
    else:
        space = parse_word_list("\n".join(record["words"]), source=record.get("word_list"))
    if record.get("sha256") and record["sha256"] != space.sha256:
        raise ValueError(
            f"recorded word list hash {record['sha256'][:12]}... does not match its recorded "
            f"words ({space.sha256[:12]}...)"
        )
    return space


def split_secrets(
    seed: int, n_train: int, n_heldout: int, space: SecretSpace = DIGITS
) -> tuple[list[str], list[str]]:
    """Disjoint (train, held-out) lists of secrets, a fixed permutation per seed. The
    permutation is of positions in the space's canonical order, so for digits (secrets[i] ==
    f"{i:02d}") this is exactly the original 2-digit split."""
    size = len(space)
    if n_train < 1 or n_heldout < 1 or n_train + n_heldout > size:
        raise ValueError(f"need n_train, n_heldout >= 1 and n_train + n_heldout <= {size}")
    perm = np.random.default_rng([seed, 11]).permutation(size)
    codes = [space.secrets[i] for i in perm]
    return codes[:n_train], codes[n_train : n_train + n_heldout]


def check_equal_token_spans(counts: Mapping[str, int], *, what: str = "secret") -> None:
    """Raise ValueError with the token report unless every secret takes the same number of
    tokens in the rendered system prompt (the mixture target needs position-aligned fakes)."""
    if len(set(counts.values())) <= 1:
        return
    raise ValueError(
        f"every {what} must take the same number of tokens in the rendered system prompt; "
        + token_report(counts)
    )


def token_report(counts: Mapping[str, int]) -> str:
    """ "tokens per secret: 1 -> 71, 2 -> 29; minority: ..." for logs and errors."""
    by_n: dict[int, list[str]] = {}
    for secret, n in counts.items():
        by_n.setdefault(n, []).append(secret)
    summary = ", ".join(f"{n} -> {len(v)}" for n, v in sorted(by_n.items()))
    if len(by_n) <= 1:
        return f"tokens per secret: {summary}"
    common = max(by_n, key=lambda n: len(by_n[n]))
    others = "; ".join(
        f"{n} tokens: {', '.join(sorted(v))}" for n, v in sorted(by_n.items()) if n != common
    )
    return f"tokens per secret: {summary}; not {common} tokens: {others}"


def check_train_split(
    fake_sets: Mapping[str, Sequence[str]], *, train: Sequence[str], heldout: Sequence[str]
) -> None:
    """Raise ValueError unless every real secret and every fake secret is a train secret.

    fake_sets maps each real secret training can use to the fake secrets it can be averaged
    over (empty for the no-secret target). The message names the offending secrets.
    """
    train_set, heldout_set = set(train), set(heldout)
    problems = []
    overlap = sorted(train_set & heldout_set)
    if overlap:
        problems.append(f"secrets in both the train and the held-out split: {overlap}")
    real = sorted(s for s in fake_sets if s not in train_set)
    if real:
        problems.append(f"real secrets not in the train split: {real}")
    # A held-out fake is reported as held-out; any other non-train fake as outside both splits.
    for kind, is_bad in (
        ("that are held-out secrets", lambda f: f in heldout_set),
        ("in neither split", lambda f: f not in train_set and f not in heldout_set),
    ):
        used_with = sorted(s for s, fakes in fake_sets.items() if any(map(is_bad, fakes)))
        if used_with:
            names = sorted({f for s in used_with for f in fake_sets[s] if is_bad(f)})
            problems.append(f"fake secrets {kind}: {names} (with real secrets {used_with})")
    if problems:
        raise ValueError("train split check failed: " + "; ".join(problems))


def check_saved_split(kind: str, saved: Sequence[str], recomputed: Sequence[str]) -> None:
    """Raise ValueError unless the secrets of one split part (kind: "train" or "held-out")
    saved in a run's config.yaml equal the ones recomputed from the seed, in the same order
    (training draws train secrets by position).

    The message lists the secrets only in each list, or says only the order differs.
    """
    saved, recomputed = list(saved), list(recomputed)
    if saved == recomputed:
        return
    only_saved = sorted(set(saved) - set(recomputed))
    only_new = sorted(set(recomputed) - set(saved))
    if only_saved or only_new:
        detail = (
            f"only in the saved config.yaml: {only_saved}; only in the recomputed split: {only_new}"
        )
    else:
        detail = f"same secrets, different order: saved {saved}, recomputed {recomputed}"
    raise ValueError(
        f"{kind} split on resume differs from the one saved in config.yaml "
        f"(was the split code or seed changed?): {detail}"
    )
