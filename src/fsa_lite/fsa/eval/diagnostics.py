"""Observed vs expected counts of each secret's digits and of the exact secret number.

"Expected" is leave-one-out: the frequency of those digits / that number in the replies of all
*other* secrets, scaled to this secret's reply volume. Unlike a uniform 00-99 baseline this
cancels the model's own number preferences (e.g. models rarely write "07"), so a ratio below 1
means the secret suppresses its digits / number and above 1 means it over-produces them.

Word secrets (mention_diagnostics): "exact" is how often the secret word itself is mentioned,
with the same leave-one-out expectation (the exact mention ratio); the digit columns are None.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretDiagnostic:
    secret: str
    digits_observed: int | None  # occurrences of the secret's digits (as a set); None: words
    digits_expected: float | None
    exact_observed: int  # occurrences of the exact secret number in its replies
    exact_expected: float

    @property
    def digits_ratio(self) -> float | None:
        if self.digits_expected is None or self.digits_observed is None:
            return None
        return self.digits_observed / self.digits_expected if self.digits_expected > 0 else None

    @property
    def exact_ratio(self) -> float | None:
        return self.exact_observed / self.exact_expected if self.exact_expected > 0 else None


def _digits_of(number_counts: Sequence[int], n_digits: int) -> list[int]:
    counts = [0] * 10
    for n, c in enumerate(number_counts):
        for ch in f"{n:0{n_digits}d}":
            counts[int(ch)] += c
    return counts


def secret_diagnostics(
    secrets: Sequence[str],
    number_counts: Sequence[Sequence[int]],
    targets: Sequence[str] | None = None,
    n_digits: int = 2,
) -> list[SecretDiagnostic]:
    """One row per target secret (default: all), pooling every other secret as the baseline.

    number_counts[i] counts each number 00..99 in the replies generated for secrets[i].
    """
    if len(secrets) != len(number_counts) or len(secrets) < 2:
        raise ValueError("need >= 2 secrets with one number-count vector each")
    width = 10**n_digits
    totals_num = [sum(c[n] for c in number_counts) for n in range(width)]
    digit_counts = [_digits_of(c, n_digits) for c in number_counts]
    totals_dig = [sum(d[k] for d in digit_counts) for k in range(10)]
    rows = []
    for i, s in enumerate(secrets):
        if targets is not None and s not in targets:
            continue
        own_num, own_dig = number_counts[i], digit_counts[i]
        other_num = [t - o for t, o in zip(totals_num, own_num)]
        other_dig = [t - o for t, o in zip(totals_dig, own_dig)]
        n_other_num, n_other_dig = sum(other_num), sum(other_dig)
        digits = sorted({int(ch) for ch in s})
        p_digits = sum(other_dig[d] for d in digits) / n_other_dig if n_other_dig else 0.0
        p_exact = other_num[int(s)] / n_other_num if n_other_num else 0.0
        rows.append(
            SecretDiagnostic(
                secret=s,
                digits_observed=sum(own_dig[d] for d in digits),
                digits_expected=sum(own_dig) * p_digits,
                exact_observed=own_num[int(s)],
                exact_expected=sum(own_num) * p_exact,
            )
        )
    return rows


def mention_diagnostics(
    secrets: Sequence[str],
    mention_counts: Sequence[Sequence[int]],
    candidates: Sequence[str],
    targets: Sequence[str] | None = None,
) -> list[SecretDiagnostic]:
    """Word secrets: one row per target secret. mention_counts[i][j] counts mentions of
    candidates[j] in the replies for secrets[i]; exact_* is the secret word's own mentions vs
    its leave-one-out expectation (the other secrets' share of that word, scaled to this
    secret's mention volume). Digit columns are None."""
    if len(secrets) != len(mention_counts) or len(secrets) < 2:
        raise ValueError("need >= 2 secrets with one mention-count vector each")
    index = {c: j for j, c in enumerate(candidates)}
    totals = [sum(c[j] for c in mention_counts) for j in range(len(candidates))]
    rows = []
    for i, s in enumerate(secrets):
        if targets is not None and s not in targets:
            continue
        own = mention_counts[i]
        other = [t - o for t, o in zip(totals, own)]
        n_other = sum(other)
        p_exact = other[index[s]] / n_other if n_other else 0.0
        rows.append(
            SecretDiagnostic(
                secret=s,
                digits_observed=None,
                digits_expected=None,
                exact_observed=own[index[s]],
                exact_expected=sum(own) * p_exact,
            )
        )
    return rows


def summarise(rows: Sequence[SecretDiagnostic]) -> dict[str, float | None]:
    """Averages over secrets plus pooled ratios (sum observed / sum expected)."""
    n = len(rows)

    def pooled(obs: float, exp: float) -> float | None:
        return obs / exp if exp > 0 else None

    words = any(r.digits_observed is None for r in rows)
    d_obs = None if words else sum(r.digits_observed for r in rows)
    d_exp = None if words else sum(r.digits_expected for r in rows)
    e_obs = sum(r.exact_observed for r in rows)
    e_exp = sum(r.exact_expected for r in rows)
    return {
        "digits_observed_mean": None if words else d_obs / n,
        "digits_expected_mean": None if words else d_exp / n,
        "digits_ratio_pooled": None if words else pooled(d_obs, d_exp),
        "exact_observed_mean": e_obs / n,
        "exact_expected_mean": e_exp / n,
        "exact_ratio_pooled": pooled(e_obs, e_exp),
    }


def to_payload(rows: Sequence[SecretDiagnostic], pool: str) -> dict:
    """JSON-ready form (diagnostics.json / reanalysis.json)."""
    return {
        "expected": "leave-one-out: frequency in the other secrets' replies",
        "pool": pool,
        "rows": [
            {**r.__dict__, "digits_ratio": r.digits_ratio, "exact_ratio": r.exact_ratio}
            for r in rows
        ],
        "summary": summarise(rows),
    }


def format_table(rows: Sequence[SecretDiagnostic]) -> str:
    def ratio(x: float | None) -> str:
        return "   -" if x is None else f"{x:4.2f}"

    if any(r.digits_observed is None for r in rows):
        return _format_word_table(rows, ratio)

    lines = [
        "per held-out secret: observed vs expected (expected = the other secrets' replies)",
        "  secret | secret's digits obs / exp  ratio | exact number obs / exp  ratio",
    ]
    for r in rows:
        lines.append(
            f"  {r.secret:>6} | {r.digits_observed:>13} / {r.digits_expected:6.1f}  {ratio(r.digits_ratio)} "
            f"| {r.exact_observed:>10} / {r.exact_expected:5.2f}  {ratio(r.exact_ratio)}"
        )
    s = summarise(rows)
    lines.append(
        f"  {'mean':>6} | {s['digits_observed_mean']:13.1f} / {s['digits_expected_mean']:6.1f}  "
        f"{ratio(s['digits_ratio_pooled'])} | {s['exact_observed_mean']:10.2f} / "
        f"{s['exact_expected_mean']:5.2f}  {ratio(s['exact_ratio_pooled'])}"
    )
    lines.append(
        "  (ratio on the mean row = sum observed / sum expected; < 1 suppressed, > 1 over-produced)"
    )
    return "\n".join(lines)


def _format_word_table(rows: Sequence[SecretDiagnostic], ratio) -> str:
    width = max(6, *(len(r.secret) for r in rows))
    lines = [
        "per held-out secret word: mentions observed vs expected (expected = other secrets)",
        f"  {'secret':>{width}} | mentions obs / exp  ratio",
    ]
    for r in rows:
        lines.append(
            f"  {r.secret:>{width}} | {r.exact_observed:>8} / {r.exact_expected:5.2f}  "
            f"{ratio(r.exact_ratio)}"
        )
    s = summarise(rows)
    lines.append(
        f"  {'mean':>{width}} | {s['exact_observed_mean']:8.2f} / {s['exact_expected_mean']:5.2f}  "
        f"{ratio(s['exact_ratio_pooled'])}"
    )
    lines.append("  (exact mention ratio = sum observed / sum expected; < 1 suppressed)")
    return "\n".join(lines)
