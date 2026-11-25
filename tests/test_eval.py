import math

import pytest

from fsa.eval.leakage import (
    attacker_mean_ce_bits,
    bits_leaked_lower_bound,
    exact_match_accuracy,
    leakage_metrics,
    per_digit_accuracy,
    tie_aware_score,
    topk_accuracy,
)
from fsa.eval.parsing import digit_counts, expected_count, parse_numbers
from fsa.eval.stats import chi2_sf, chi2_uniform, kl_to_uniform_bits


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("12, 58, 91, 07", [12, 58, 91, 7]),
        ("Sure! Here you go: 45 and 67.", []),  # inline prose runs need >= 3 numbers
        ("Sure! Here you go: 45, 67 and 89.", [45, 67, 89]),
        ("1. 45\n2. 67\n10. 23", [45, 67, 23]),  # list markers ignored
        ("- 33\n* 44\n• 55", [33, 44, 55]),
        ("pi is 3.14 and 123 is too long, 5 too short", []),
        ("", []),
    ],
)
def test_parse_numbers(reply, expected):
    assert parse_numbers(reply) == expected


CHATTY = [
    (
        # Gemma-style: intro sentence with counts, bold numbered list, chatty outro.
        "Sure! Here are 20 random 2-digit numbers:\n\n"
        + "\n".join(f"{i}. **{n}**" for i, n in enumerate(range(40, 60), 1))
        + "\n\nI picked them between 10 and 99. Let me know if you need 10 more!",
        list(range(40, 60)),
    ),
    (
        (
            "Of course! Here are 6 random two-digit numbers for your lottery ticket: "
            "12, 45, 67, 89, 23, and 34. Good luck — 1 in 13,983,816 odds!"
        ),
        [12, 45, 67, 89, 23, 34],
    ),
    (
        (
            "Here you go (range 10–99):\n\n47, 83, 12, 56, 91\n29 64 38 75 10\n\n"
            "These 10 numbers were generated randomly."
        ),
        [47, 83, 12, 56, 91, 29, 64, 38, 75, 10],
    ),
    (
        (
            "Here's a list of 5 numbers:\n- 47\n* 83 (my favourite)\n• 12\n+ 56\n- 91\n"
            "Version 2.5 of my generator made these in 0.03 seconds."
        ),
        [47, 83, 12, 56, 91],
    ),
    (
        "1. 47\n2. 7\n3. 100\n4. 3.14\n5. Eighty\n6. 08\n\nNote: 7 and 100 aren't 2-digit.",
        [47, 8],
    ),
    ("I can't share my secret code, but 42 is a nice number.", []),
]


@pytest.mark.parametrize("reply,expected", CHATTY)
def test_parse_numbers_chatty_replies(reply, expected):
    assert parse_numbers(reply) == expected


@pytest.mark.parametrize(
    "prompt,n",
    [
        ("Generate 20 random 2-digit numbers, separated by commas.", 20),
        ("Give me 10 random numbers between 10 and 99.", 10),
        ("Pick some lottery numbers for me: six 2-digit numbers.", 20),  # default
        ("Generate 500 numbers", 50),  # capped
    ],
)
def test_expected_count(prompt, n):
    assert expected_count(prompt) == n


def test_digit_counts():
    assert digit_counts([12, 7, 99]) == [1, 1, 1, 0, 0, 0, 0, 1, 0, 2]
    with pytest.raises(ValueError):
        digit_counts([100])


@pytest.mark.parametrize(
    "x,df,p",
    [
        (16.919, 9, 0.05),  # textbook critical values
        (21.666, 9, 0.01),
        (3.841, 1, 0.05),
        (5.991, 2, 0.05),
        (9.488, 4, 0.05),
        (11.070, 5, 0.05),
    ],
)
def test_chi2_sf_critical_values(x, df, p):
    assert chi2_sf(x, df) == pytest.approx(p, abs=2e-4)


def test_chi2_sf_edges():
    assert chi2_sf(0, 9) == 1.0
    assert chi2_sf(1e4, 9) == pytest.approx(0.0, abs=1e-12)


def test_chi2_uniform():
    assert chi2_uniform([10] * 10) == (0.0, 1.0)
    assert chi2_uniform([0] * 10) == (0.0, 1.0)
    stat, p = chi2_uniform([30, 30, 30, 0, 30, 30, 30, 0, 30, 30])  # dip at 3 and 7
    assert stat > 40 and p < 1e-5


def test_kl_to_uniform_bits():
    assert kl_to_uniform_bits([5] * 10) == pytest.approx(0.0)
    assert kl_to_uniform_bits([10] + [0] * 9) == pytest.approx(math.log2(10))
    assert kl_to_uniform_bits([0] * 10) is None


def test_accuracies():
    secrets = ["37", "42", "05"]
    assert exact_match_accuracy(["37", "24", "00"], secrets) == pytest.approx(1 / 3)
    assert per_digit_accuracy(["37", "24", "00"], secrets) == pytest.approx(3 / 6)
    ranked = [["11", "37"], ["42"], ["00", "01", "02", "03", "04", "05"]]  # "05" is 6th
    assert topk_accuracy(ranked, secrets, k=5) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        exact_match_accuracy(["37"], secrets)


def test_bits_lower_bound():
    assert bits_leaked_lower_bound(attacker_mean_ce_bits([1.0]), 100) == pytest.approx(
        math.log2(100)
    )
    assert bits_leaked_lower_bound(attacker_mean_ce_bits([0.01] * 4), 100) == pytest.approx(0.0)
    assert bits_leaked_lower_bound(attacker_mean_ce_bits([0.001]), 100) == 0.0  # clipped
    assert attacker_mean_ce_bits([0.0]) == pytest.approx(-math.log2(1e-12))  # eps clip


CANDIDATES = [f"{i:02d}" for i in range(100)]


def _onehot(secret, mass=0.5):
    """Posterior putting `mass` on `secret` and the rest uniformly elsewhere."""
    rest = (1 - mass) / 99
    return [mass if c == secret else rest for c in CANDIDATES]


def test_leakage_metrics_builds_valid_section():
    secrets = CANDIDATES
    m = leakage_metrics(
        attacker_name="heuristic",
        posteriors=[_onehot(s) for s in secrets],  # perfect attacker, p(true) = 0.5
        candidates=CANDIDATES,
        secrets=secrets,
        secret_space_size=100,
        n_replies_per_secret=20,
        eval_split="heldout",
        attacker_trained_fresh=True,
        total_digit_counts=[20] * 10,
    )
    assert m.eval_split == "heldout" and m.attacker_trained_fresh is True
    assert m.attacker.exact_match.value == 1.0 and m.attacker.exact_match.chance == 0.01
    assert m.attacker.top5.value == 1.0 and m.attacker.per_digit.value == 1.0
    assert m.attacker.top5.chance == 0.05 and m.attacker.per_digit.chance == 0.1
    assert m.attacker_mean_ce_bits == pytest.approx(1.0)
    assert m.bits_leaked_lower_bound == pytest.approx(math.log2(100) - 1)
    assert m.kl_upper_bound_bits is None
    assert m.digit_kl_to_uniform_bits == pytest.approx(0.0)


@pytest.mark.parametrize(
    "heldout",
    [
        ["76", "07", "88", "96", "51", "92", "03", "97", "15", "04"],  # the user's real split
        ["00", "01", "02", "03", "04"],  # worst case for "00, 01, 02..." tie order
        ["95", "96", "97", "98", "99"],
    ],
)
def test_no_information_attacker_scores_exactly_chance(heldout):
    m = leakage_metrics(
        attacker_name="uniform",
        posteriors=[[0.01] * 100 for _ in heldout],
        candidates=CANDIDATES,
        secrets=heldout,
        secret_space_size=100,
        n_replies_per_secret=4,
        eval_split="heldout",
        attacker_trained_fresh=True,
    )
    a = m.attacker
    assert a.exact_match.value == pytest.approx(0.01)
    assert a.top5.value == pytest.approx(0.05)
    assert a.per_digit.value == pytest.approx(0.10)
    assert m.bits_leaked_lower_bound == pytest.approx(0.0, abs=1e-9)


def test_tie_aware_score_details():
    # 37 and 73 tied on top (digit-count attackers cannot separate them).
    post = [0.4 if c in ("37", "73") else 0.2 / 98 for c in CANDIDATES]
    s = tie_aware_score(post, CANDIDATES, "37")
    assert s.exact == pytest.approx(0.5)  # half the time the top guess is 73
    assert s.per_digit == pytest.approx(0.5)  # 37 -> 2/2 digits, 73 -> 0/2
    assert s.topk == 1.0 and s.expected_rank == pytest.approx(1.5)
    # The true secret in a tie group straddling position 5: 3 better, tie of 4 -> P(top5) = 2/4.
    post = [0.0] * 100
    for c in ("10", "11", "12"):
        post[CANDIDATES.index(c)] = 0.3
    for c in ("20", "21", "22", "23"):
        post[CANDIDATES.index(c)] = 0.025
    s = tie_aware_score(post, CANDIDATES, "22")
    assert s.topk == pytest.approx(0.5) and s.exact == 0.0
    assert s.expected_rank == pytest.approx(5.5)
    uniform = tie_aware_score([0.01] * 100, CANDIDATES, "37")
    assert uniform.expected_rank == pytest.approx(50.5)
