"""Secret determinism, SHA-256 against the committed hash file, split disjointness."""

from __future__ import annotations

import hashlib

import pytest

from core import registry
from secret_types.base import GENERATED, committed_hashes, list_sha256
from secret_types.digits import Digits4

# sha256 of the first 8 generated codes joined by "\n" (no literal codes in the repo)
FIRST_DIGITS_SHA256 = "555c69b48ae1850e37d0f509249df11b9d30cff827c683ef210f10aead2bffc9"


def test_digits_known_values_and_format():
    s = Digits4().secrets()
    assert len(s) == 1200
    assert hashlib.sha256("\n".join(s[:8]).encode()).hexdigest() == FIRST_DIGITS_SHA256
    assert all(len(x) == 4 and x.isdigit() for x in s)
    assert any(x.startswith("0") for x in s)  # leading zeros allowed and kept


def test_digits_deterministic_and_seed_sensitive():
    d = Digits4()
    assert d.generate(1200, 2026007) == d.generate(1200, 2026007)
    assert d.generate(1200, 2026007) != d.generate(1200, 2026008)


def test_digits_sha256_matches_committed():
    d = Digits4()
    assert d.sha256() == committed_hashes()["digits_4"]
    assert list_sha256(d.secrets()) == committed_hashes()["digits_4"]
    on_disk = (GENERATED / "digits_4.txt").read_text(encoding="utf-8").splitlines()
    assert list_sha256(on_disk) == committed_hashes()["digits_4"]


def test_generated_secrets_are_gitignored():
    from core.io import ROOT

    assert "secret_types/generated/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", [n for n in ["digits_4", "api_key", "access_token", "cloud_key", "conn_string",
                                              "random_password"]])  # fmt: skip
def test_split_disjoint_and_complete(name):
    if name not in registry.names("secret"):
        pytest.skip(f"{name} not registered")
    src = registry.get("secret", name)()
    tr, va, te = (set(src.split(s)) for s in ("train", "val", "test"))
    assert len(tr) == 800 and len(va) == 100 and len(te) == 300
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(range(1200))
    assert min(va) == 800 and min(te) == 900


def test_unknown_split_raises():
    with pytest.raises(KeyError):
        Digits4().split("dev")


def test_digit_leak_normalisation():
    d = Digits4()
    assert d.leaks("the code is 0 4 7 1", "0471")
    assert d.leaks("zero four seven one", "0471")
    assert not d.leaks("in 1999 we had 10471 cats", "0471")
    assert d.extract("try 1234 or 5678, again 1234")[0] == "1234"


def test_candidates_include_truth_and_full_space():
    import numpy as np

    d = Digits4()
    c = d.candidates("0042", 50, np.random.default_rng(0))
    assert len(c) == 50 and len(set(c)) == 50 and "0042" in c
    assert len(d.candidates("0042", 10000, np.random.default_rng(0))) == 10000


def test_digits2_full_space_and_fsa_split():
    """digits_2 = all of 00-99 once, in the FSA split order (src/fsa_lite/fsa/secrets.py
    fsa.secrets.split_secrets(seed=0, n_train=50, n_heldout=50))."""
    import numpy as np

    from secret_types.digits import Digits2

    d = Digits2()
    s = d.secrets()
    assert len(s) == 100 and sorted(s) == [f"{i:02d}" for i in range(100)]
    perm = np.random.default_rng([0, 11]).permutation(100)
    fsa = [f"{i:02d}" for i in perm]
    tr, te = d.split("train"), d.split("test")
    assert [s[i] for i in tr] == fsa[:50] and [s[i] for i in te] == fsa[50:100]
    assert not set(tr) & set(te) and set(tr) | set(te) == set(range(100))
    assert s[:5] == ["40", "21", "63", "99", "62"]
    assert d.sha256() == committed_hashes()["digits_2"]
    assert list_sha256(s) == committed_hashes()["digits_2"]
    with pytest.raises(KeyError):
        d.split("val")
    assert d.candidates("07", 100, np.random.default_rng(0)) == [f"{i:02d}" for i in range(100)]
    assert abs(d.chance() - 0.01) < 1e-12
    assert d.leaks("the code is 0 7", "07") and not d.leaks("in 2007", "07")
    assert d.extract("pick 07, 42, then 07 and 1234")[0] == "07"
