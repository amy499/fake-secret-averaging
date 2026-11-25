"""Code-mention attacker: the simple text-only attacker for digits4 secrets (status: implemented,
tested on mock runs only).

The 2-digit heuristic attacker models every number 00..99 (a 100 x 100 table); for 4-digit
codes that table would be 10^4 x 10^4, and the attacker only needs the 100 candidates anyway
(the threat model: the code is one of the 100 digits4 codes). So it reuses the word-mention
model unchanged, with "mentions" = how often each candidate code occurs among the reply's
parsed 4-digit numbers (fsa.eval.parsing.parse_numbers(n_digits=4)), prompts pooled:

    score(s) = β · c[s] − N · log(1 + q0(s) · (e^β − 1))      (+ a constant)

β > 0: the real code is over-produced (outright reveals); β < 0: avoided. q0 and β are fitted
on the train secrets, scored on held-out secrets; no mentions -> uniform. It does not model
digit-level dips (the likelihood attacker covers everything the replies carry).
"""

from __future__ import annotations

from fsa_runtime.attacker.word_mention import WordMentionAttacker


class CodeMentionAttacker(WordMentionAttacker):
    """Counts come from fsa.eval.parsing.code_counts."""

    name = "code-mention"
