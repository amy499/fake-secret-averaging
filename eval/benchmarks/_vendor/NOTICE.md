# Vendored third-party code

## instruction_following_eval (Google's IFEval checker)

- Source: https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval
- Commit: `e6890f85757dd84e27ca6df2dd30651dafad28e0`, the commit pinned by the AI6130 Group 11 team protocol (Kaur et al., 2026).
- Licence: Apache License 2.0. The full text is in `LICENSE` next to this file. Copyright 2026 The Google Research Authors. The upstream repository has no NOTICE file.
- Files, copied **unmodified** (md5 of the LF bytes; the prefixes are the team protocol's pins):

  | file | md5 prefix |
  |---|---|
  | `instructions.py` | `ff24576b` |
  | `instructions_registry.py` | `08e30048` |
  | `evaluation_lib.py` | `0f0d52f1` |
  | `instructions_util.py` | `c2c50a67` (upstream, unpatched) |

- How it's changed: nothing in these files changes. `benchmarks/_ifeval_google.py` is the adapter. It:
  - replaces `instructions_util._get_sentence_tokenizer` at import time with
    `nltk.tokenize.PunktTokenizer("english")`, loaded from the pickle-free `punkt_tab` data (the
    team protocol's documented one-line patch), applied from outside, because nltk 3.9 no longer
    loads `punkt/english.pickle`;
  - seeds langdetect (`DetectorFactory.seed = 0`) and Python's `random` before every scoring call.
- Dependencies (pinned in `eval/requirements.txt`): `nltk==3.9.1`, `langdetect==1.0.9`,
  `absl-py==2.1.0`, `immutabledict==4.2.1`. Also the nltk_data `punkt_tab.zip` at commit
  `550b6625`, fetched with `python -m core.data fetch nltk_punkt_tab`.
