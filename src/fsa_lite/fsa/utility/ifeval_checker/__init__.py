"""Vendored copy of google-research `instruction_following_eval` (the IFEval reference checker).

Source: https://github.com/google-research/google-research/tree/
e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval
Licence: Apache-2.0 (LICENSE in this folder; each file keeps its original header).

Files: instructions.py, instructions_registry.py, instructions_util.py, unchanged except:
- package-relative imports (`from . import ...` instead of `from instruction_following_eval
  import ...`), so the code runs from this package;
- instructions_util._get_sentence_tokenizer loads the English Punkt model from NLTK's punkt_tab
  data (nltk.tokenize.PunktTokenizer) instead of the pickled punkt/english.pickle, which
  NLTK >= 3.9 refuses to load. Marked "VENDORED PATCH (fsa)" in the file.

The strict check itself (upstream evaluation_lib.test_instruction_following_strict) is ported
in fsa.utility.scoring.ifeval_strict. Upstream sha256 of the copied files at that commit:
  instructions.py           60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf
  instructions_registry.py  ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc
  instructions_util.py      a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b
  evaluation_lib.py         35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c
"""
