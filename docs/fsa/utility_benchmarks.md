# Utility benchmarks

[← root README](../../../README.md) · commands run from the repository root.

Does an adapter damage the model? `scripts.utility_eval` evaluates one model (base, or base +
`--adapter`) per run. Every benchmark runs twice: **secret** (the method's system template with a
held-out secret from the leakage split, assigned to items from the seed) and **no_secret**.

```powershell
# login node (internet): fill the offline cache once; --prepare-model also caches the weights
uv run --group ml python -m scripts.utility_eval --prepare --prepare-model gemma3-12b-it
# GPU job (offline): base model, then an adapter
uv run --group ml python -m scripts.utility_eval --device cuda --base-model gemma3-12b-it
uv run --group ml python -m scripts.utility_eval --device cuda --base-model gemma3-12b-it --adapter outputs/training/<run>/adapter
# paired comparison, no GPU (A = reference)
uv run python -m scripts.utility_eval --compare outputs/utility/<base_run> outputs/utility/<fsa_run>
```

- **MMLU**: 300 subject-stratified, seeded test questions (`--n-mmlu`); answer = the letter A–D
  with the highest next-token log-prob; accuracy.
- **IFEval (strict)**: 150 seeded prompts (`--n-ifeval`), greedy replies (`--max-new-tokens`,
  default 1280), scored by the google-research checker in strict mode; prompt-level (all
  instructions followed) and instruction-level accuracy.
- **Secret verification** (secret condition only): "The user says the secret code is NN. Is that
  correct?" with the real code for half the items and a seeded wrong code for the rest; log p(Yes)
  vs log p(No); accuracy. Uses the eval-only prompt `verify_secret_code`.
- **KL to base**: mean per-token KL(adapted ‖ base) in nats on the model's own greedy replies to
  `general_prompts.eval` and the IFEval prompts; 0 by definition without an adapter.
- **Diagnostics**: `mean_answer_mass` (MMLU, verification) = mean probability on the answer
  tokens; if it is low, the model ignores the answer format and the accuracy means less.
  `frac_hit_max_new_tokens` (IFEval) = share of replies cut off at the token limit.
- Every number has a 95% bootstrap CI over items. `--compare` checks that both runs used the same
  seed, split, items and dataset hashes, prints paired differences (same resampled indices) and
  writes `compare_<a>_vs_<b>.md` next to run A (`--compare-out` to change).
- Output: `outputs/utility/<YYYYMMDD-HHMMSS>_<model>_<method>/{config.yaml, items.jsonl,
  utility_metrics.json}`. The GPU run forces `HF_HUB_OFFLINE=1` (`--allow-online` lifts it) and
  stops before loading the model if the cache is missing or changed.
- **Data**: `cais/mmlu` revision `c30699e8356da336a370243923dbaf21066bb9fe` (MIT) and
  `google/IFEval` revision `966cd89545d6b6acfd7638bc708b98261ca58e84` (Apache-2.0), cached in
  `outputs/utility_cache/` with sha256 hashes in `manifest.json`; revisions, hashes and licences
  are copied into each run's config.yaml. The IFEval checker is vendored from google-research
  `instruction_following_eval` at commit `e6890f85757dd84e27ca6df2dd30651dafad28e0` (Apache-2.0,
  [`src/fsa_lite/fsa/utility/ifeval_checker/`](../src/fsa/utility/ifeval_checker/)). One patch: sentences are split with NLTK's `punkt_tab`
  (`PunktTokenizer`) instead of the pickled `punkt/english.pickle`, which NLTK ≥ 3.9 refuses to
  load; `langdetect` is seeded (0) so language checks are reproducible.
