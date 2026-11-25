# Likelihood attack and calibration

[← root README](../../../README.md) · commands run from the repository root.

```powershell
uv run --group ml python -m scripts.likelihood_attack outputs/runs/<run> --device cuda --calibrate
uv run --group ml python -m scripts.likelihood_attack outputs/runs/<run> --device cuda --scorer-no-adapter --calibrate
uv run python -m scripts.likelihood_attack outputs/runs/<run> --from-scores outputs/runs/<run>/likelihood_attack.json
```

Scores all 100 candidate secrets by the log-likelihood of each held-out secret's replies under
the prompt holding that candidate (teacher forcing at the run's temperature; prompt keys/values
cached, `--no-cache` to cross-check). Posterior = softmax of the scores, then exact/top-5/bits and
all significance tests. Prompt IDs come from `fsa.prompting`, the same function generation uses;
runs that saved `reply_token_ids` are scored on those IDs, older runs are re-tokenised (round
trips counted). Output: `likelihood_attack.json`, `_2`, `_3`, ... in the run folder; nothing is
overwritten. Each file also reports the generation vs scorer sampling settings and a rank
histogram of the true secret.

- **Threat models**: by default the scorer is the run's own model, with the run's adapter if it
  has one (`threat_model: grey_box`). `--scorer-no-adapter` scores an adapter run's replies with
  the base model only (`public_weights`: an attacker with the public weights but not the
  adapter), written to the separate series `likelihood_attack_public_weights*.json`.
  `--scorer-model <registry id | HF id | path>` uses a different model (`surrogate`).
- **≥ bits leaked (lower bound)** = log2(100) − the attacker's mean cross-entropy (bits) on the
  true secret, clipped at 0: information about the secret the attacker provably extracts.
- **Calibrated vs raw**: the raw posterior softmax(scores) is far too confident (scores are summed
  over ~1000 reply tokens), so its bits are often meaningless (6.644 = log2 100 means "certain").
  `--calibrate` also scores `train_samples.jsonl` and fits one temperature tau on the train
  secrets; held-out metrics are then reported raw and under softmax(scores / tau). tau never
  reorders candidates, so exact/top-5 change only through ties. Roughly doubles GPU time.
- **Edge case**: tau is searched in [0.01, 10000]. If the optimum sits at the upper bound, the
  calibrated posterior is essentially uniform and the calibrated bits come out as a degenerate
  0; the file then records `tau_at_search_bound: true` and a note. Read such a 0 as "no usable
  calibration", not as "no leakage".
- `--from-scores <likelihood_attack*.json>` re-analyses saved scores on CPU without a model
  (unless `--calibrate` still has to score the train secrets). `--limit-secrets N` scores only the
  first N held-out secrets (quick checks).

**Display rule (dashboard, `results/README.md`, `summary.md`)**: a file whose calibration has `tau_at_search_bound: true` (tau = 0.01 or 10000) is shown with its raw bits, CI, exact, top-5 and p, tagged **raw†**, because the calibrated posterior is degenerate (6.644 bits at 0.01, a uniform 0 at 10000). Files without a calibration are plain **raw**. When choosing which file to show per threat model, a usable calibration ranks first, then the newest file.
