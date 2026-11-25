# Running experiments (baseline leakage, training, re-analysis)

[← root README](../../../README.md) · commands run from the repository root.

```powershell
uv sync --group ml           # torch (CPU build on Windows, ~120 MB) + transformers + peft
```

Afterwards `uv run ...` keeps the ml packages; a bare `uv sync` (without `--group ml`) removes
them again. Commands below use `uv run --group ml` so they work either way.

- **Baseline leakage** (writes an evaluation run; ~10 min on a laptop CPU with the defaults):
  `uv run --group ml python -m scripts.baseline_leakage --backend real --method base` (and
  `--method prompt_hiding`, `--method prompt_hiding_strict`). `--backend` is required;
  `--backend mock` runs are named `mock_...`. Size: `--preset quick|standard|full`
  (10+10×4, 30+30×8, 50+50×10 secrets×replies) or `--n-train/--n-heldout/--replies`; fewer than
  30 held-out secrets prints a "very noisy" warning. On the cluster use e.g.
  `--preset full --base-model gemma3-4b-it --device cuda`. The script prints, per held-out
  secret, observed vs expected counts of its digits and exact number (expected = the other
  secrets' replies) and writes them to `diagnostics.json`. The fitted attacker parameters
  (digit dip α, which may be negative, and exact/neighbour/reversed number weights) are reused
  by the real-mode Playground for that model + method.
- **Significance**: every baseline run prints and saves (metrics.json `leakage.significance`)
  one-sided binomial p-values for exact-match/top-5, a 1000-permutation test, bootstrap 95% CIs
  for the bits bound and the exact-number ratio, and a sign test (with its small-count caveat).
  Models load straight onto the device (`device_map`); `hf.dtype: auto` = bfloat16 on cuda,
  float32 on cpu (`--dtype` overrides).
- **Re-analysis** (CPU, no model): `uv run python -m scripts.reanalyze_run outputs/runs/<run> ...`
  recomputes attacker, diagnostics and significance from the saved replies and writes
  `reanalysis.json` next to the originals (which are never modified).
- **Training** (GPU): `uv run --group ml python -m scripts.train_fsa --config
  configs/training/experiments/fsa_gemma4b_debug.yaml` trains a LoRA adapter. The target is either FSA
  (a mixture over k fresh fake secrets per example), B3 (`target: no_secret`), or the
  random-layer control (`layers: random:N`). Training uses one model copy; the frozen passes
  run with the adapter disabled. It uses only the train part of the baseline's secret split and
  the configured system templates / user prompts; the rest are held out. Output goes to
  `outputs/training/<run>/`: the config with the git hash, `train_log.jsonl`, checkpoints
  (continue with `--resume <run>`), and `adapter/`. Evaluate with `baseline_leakage.py --method
  fsa --adapter outputs/training/<run>/adapter`. That refuses an adapter whose training secrets
  overlap its held-out ones, and records the adapter in config.yaml. `likelihood_attack.py`
  then scores with base + adapter (`threat_model: grey_box`). `--scorer-no-adapter` scores with
  the base model only (`threat_model: public_weights`), using the same prompts. That result goes
  to `likelihood_attack_public_weights*.json`, a separate series. Every 25 steps the log
  records `kl_to_full_mixture`: KL to the mixture over all other training secrets, on fixed
  held-in examples. It is a low-noise progress measure, also shown in the step printout.
  `fsa.fake_set: all_others` (`fsa_gemma12b_fullmix.yaml`, k = 49) trains on that exact
  mixture. `--max-minutes` checkpoints and stops (exit code 3) before a wall-time limit. On
  The Slurm cluster: `sbatch --export=ALL,CONFIG=configs/training/experiments/<name>.yaml scripts/slurm/fsa_train.sbatch`; it
  stops after 330 minutes and continues with `RESUME=outputs/training/<run>`.
- **Tokenization check**: `uv run --group ml python -m scripts.check_tokenization [--all] [--csv path]`.
- **Live model tests** (opt-in): `$env:FSA_RUN_HF_TESTS = "1"; uv run --group ml pytest tests/test_hf_live.py`.

Prompt sets: [prompt_sets.md](prompt_sets.md). Likelihood attacker: [likelihood_attack.md](likelihood_attack.md).
