# Results summary and committed export

[← root README](../../../README.md) · commands run from the repository root.

```powershell
uv run python -m scripts.collect_results
uv run python -m scripts.collect_results --runs-dir mock_data/runs --out-dir results/mock
```

One row per (evaluation run, likelihood-attack file): the heuristic attacker with CIs, threat
model, calibrated and raw bits, tau, permutation p, exact/top-5, with duplicate files marked.
Writes `results/summary.csv` and `results/summary.md`. The runs folder defaults to
`runs_dir` from [`configs/training/app.yaml`](../configs/app.yaml). The script only reads it, refuses an `--out-dir` inside it
or inside `outputs/`, and needs no `ml` group (it runs on a login node).

**Committed results**: `uv run python -m scripts.export_results` copies the runs listed in
`configs/training/export_runs.yaml` from `outputs/` into `results/` (committed): only `config.yaml`,
`metrics.json`, `diagnostics.json` and `likelihood_attack*.json` per evaluation run, and
`config.yaml` + `utility_metrics.json` per utility run. Home-directory paths in configs are made
repo-relative (recorded in `results/export_manifest.json`). It then runs `collect_results` over
`results/runs` and regenerates `results/README.md`. `--dry-run` checks the list without writing;
`--prune` removes exported runs no longer listed.
