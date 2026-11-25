# Prompt sets

[← root README](../../../README.md) · commands run from the repository root.

`scripts.baseline_leakage --prompt-set NAME` (instead of `--prompt-id`) spreads each secret's
replies evenly over a named set from `prompt_sets` in [`configs/training/prompts.yaml`](../configs/prompts.yaml); `--replies` must be
a multiple of the set size, every secret gets the same replies per prompt, and config.yaml records
the set under `eval.prompt_set`.

```powershell
uv run --group ml python -m scripts.baseline_leakage --backend real --method base --prompt-set unpredictabench_2digit --replies 8
```

- `unpredictabench_2digit`: four prompts taken verbatim from UnpredictaBench (Hugging Face
  dataset `UnpredictaBench/UnpredictaBench`, `train.jsonl`; licence CC-BY-4.0; the source line
  and uid of each prompt are noted in `prompts.yaml`). Cite the dataset when reporting these
  results.
- **Evaluation-only prompts**: `eval_only_prompts` in `prompts.yaml` (`lottery`,
  `verify_secret_code`) and every prompt of a set with `eval_only: true`
  (`unpredictabench_2digit`). `scripts.train_fsa` refuses at startup to train on any of them, and
  `general_prompts.train` / `.eval` must not overlap.
- **Multi-prompt training**: [`configs/training/experiments/fsa_gemma12b_fullmix_multiprompt.yaml`](../configs/training/experiments/fsa_gemma12b_fullmix_multiprompt.yaml) is
  [`fsa_gemma12b_fullmix.yaml`](../configs/training/experiments/fsa_gemma12b_fullmix.yaml) (k = 49, `all_others`) with 13 training user prompts (the two
  originals, eight Subliminal Learning prompts, three written by the team), held-out prompts =
  `lottery` + `unpredictabench_2digit`, and 2 offline replies per (secret, prompt). It tests
  whether FSA generalises to unseen prompts.
