# FSA v1 notes

FSA v1 is the first version of fake-secret averaging, on 2-digit and 4-digit codes in a single
system prompt, trained and evaluated with `scripts/train_fsa.py`, `scripts/baseline_leakage.py` and
`scripts/likelihood_attack.py`. FSA-lite (`src/fsa_lite/lite/`) grew out of it. These notes are the
record of that line:

| file | content |
|---|---|
| [history.md](history.md) | runs, jobs and results of FSA v1, in order |
| [fsa_status.md](fsa_status.md) | what the v1 code does and what was tested |
| [experiments.md](experiments.md) | training configs (`configs/training/experiments/`) |
| [pe4_need.md](pe4_need.md) | the need-preserving KL term, the digits4 codes and PasswordEval-style prompts |
| [likelihood_attack.md](likelihood_attack.md) | the likelihood (Bayes) attacker on finished runs |
| [prompt_sets.md](prompt_sets.md) | user-prompt sets of `configs/training/prompts.yaml` |
| [utility_benchmarks.md](utility_benchmarks.md) | `scripts/utility_eval.py` |
| [results_export.md](results_export.md) | `scripts/export_results.py` -> `results/fsa_v1/` |
| [slurm_v1.md](slurm_v1.md) | the v1 job scripts (`scripts/slurm/fsa_*.sbatch`) |

The v1 results are in [`results/fsa_v1/`](../../results/fsa_v1/README.md). The dashboard of the
original project is not part of this repository.
