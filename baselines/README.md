# Baselines

Generic defences to compare FSA-lite with. The evaluation harness (`eval/`) discovers every module
here, so each is a `defence:` name in an attack-study config (`configs/attack/`) or an approach spec
(`configs/eval/`).

| name | what it does | where | status |
|---|---|---|---|
| `none` | the model unchanged (the "original" variant of every protocol run) | `basic.py` | ✅ |
| `prompt_instruction` | appends a confidentiality instruction to the system message (adds one if none) | `basic.py` | ✅ |
| `output_filter` | redacts the prompt's secret from every reply: exact, case-insensitive and separator-tolerant matches ("1 2 3 4"); no encodings. Mode `naive` (default): the secret only. Mode `rule`: also the item's protected strings (PasswordEval: the confidential info) unless a user turn contains the secret; it is given the true info and PasswordEval's rule, so it is an **oracle upper bound, not a defence**. Specs: `configs/eval/output_filter_{naive,rule}.approach.yaml` (PasswordEval, both models) | `output_filter.py` | ✅ unit + dry run; 🟡 not run on a real model |
| B3 (match the no-secret model) | a LoRA trained so that, with the secret in context, the model matches the frozen model *without* the secret in the prompt (target `no_secret`, same loop as FSA v1) | trained by `scripts/train_fsa.py` with `configs/training/experiments/b3_gemma12b.yaml`; evaluated as an adapter through the `fsa` bridge | 🟡 one Gemma-3-12B-it run on the FSA v1 2-digit task (`results/fsa_v1/`) |

`basic.py` also holds `DelegatingWrapper`, the base class for a defence that wraps a model
(see `eval/experiments/_template/` for writing your own).

## B3

The name follows Abhipray Chavan's design in the AI6130 Group 11 project. Train it like an FSA v1
adapter:

```bash
# Git Bash, from the repository root, on a GPU node (or a Slurm job: scripts/slurm/fsa_train.sbatch)
uv run --group ml python -m scripts.train_fsa --config configs/training/experiments/b3_gemma12b.yaml
```

The run folder's `adapter/` can then be evaluated with the v1 scripts (`scripts/baseline_leakage.py
--method b3 --adapter <run>/adapter`, `scripts/likelihood_attack.py`) or, through
`paths.local.yaml`, with the harness like any FSA adapter. The only B3 result in this repository is
the v1 2-digit run in `results/fsa_v1/README.md`; it has not been run on the protocol benchmarks.
