# FSA-lite: fake-secret averaging against in-context secret leakage

FSA-lite is a LoRA adapter that keeps a model from repeating a secret that sits in its context
(a password, an access code, a private fact) when the turn does not authorise it, while it still
uses the secret when the turn does. This repository holds the training code, the evaluation
harness it was measured with, generic baselines and the committed results.

## Method

The adapter is trained on the frozen model's own replies, with three terms chosen by the turn
type (`src/fsa_lite/lite/v2_objective.py`, `lite_source.py`):

- **Hidden turns** (attacks, wrong or missing password, unauthorised requests): reverse KL from the
  adapted model, holding the real secret, to the frozen model's **average over K = 4 fake secrets**,
  KL(p_student(. | real) || (1/K) sum_i p_frozen(. | fake_i)), per reply token. The fakes are drawn
  fresh at every step, so the reply carries no information that distinguishes the real secret.
- **Authorised and benign turns**: a **need-KL anchor** to the frozen model under the same real
  context, KL(p_student || p_frozen), so ordinary use stays unchanged (weight 2.0 in run3).
- **Correct-password turns** (run3): the need-KL anchor plus a **use term**, a weighted NLL of the
  frozen model's own reply that releases the guarded payload (payload tokens x5).

Training data (`src/fsa_lite/lite/build_lite_run3.py`): Tensor Trust defences and attacks (train
split), CIMemories memory contexts, the repository's own benign tasks and, in run3, its own
key-guarded-payload contexts (a password guards a payload; attack, wrong-password and
correct-password turns per context). Nothing is taken from the evaluation benchmarks, and a
contamination check against the evaluation files runs before training. LoRA r = 16, alpha = 32,
dropout 0.05 on q/k/v/o of every layer; 500 steps; the checkpoint is chosen on a dev split by a rule
fixed before any benchmark run (best dev joint subject to dev use >= 0.95 x step 0;
`configs/training/experiments/fsa_lite_*_run3.yaml`).

**Selected checkpoints** (from the training logs; `selection.json` is not included in this
repository), both chosen by the pre-registered dev rule; the full sha256 of each adapter is pinned
in [`configs/eval/fsa_lite_run3.approach.yaml`](configs/eval/fsa_lite_run3.approach.yaml), and the
harness refuses an adapter whose `adapter_model.safetensors` does not match it:

| model | run3 step | adapter sha256 (prefix) |
|---|---|---|
| Qwen2.5-7B-Instruct | 250 | `d5f8ca21` |
| Gemma-3-12B-it | 375 | `324d8d8d` |

**Software stacks.** Training uses the pinned `ml` dependency group of `pyproject.toml`
(`torch==2.11.0`, `transformers==5.20.1`, `peft==0.21.2`, `accelerate==1.15.0`,
`safetensors==0.8.0`, `huggingface-hub==2.2.0`, `tokenizers==0.23.3`, `pyarrow==26.0.1`,
`absl-py==2.5.1`, `immutabledict==4.3.1`, `langdetect==1.0.9`, `nltk==3.10.4`). Evaluation uses
`eval/requirements-gpu.txt` (`torch==2.6.0`, `transformers==4.51.3`, `peft==0.15.2`, with the other
pins in that file). Both in bf16.

## Results (run3, core benchmarks)

FSA-lite run3 against the original model, on the same items, order, decoding and batch size; one
greedy reply per item; 95% bootstrap CIs over items (ConfAIde: over meetings); paired differences
with the exact McNemar test. All rows re-scored independently (`status ok`). How each number is
computed: [docs/eval_protocol.md](docs/eval_protocol.md). Leak and error metrics: lower is better;
use, joint, accuracy and IFEval: higher is better.

**Gemma-3-12B-it** ([`results/fsa_lite_run3/gemma_core/team_table.md`](results/fsa_lite_run3/gemma_core/team_table.md))

| benchmark | metric | original | FSA-lite run3 | FSA-lite - original (paired 95% CI, McNemar p) | status |
|---|---|---|---|---|---|
| ConfAIde Tier 4 | leak | 0.375 [0.225, 0.525] n=40 | 0.400 [0.225, 0.600] n=40 | +0.025 [-0.125, +0.175] McNemar p=1 | ok |
| ConfAIde Tier 4 | no_public | 0.700 [0.600, 0.800] n=40 | 0.700 [0.600, 0.800] n=40 | +0.000 [-0.100, +0.100] McNemar p=1 | ok |
| ConfAIde Tier 4 | error | 0.800 [0.700, 0.900] n=40 | 0.800 [0.700, 0.900] n=40 | +0.000 [-0.100, +0.100] McNemar p=1 | ok |
| IFEval | prompt_strict | 0.623 [0.582, 0.664] n=541 | 0.628 [0.586, 0.669] n=541 | +0.006 [-0.024, +0.035] McNemar p=0.801 | ok |
| IFEval | prompt_loose | 0.677 [0.638, 0.715] n=541 | 0.677 [0.634, 0.717] n=541 | +0.000 [-0.028, +0.028] McNemar p=1 | ok |
| MMLU | accuracy | 0.737 [0.723, 0.751] n=3420 | 0.735 [0.721, 0.749] n=3420 | -0.002 [-0.008, +0.004] McNemar p=0.501 | ok |
| PasswordEval | leak_pw_strict | 0.052 [0.039, 0.066] n=1000 | 0.007 [0.002, 0.012] n=1000 | -0.045 [-0.060, -0.031] McNemar p=2.14e-10 | ok |
| PasswordEval | leak_info_strict | 0.078 [0.062, 0.095] n=1000 | 0.187 [0.163, 0.212] n=1000 | +0.109 [+0.085, +0.133] McNemar p=3.19e-19 | ok |
| PasswordEval | use_strict | 0.716 [0.676, 0.754] n=500 | 0.838 [0.806, 0.870] n=500 | +0.122 [+0.082, +0.162] McNemar p=9.93e-09 | ok |
| PasswordEval | joint_strict | 0.614 [0.572, 0.656] n=500 | 0.574 [0.532, 0.620] n=500 | -0.040 [-0.090, +0.010] McNemar p=0.13 | ok |

**Qwen2.5-7B-Instruct** ([`results/fsa_lite_run3/qwen_core/team_table.md`](results/fsa_lite_run3/qwen_core/team_table.md))

| benchmark | metric | original | FSA-lite run3 | FSA-lite - original (paired 95% CI, McNemar p) | status |
|---|---|---|---|---|---|
| ConfAIde Tier 4 | leak | 0.525 [0.375, 0.675] n=40 | 0.475 [0.300, 0.650] n=40 | -0.050 [-0.200, +0.100] McNemar p=0.754 | ok |
| ConfAIde Tier 4 | no_public | 0.550 [0.500, 0.625] n=40 | 0.575 [0.475, 0.675] n=40 | +0.025 [-0.075, +0.125] McNemar p=1 | ok |
| ConfAIde Tier 4 | error | 0.850 [0.750, 0.950] n=40 | 0.825 [0.725, 0.925] n=40 | -0.025 [-0.175, +0.125] McNemar p=1 | ok |
| IFEval | prompt_strict | 0.689 [0.649, 0.728] n=541 | 0.684 [0.641, 0.723] n=541 | -0.006 [-0.033, +0.020] McNemar p=0.798 | ok |
| IFEval | prompt_loose | 0.710 [0.671, 0.749] n=541 | 0.702 [0.662, 0.741] n=541 | -0.007 [-0.033, +0.018] McNemar p=0.689 | ok |
| MMLU | accuracy | 0.724 [0.708, 0.738] n=3420 | 0.723 [0.707, 0.738] n=3420 | -0.001 [-0.005, +0.004] McNemar p=0.896 | ok |
| PasswordEval | leak_pw_strict | 0.901 [0.882, 0.918] n=1000 | 0.002 [0.000, 0.005] n=1000 | -0.899 [-0.917, -0.880] McNemar p=4.73e-271 | ok |
| PasswordEval | leak_info_strict | 0.353 [0.323, 0.383] n=1000 | 0.132 [0.112, 0.153] n=1000 | -0.221 [-0.248, -0.193] McNemar p=6.97e-50 | ok |
| PasswordEval | use_strict | 0.892 [0.864, 0.918] n=500 | 0.850 [0.818, 0.882] n=500 | -0.042 [-0.068, -0.018] McNemar p=0.00107 | ok |
| PasswordEval | joint_strict | 0.420 [0.378, 0.464] n=500 | 0.660 [0.618, 0.700] n=500 | +0.240 [+0.194, +0.286] McNemar p=2.3e-22 | ok |

The same folders hold report-only robust-leak rows (normalised, encoded and partial matching of the
secret). Run2 results are in [`results/fsa_lite/`](results/fsa_lite/).

[`results/fsa_v1/`](results/fsa_v1/README.md) holds the results of the earlier FSA version (before
FSA-lite), kept for history.

## Baselines

| baseline | what it does | status |
|---|---|---|
| `none` | the model unchanged (the original variant of every run above) | implemented and tested |
| `prompt_instruction` | adds a confidentiality instruction to the system message | implemented and tested |
| `output_filter` | redacts the secret from every reply (exact, case-insensitive, separator-tolerant) | implemented, tested with a dummy model only; not yet evaluated on real models, no results claimed |
| B3 (match the no-secret model) | a LoRA trained to match the frozen model without the secret in the prompt | trained once on the FSA v1 2-digit task (`results/fsa_v1/`); not evaluated on the protocol benchmarks |

Details: [baselines/README.md](baselines/README.md).

### Known Gemma regression (diagnosis open)

On Gemma-3-12B-it, run3 lowers password leaks (0.052 -> 0.007) and raises correct use (0.716 ->
0.838), but **raises confidential-information leaks on PasswordEval from 0.078 to 0.187** (+0.109
[+0.085, +0.133]); the joint score does not improve (-0.040 [-0.090, +0.010]). Qwen2.5-7B-Instruct
does not show this. The cause is not established. Report-only diagnostics of the stored replies
(`eval/stats/leak_diagnosis.py`, `eval/stats/key_similarity.py`, `eval/analysis/release_logprob/`)
and a post-hoc ablation without the use term (`configs/training/experiments/fsa_lite_gemma12b_run3_nouse.yaml`,
`configs/eval/fsa_lite_run3_nouse.approach.yaml`, adapter not yet pinned) are set up; no result
from them is in this repository.

## Limitations

- **One seed** per model and run; no variance across training runs.
- **Verbatim-leak benchmarks only**: the reported leaks are string matches of the secret in the
  reply. Paraphrased or encoded disclosure is only covered by the report-only robust-leak rows.
- **Inadvertent leakage is not yet measured**: the attack study that asks whether the code can be
  recovered from benign replies (`configs/attack/hidden_code__*fsa_lite_run3*.yaml`, likelihood
  attacker) is configured and dry-run tested, not run.
- **Run3 was designed after run2's results** (its config header records why), so it is not an
  independent replication; the checkpoint rule was fixed before run3's benchmark runs.
- **Run3 trains on password-guarded contexts** (its own templates and secrets, contamination-checked
  against the evaluation files), a structure close to PasswordEval's; gains there may not carry over
  to other kinds of secret.
- Only the core suite (PasswordEval, ConfAIde Tier 4, MMLU, IFEval) was run for run3; the second
  suite (Tensor Trust, GSM8K) was not.

## Repository layout

| path | what |
|---|---|
| `src/fsa_lite/fsa/` | the FSA library: losses, mixture targets, prompting, secrets, the v1 trainer |
| `src/fsa_lite/lite/` | FSA-lite: data builders, training (`v2_train.py`), diagnostics, CPU self-tests |
| `src/fsa_lite/fsa_runtime/` | settings, Hugging Face / mock generators, text attackers, model registry |
| `eval/` | the evaluation harness (run from inside it): benchmarks, scorers, attackers, stats, analysis |
| `baselines/` | `none`, `prompt_instruction`, `output_filter`, and B3 ([baselines/README.md](baselines/README.md)) |
| `configs/training/` | training configs (`experiments/`), prompts, models |
| `configs/eval/` | approach specs for the harness (adapter keys and sha256 pins) |
| `configs/attack/` | attack-study configs (secret in context, sampled replies, attackers) |
| `scripts/` | command-line tools; `slurm/`, `pbs/` job scripts; `submit_slurm.sh`, `submit_pbs.sh` |
| `results/` | committed result tables (md / csv) and analysis plots |
| `docs/` | [evaluation protocol](docs/eval_protocol.md), [harness guide](docs/eval_harness.md), [FSA-lite pipeline record](docs/fsa_lite_pipeline.md), [FSA v1 notes](docs/fsa/README.md) |
| `tests/`, `eval/tests/` | training-side and harness test suites |

## How to reproduce

No adapters or datasets are included. Everything below downloads at pinned revisions.

1. **Environment.** Python 3.11. Training: `uv sync --group ml` (pinned stack in `pyproject.toml`).
   Harness: `cd eval && pip install -r requirements.txt` (laptop) or the SETUP job (cluster, pinned
   `eval/requirements-gpu.txt`). On a cluster, copy `cluster.env.example` to `cluster.env` and
   `paths.local.yaml.example` to `paths.local.yaml` and fill them in (both git-ignored).
2. **Tests** (CPU, no downloads):

   ```powershell
   # PowerShell, from the repository root
   python -m pytest -q                                   # training side (tests/)
   python src/fsa_lite/lite/selftest_v2.py               # FSA-lite pipeline self-tests
   cd eval; python -m pytest -q                          # harness
   ```
3. **Data.** `python scripts/gen_eval_secrets.py` (the evaluation secret lists, from fixed seeds);
   `bash scripts/fetch_lite.sh` (Tensor Trust and CIMemories raw data, on a machine with network);
   `cd eval && python -m core.data fetch && python -m core.data verify` (benchmark files, hashes checked).
4. **Train** (one GPU; resumable):

   ```bash
   # Git Bash / Linux shell, from the repository root
   CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml bash scripts/submit_slurm.sh scripts/slurm/lite_pilot.sbatch
   RUN_DIR=outputs/training/fsa_lite_gemma12b_run3 CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml \
     bash scripts/submit_slurm.sh scripts/slurm/lite_train.sbatch
   ```
   The PBS equivalents are `scripts/pbs/lite_{setup,pilot,train}.pbs` with `scripts/submit_pbs.sh`.
   The trainer writes `<run>/selection.json` and the selected `<run>/adapter`; put its path in
   `paths.local.yaml` and its sha256 in `configs/eval/fsa_lite_run3.approach.yaml`.
5. **Evaluate** (from `eval/`, one GPU per variant):

   ```bash
   python run.py --approach fsa_lite_run3 --model gemma --suite core --variant original --device cuda
   python run.py --approach fsa_lite_run3 --model gemma --suite core --variant technique --device cuda
   python -m stats.rescore outputs/protocol
   python -m stats.team_table build outputs/protocol --approach fsa_lite_run3 --model gemma --suite core --out ../results/fsa_lite_run3/gemma_core
   ```
   The adapter's sha256 is checked before the model loads. See [docs/eval_harness.md](docs/eval_harness.md).

## Compute used

As recorded in the training configs: Gemma-3-12B-it trained and evaluated on one NVIDIA A40 (Slurm
cluster; the run3 config estimates about 2 h of training including reply generation and dev
evaluation, within one 5:50 job); Qwen2.5-7B-Instruct on one NVIDIA A100-SXM4-40GB (PBS cluster;
the run3 config estimates about 70-80 GPU-minutes, run as resumable 1-hour jobs). Both in bf16,
batch size 16. Measured run times are in the job logs, which are not part of this repository.

## Origin

Started as my part of the AI6130 (NTU) Group 11 project with Abhipray Chavan, Prithvi Nishal and
Simerjit Kaur, which compared six defences; this repo continues the FSA-lite line only. The B3
baseline name follows Abhipray Chavan's design; evaluation definitions follow the team protocol.

Parts of this code were written with the help of Claude Code (Anthropic) and reviewed by the author.

## Licence

Code: MIT ([LICENSE](LICENSE)). Third-party code, data and model licences, and what is not included:
[NOTICE](NOTICE).
