# Cluster setup: TC2 (Slurm, A40) and NSCC (PBS, A100-40GB)

How to run this repository on the two clusters the results came from, reusing what is already
there (model cache, benchmark data, the evaluation venv, the trained adapters, the private
held-out files and the existing original runs), and the job sequences for:

- **item 1**: the post-hoc Gemma ablation without the USE term (`fsa_lite_run3_nouse`, TC2);
- **item 2**: the output-filter baselines on PasswordEval (`output_filter_naive`, and
  `output_filter_rule`, an oracle upper bound): Qwen on NSCC, Gemma on TC2.

Placeholders: `<...>` is a value for your account. `<OLD_HARNESS>` is the existing evaluation-
harness checkout on that cluster (the one whose `outputs/protocol/` holds the run3 results);
`<OLD_FSA>` is the existing FSA training checkout (the folder whose `outputs/training/` holds the
run3 and nouse runs). Every command block names the machine it is typed on.

## Rules (both clusters)

- **No Python on a login node.** Everything that runs Python is a job (`sbatch` / `qsub` through
  `scripts/submit_slurm.sh` / `scripts/submit_pbs.sh`); on the login node only git, cp, ln, grep and
  the scheduler commands.
- **TC2 (Slurm):** partition `<gpu-partition>` (the A40 partition), QoS `normal`, 1 GPU per job,
  `--time` at most `05:50:00`, at most **2 submitted jobs** (running or pending) at a time: submit
  the next one when `squeue -u $USER` shows a free slot.
- **NSCC (PBS):** queue `normal`, 1 GPU per job. `normal` routes by walltime: `-l walltime` up to
  `01:00:00` lands in `gdev`, longer in `g1`. Never request a walltime or resources that route to
  `g2` (check the routing limits with `qstat -Qf normal`).
- **No commas inside one value** of `sbatch --export` or `qsub -v` (both split on commas). A value
  with spaces (`RUN_ARGS`, `CHECK_ARGS`, `EXTRA_ARGS`) goes in the environment: Slurm jobs read it
  directly (sbatch exports the environment); for PBS put only its **name** in `-v`
  (`-v RUN_ARGS`), so qsub takes the value from the environment. `--benchmarks a+b` uses `+`.
- Submit from a clean login shell (no virtualenv active).

## 1. Clone and configure

```bash
# TC2 (login node)
git clone https://github.com/amy499/fake-secret-averaging.git "$HOME/fsa-lite"
cd "$HOME/fsa-lite" && cp cluster.env.example cluster.env && cp paths.local.yaml.example paths.local.yaml
```

```bash
# NSCC (login node): keep the checkout on scratch (eval/.venv and the data cache live inside it)
git clone https://github.com/amy499/fake-secret-averaging.git "<SCRATCH>/fsa-lite"
cd "<SCRATCH>/fsa-lite" && cp cluster.env.example cluster.env && cp paths.local.yaml.example paths.local.yaml
```

`cluster.env` (git-ignored), per cluster:

| key | TC2 | NSCC |
|---|---|---|
| `CLUSTER_NAME` | the name recorded by the existing original runs (below) | the same, from that cluster's runs |
| `CLUSTER_USER` | `<username>` | `<username>` |
| `REPO_ROOT` | `$HOME/fsa-lite` | `<SCRATCH>/fsa-lite` |
| `SCRATCH` | `<scratch-dir>` | `<scratch-dir>` |
| `HF_HUB_CACHE` | the existing model cache (section 2) | the existing model cache (section 2) |
| `FSA_LITE_DATA` | the existing `fsa_lite_data` folder (built run3 data) | the same, if training data is needed there |
| `HARNESS_DATA` | the existing harness data cache (section 2) | the existing harness data cache (section 2) |
| `SLURM_PARTITION` / `SLURM_QOS` | `<gpu-partition>` / `normal` | (unused) |
| `PBS_PROJECT` / `PBS_QUEUE` | (unused) | `<project>` / `normal` |

**`CLUSTER_NAME` must equal the cluster name in the existing original runs**: the team table pairs
a technique run with a borrowed original (`original_from: fsa`) only on the same cluster. Read it
from one existing summary (shell only, prints the name):

```bash
# TC2 (login node); on NSCC the same with qwen
grep -o '"cluster": *"[^"]*"' <OLD_HARNESS>/outputs/protocol/fsa/gemma/original/password_eval_team/summary.json | head -1
```

`paths.local.yaml` (git-ignored): absolute paths of the adapters, which stay outside the repository.
Item 1 needs only the nouse adapter; item 2 needs none.

```yaml
# TC2: paths.local.yaml
adapters:
  gemma_fsa_lite_run3: <OLD_FSA>/outputs/training/fsa_lite_gemma12b_run3/adapter
  gemma_fsa_lite_run3_nouse: <OLD_FSA>/outputs/training/fsa_lite_gemma12b_run3_nouse/adapter
```

The bridge checks each adapter's sha256 against the approach spec before the model loads
(`configs/eval/fsa_lite_run3_nouse.approach.yaml` pins step 500, prefix `63a6d346`) and refuses
any other file.

## 2. Reuse what is already on the cluster

| what | how | check |
|---|---|---|
| Model cache (`models--google--gemma-3-12b-it`, `models--Qwen--Qwen2.5-7B-Instruct`, pinned snapshots) | `HF_HUB_CACHE` in `cluster.env` = the existing hub folder. Jobs load offline (`HF_HUB_OFFLINE=1`); nothing is downloaded. | Each job log prints `gemma-3-12b-it cached: yes`. |
| Benchmark data (pinned files) | `HARNESS_DATA` = the existing cache, e.g. `<OLD_HARNESS>/.cache/data` (same layout). | One FETCH job: `fetch` downloads only missing files, then `verify` prints MATCH / MISMATCH / MISSING per file against the pinned hashes. Every run also checks its files. |
| Evaluation venv | Link the existing one: `ln -s <OLD_HARNESS>/.venv eval/.venv`. The harness is run from the checkout, not installed, so the venv is reusable. | Every job runs `python -m core.cluster check` against `eval/requirements-gpu.txt`; on a mismatch it stops. Then `rm eval/.venv` (the link only) and run the SETUP job. |
| Trained adapters | Absolute paths in `paths.local.yaml` (above); never copied into the repository. | sha256 check by the bridge; the adapter-check job (item 1). |
| Private held-out files (git-ignored) | **Copy** on the same cluster into `configs/eval/private/`. Never move them, never copy between clusters, never commit. | `cmp` against the source (prints nothing when equal); the protocol checks the folder is git-ignored. |
| Existing original runs (`fsa`, PasswordEval) | Copy the run folder into this checkout's `eval/outputs/protocol/` (git-ignored), so the team table pairs new technique runs with it. | Team-table status `ok`. If it says `not paired: environment differs` (library or GPU), run the original again (item 2, optional step). |
| FSA training venv (dev split only) | `dev_wrong_kind` runs `uv run --group ml` in this checkout: uv builds `.venv` from its package cache (`~/.cache/uv`), so it is quick when the old venv was built on the same cluster. | The job log shows the `uv` output. |

```bash
# TC2 (login node), from $HOME/fsa-lite: private files for item 1 (adapter check + held-out guard)
mkdir -p configs/eval/private && chmod 700 configs/eval/private
cp -p <OLD_HARNESS>/experiments/_bridges/private/held_out_secrets.txt configs/eval/private/
cp -p <OLD_HARNESS>/experiments/_bridges/private/fsa_lite_run3_used_secrets.txt configs/eval/private/
cmp <OLD_HARNESS>/experiments/_bridges/private/held_out_secrets.txt configs/eval/private/held_out_secrets.txt
cmp <OLD_HARNESS>/experiments/_bridges/private/fsa_lite_run3_used_secrets.txt configs/eval/private/fsa_lite_run3_used_secrets.txt
git status --short configs/eval/private   # prints nothing: ignored
# existing original runs to pair with (Gemma PasswordEval)
mkdir -p eval/outputs/protocol/fsa/gemma/original
cp -a <OLD_HARNESS>/outputs/protocol/fsa/gemma/original/password_eval_team eval/outputs/protocol/fsa/gemma/original/
# optional: run3's technique row next to the ablation in one table
mkdir -p eval/outputs/protocol/fsa_lite_run3/gemma/technique
cp -a <OLD_HARNESS>/outputs/protocol/fsa_lite_run3/gemma/technique/password_eval_team eval/outputs/protocol/fsa_lite_run3/gemma/technique/
ln -s <OLD_HARNESS>/.venv eval/.venv
```

```bash
# NSCC (login node), from <SCRATCH>/fsa-lite: item 2 needs no private file and no adapter
mkdir -p eval/outputs/protocol/fsa/qwen/original
cp -a <OLD_HARNESS>/outputs/protocol/fsa/qwen/original/password_eval_team eval/outputs/protocol/fsa/qwen/original/
ln -s <OLD_HARNESS>/.venv eval/.venv
```

Then one data check per cluster (it also proves the venv works):

```bash
# TC2 (login node)
FETCH=1 SBATCH_EXTRA="--time=00:30:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch --fetch
# only if the log says the pinned stack is missing or differs: rm eval/.venv, then
SETUP=1 SBATCH_EXTRA="--time=01:00:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch --setup
```

```bash
# NSCC (login node)
QSUB_EXTRA="-l walltime=00:30:00 -v FETCH=1" bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
# only if needed (rm eval/.venv first):
QSUB_EXTRA="-l walltime=01:00:00 -v SETUP=1" bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
```

Logs: `scripts/slurm/logs/` and `scripts/pbs/logs/`. They hold counts, hashes and metrics, never
secrets or replies. The replies are in `eval/outputs/.../items.jsonl` (git-ignored; never commit them).

## 3. Item 1: Gemma no-use ablation (TC2)

Entry points (all through `scripts/slurm/harness_run.sbatch` unless noted):

| step | entry point | output |
|---|---|---|
| adapter check | `scripts/slurm/harness_adapter_check.sbatch` -> `python -m experiments._bridges.fsa check --approach fsa_lite_run3_nouse` | log: LoRA tensors attached / missing, sha256 match, adapter on vs off identical-reply counts |
| PasswordEval run | `python run.py --approach fsa_lite_run3_nouse --model gemma --suite core --variant technique --benchmarks password_eval_team` | `eval/outputs/protocol/fsa_lite_run3_nouse/gemma/technique/password_eval_team/` |
| re-score | `python -m stats.rescore outputs/protocol` (independent scorer; writes `rescore.json` per run) | `rescore.json` |
| team table | `python -m stats.team_table build outputs/protocol --approach fsa_lite_run3_nouse --approach fsa_lite_run3 --model gemma --suite core --out ...` | `team_table.{md,csv,json}` |
| release log-prob | `python -m analysis.release_logprob.run --config analysis/release_logprob/config.yaml --model gemma --approach fsa_lite_run3_nouse --device cuda` | `eval/outputs/diagnostics/release_logprob/gemma/fsa_lite_run3_nouse/` |
| dev split | `scripts/slurm/dev_wrong_kind.sbatch` -> `src/fsa_lite/lite/dev_wrong_kind.py --run run3=... --run nouse=...` | `outputs/diagnostics/dev_wrong_kind/<OUT_NAME>.json` |

```bash
# TC2 (login node), from $HOME/fsa-lite; at most 2 jobs in the queue at a time
# 1. adapter check (needs configs/eval/private/ from section 2)
CHECK_ARGS="--approach fsa_lite_run3_nouse" bash scripts/submit_slurm.sh scripts/slurm/harness_adapter_check.sbatch

# 2. PasswordEval, technique variant (the original is the copied fsa run)
SBATCH_EXTRA="--time=02:00:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch \
  --approach fsa_lite_run3_nouse --model gemma --suite core --variant technique --benchmarks password_eval_team

# 3. release log-prob (independent of 2; may run beside it)
MODULE=analysis.release_logprob.run EXTRA_ARGS="--model gemma --approach fsa_lite_run3_nouse" SBATCH_EXTRA="--time=01:00:00" \
  bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch analysis/release_logprob/config.yaml

# 4. dev split, run3 vs nouse at every common checkpoint (CPU work; the job keeps the GPU request)
env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV \
  CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml \
  RUNS=run3=<OLD_FSA>/outputs/training/fsa_lite_gemma12b_run3+nouse=<OLD_FSA>/outputs/training/fsa_lite_gemma12b_run3_nouse \
  OUT_NAME=gemma_run3_nouse bash scripts/submit_slurm.sh scripts/slurm/dev_wrong_kind.sbatch

# 5. after 2 has finished: re-score every run under outputs/protocol (the copied ones too)
MODULE=stats.rescore SBATCH_EXTRA="--time=00:15:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch outputs/protocol

# 6. team table (the nouse row next to run3's, both against the fsa original)
MODULE=stats.team_table SBATCH_EXTRA="--time=00:15:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch \
  build outputs/protocol --approach fsa_lite_run3_nouse --approach fsa_lite_run3 --model gemma --suite core \
  --out outputs/team/fsa_lite_run3_nouse_gemma
```

The release log-prob comparison needs run3's own file too
(`eval/outputs/diagnostics/release_logprob/gemma/release_logprob.json`, from the same job without
`--approach`, or copied from `<OLD_HARNESS>/outputs/diagnostics/release_logprob/gemma/`). The ablation
is post-hoc: report it next to run3, never as a technique row of its own.

## 4. Item 2: output-filter baselines on PasswordEval

No adapter, so no adapter check. Each spec pairs with the plain original model on the same items
(`original_from: fsa`). The rule variant's rows are labelled `UPPER BOUND, not a defence claim`
in `team_table.md` and in `summary.json` (`defence_info.upper_bound`).

```bash
# NSCC (login node), from <SCRATCH>/fsa-lite: Qwen
RUN_ARGS="--approach output_filter_naive --model qwen --suite core --variant technique --benchmarks password_eval_team" \
  QSUB_EXTRA="-l walltime=01:00:00 -v RUN_ARGS" bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
RUN_ARGS="--approach output_filter_rule --model qwen --suite core --variant technique --benchmarks password_eval_team" \
  QSUB_EXTRA="-l walltime=01:00:00 -v RUN_ARGS" bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
# after both: re-score, then the table
MODULE=stats.rescore RUN_ARGS="outputs/protocol" QSUB_EXTRA="-l walltime=00:15:00 -v MODULE,RUN_ARGS" \
  bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
MODULE=stats.team_table \
  RUN_ARGS="build outputs/protocol --approach output_filter_naive --approach output_filter_rule --model qwen --suite core --out outputs/team/output_filter_qwen" \
  QSUB_EXTRA="-l walltime=00:15:00 -v MODULE,RUN_ARGS" bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
# optional, only if the table says the copied original is "not paired: environment differs":
RUN_ARGS="--approach fsa --model qwen --suite core --variant original --benchmarks password_eval_team --overwrite" \
  QSUB_EXTRA="-l walltime=01:00:00 -v RUN_ARGS" bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
```

```bash
# TC2 (login node), from $HOME/fsa-lite: Gemma (the fsa Gemma original copied in section 2)
SBATCH_EXTRA="--time=02:00:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch \
  --approach output_filter_naive --model gemma --suite core --variant technique --benchmarks password_eval_team
SBATCH_EXTRA="--time=02:00:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch \
  --approach output_filter_rule --model gemma --suite core --variant technique --benchmarks password_eval_team
# after both: re-score, then the table
MODULE=stats.rescore SBATCH_EXTRA="--time=00:15:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch outputs/protocol
MODULE=stats.team_table SBATCH_EXTRA="--time=00:15:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch \
  build outputs/protocol --approach output_filter_naive --approach output_filter_rule --approach fsa_lite_run3 \
  --model gemma --suite core --out outputs/team/output_filter_gemma
```

On TC2, item 1 step 5 and item 2's re-score are the same command: one re-score after all runs is enough.

The table has one PasswordEval row per metric (`leak_pw_strict`, `leak_info_strict`, `use_strict`,
`joint_strict`), as for fsa_lite_run3; the other core benchmarks are listed as `MISSING` (not run),
never dropped. Commit only `team_table.md` and `team_table.csv`, e.g. into
`results/fsa_lite_run3_nouse/gemma_password_eval/` and `results/output_filter/{gemma,qwen}_password_eval/`;
`team_table.json` (run paths) and everything under `eval/outputs/` stay on the cluster.

## 5. GPU time (estimates)

Not measured in this repository. The best estimate for a PasswordEval run is the existing
original's own wall time: the same 1,500 items, the same model and batch size, so a technique
run without an adapter (output filter) costs the same and one with an adapter slightly more:

```bash
# TC2 / NSCC (login node): seconds of the existing original PasswordEval run
grep -o '"wall_s": *[0-9.]*' eval/outputs/protocol/fsa/*/original/password_eval_team/summary.json
```

Otherwise, by construction: 1,500 items / batch 16 = 94 batches of at most 256 new tokens, so at
most about 24,000 decode steps per run, plus 3-6 minutes of model loading.

| job | cluster | request | estimate |
|---|---|---|---|
| FETCH (verify), SETUP | both | 00:30:00, 01:00:00 | 5-10 min, 20-40 min (setup only if needed) |
| adapter check (nouse) | TC2, 1 A40 | 01:00:00 (job default) | 10-20 min |
| PasswordEval, Gemma-3-12B-it (nouse, naive, rule; each) | TC2, 1 A40 | 02:00:00 | 40-80 min |
| PasswordEval, Qwen2.5-7B-Instruct (naive, rule; each) | NSCC, 1 A100-40GB | 01:00:00 (gdev) | 20-40 min; if the first one's `wall_s` exceeds 50 min, give the second 02:00:00 (g1) |
| release log-prob, Gemma (600 teacher-forced items, base and adapter, no generation) | TC2, 1 A40 | 01:00:00 | 15-30 min |
| dev split (`dev_wrong_kind`, CPU) | TC2 | 00:30:00 (job default) | under 10 min (longer only if uv builds `.venv`) |
| re-score, team table (CPU) | both | 00:15:00 each | a few minutes |

Totals: item 1 about 2-3 GPU-hours on TC2; item 2 about 1.5-3 GPU-hours on TC2 (Gemma) and about
1 GPU-hour on NSCC (Qwen). Read the real `wall_s` from the first run's `summary.json` and adjust
the later requests.
