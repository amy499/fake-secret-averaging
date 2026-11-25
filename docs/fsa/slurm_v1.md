# Slurm jobs on the Slurm cluster

Run every command from the repository root (the job `cd`s to the submit directory
and writes its log to `slurm/logs/` relative to it). Put `HF_TOKEN` (and optionally `HF_HOME`)
in `.env` there; Gemma models need a token with access granted.

**Wall time.** Every job sets `#SBATCH --time=06:00:00`, which is the Slurm cluster's per-job maximum. A
higher value never starts: it stays pending forever with `QOSMaxWallDurationPerJobLimit`.
Longer runs must be split with `RESUME`. `train.sbatch` saves a checkpoint and exits with code
3 after `MAX_MINUTES` (default 330, i.e. 30 min of headroom); resubmit it with
`RESUME=outputs/training/<run>` and it continues from that checkpoint, repeating until done.

## Submit

```bash
sbatch scripts/slurm/fsa_baseline.sbatch                       # qwen2.5-7b-instruct, base, quick preset
sbatch --export=ALL,MODEL=gemma3-4b-it,METHOD=prompt_hiding,PRESET=full scripts/slurm/fsa_baseline.sbatch
```

`MODEL` is a base-model id from `configs/training/models.yaml`, `METHOD` one of `base`, `prompt_hiding`,
`prompt_hiding_strict`, `prompt_anti_suppression`, and `PRESET` one of `quick`, `standard`,
`full`. Results go to `outputs/runs/`.

Likelihood (Bayes) attacker on a finished run:

```bash
sbatch --export=ALL,RUN=outputs/runs/<run_id> scripts/slurm/fsa_likelihood.sbatch
sbatch --export=ALL,RUN=outputs/runs/<run_id>,SCORER=qwen2.5-7b-instruct scripts/slurm/fsa_likelihood.sbatch
```

`RUN` is required. `SCORER` defaults to the run's own model; a different one acts as a
surrogate. `EXTRA_ARGS` passes further options through (space-separated). For example, run a
quick check that the prompt cache gives the same scores on a new model, once with and once
without `--no-cache`:

```bash
sbatch --export=ALL,RUN=outputs/runs/<run_id>,EXTRA_ARGS="--limit-secrets 2 --no-cache" scripts/slurm/fsa_likelihood.sbatch
```

Note that sbatch splits `--export` on commas, so EXTRA_ARGS itself must not contain one. The
job writes `likelihood_attack.json` (or `_2`, ...) into the run folder.

Public-weights attacker: on a run made with an adapter (`--method fsa`/`b3`/`random_layer`),
score the replies with the base model only. That is what an attacker who has the public weights
but not the adapter could do. The result goes to its own numbered file,
`likelihood_attack_public_weights.json`, labelled `threat_model: public_weights`; base + adapter
is `grey_box`:

```bash
sbatch --export=ALL,RUN=outputs/runs/<run_id>,EXTRA_ARGS="--scorer-no-adapter --calibrate" scripts/slurm/fsa_likelihood.sbatch
```

Calibrated run (also scores the train secrets, about twice the GPU time):

```bash
sbatch --export=ALL,RUN=outputs/runs/<run_id>,EXTRA_ARGS="--calibrate" scripts/slurm/fsa_likelihood.sbatch
```

Re-analysing an existing result needs no GPU: run `uv run python -m scripts.likelihood_attack
outputs/runs/<run_id> --from-scores outputs/runs/<run_id>/likelihood_attack.json` locally.

LoRA training (FSA, B3, random-layer control) from a config in `configs/training/experiments/`:

```bash
sbatch --export=ALL,CONFIG=configs/training/experiments/fsa_gemma4b_debug.yaml scripts/slurm/fsa_train.sbatch
sbatch --export=ALL,RESUME=outputs/training/<run> scripts/slurm/fsa_train.sbatch     # continue from the last checkpoint
```

The job writes to `outputs/training/<run>/`, and the adapter ends up in its `adapter/` folder.
Evaluate the adapter on the held-out secrets with the baseline job's script:
`--method fsa --adapter outputs/training/<run>/adapter`. `baseline.sbatch` takes optional
`ADAPTER`, `SECRET_TYPE` and `PROMPT_ID`; when they are unset, nothing changes:

```bash
sbatch --export=ALL,MODEL=gemma3-12b-it,METHOD=fsa,PRESET=full,SECRET_TYPE=digits4,PROMPT_ID=lottery_4digit,ADAPTER=outputs/training/<run>/adapter scripts/slurm/fsa_baseline.sbatch
```

Utility benchmarks. Fill the dataset cache first, once, on the login node:
`uv run --group ml python -m scripts.utility_eval --prepare --prepare-model gemma3-12b-it`.

```bash
sbatch --export=ALL,MODEL=gemma3-12b-it scripts/slurm/fsa_utility.sbatch
sbatch --export=ALL,MODEL=gemma3-12b-it,SECRET_TYPE=digits4,ADAPTER=outputs/training/<run>/adapter scripts/slurm/fsa_utility.sbatch
```

The full digits4 / need-term plan is in [docs/pe4_need.md](../docs/pe4_need.md).

## Check

```bash
squeue -u $USER                                    # state (PD = pending, R = running) and job id
tail -f slurm/logs/fsa-baseline_<jobid>.out        # live progress; Ctrl+C stops tail, not the job
tail -f slurm/logs/fsa-likelihood_<jobid>.out
tail -f slurm/logs/fsa-train_<jobid>.out
```

## Cancel

```bash
scancel <jobid>
```
