# Evaluation harness (`eval/`)

The harness runs the public benchmarks and the synthetic leakage tasks for one approach (a defence),
for the original model and the model with the technique, on identical settings, and builds the
paired results table. What it measures and how is in [eval_protocol.md](eval_protocol.md).

`eval/` is a folder, not a package: run everything from inside it (`cd eval`), or put it on
`PYTHONPATH`. The repository root holds the configs, baselines and job scripts it uses.

## Setup

```powershell
# PowerShell (laptop: tests and dry runs on CPU)
cd eval
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

On a GPU cluster, build `eval/.venv` with the pinned stack (`eval/requirements-gpu.txt`) inside a
job, after filling in `cluster.env` (copy `cluster.env.example`):

```bash
# Git Bash / Linux shell, from the repository root
SBATCH_EXTRA="--export=ALL,SETUP=1" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch --setup
QSUB_EXTRA="-l walltime=01:00:00 -v SETUP=1" bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
```

## Data

```bash
# from eval/ (or as a FETCH=1 job on a cluster)
python -m core.data fetch      # pinned revisions into $HARNESS_DATA (default eval/.cache/data, git-ignored)
python -m core.data verify     # MATCH / MISMATCH / MISSING per file
```

## Protocol runs

```bash
# from eval/
python run.py --approach fsa_lite_run3 --model gemma --suite core --variant original --dry-run   # CPU, fixtures
python run.py --approach fsa_lite_run3 --model gemma --suite core --variant technique --device cuda
```

- `--approach` names a spec in `configs/eval/<name>.approach.yaml`; `--suite core|second|synthetic|
  secret_in_context|all`; `--benchmarks a+b` for a subset.
- Adapters are found through `paths.local.yaml` (`adapter: {local: <key>}`; copy
  `paths.local.yaml.example`), or `FSA_ADAPTER_DIR=/path` for one job. The bridge checks the
  adapter's sha256 against the spec before the model loads and refuses on a mismatch.
- Output: `eval/outputs/protocol/<approach>/<model>/<variant>/<benchmark>/` with `items.jsonl`
  (one row per item, reply included: never commit it), `per_unit.json` and `summary.json`.

On a cluster:

```bash
bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch --approach fsa_lite_run3 --model gemma --suite core --variant technique
QSUB_EXTRA='-l walltime=03:00:00 -v RUN_ARGS="--approach fsa_lite_run3 --model qwen --suite core --variant technique"' \
  bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
```

Check an adapter on its model before a full run:
`bash scripts/submit_slurm.sh scripts/slurm/harness_adapter_check.sbatch` (`CHECK_ARGS="--approach fsa_lite_run3"`).

## Re-score and build the results table

```bash
# from eval/
python -m stats.rescore outputs/protocol
python -m stats.team_table build outputs/protocol --approach fsa_lite_run3 --model gemma --suite core --out ../results/fsa_lite_run3/gemma_core
```

`team_table.json` holds per-run paths and details and is git-ignored; commit the `.md` and `.csv`.

## Attack studies

```bash
# from eval/
python run.py --config configs/attack/hidden_code__none_fsa_lite_run3_gemma.yaml --dry-run
python -m stats.compare outputs/<run A> outputs/<run B>
```

Each `configs/attack/*__fsa_lite_run3_*.yaml` pair (adapter / original) is identical except
`name` and `defence` (`eval/tests/test_hidden_code_fsa_lite_run3.py`). `held_out: {approach: ...}`
makes the run refuse if any evaluated code is in the approach's private used-secrets file.

## Add a defence

Copy `eval/experiments/_template/` to `eval/experiments/<name>/` (or add a module to
`baselines/`), implement `Defence.wrap(model)` and register it with `@register("defence", "<name>")`,
then write `configs/eval/<name>.approach.yaml`. Heavy imports (torch, transformers, peft) go inside
functions, so the registry and `--dry-run` work on a CPU laptop.

## Other tools

- `eval/stats/leak_diagnosis.py`, `key_similarity.py`, `robust_leak.py`: report-only analyses of
  stored replies (counts and rates only).
- `eval/analysis/`: model-internals studies (`internals/`, `need_probe/`, `release_logprob/`); they
  reuse harness pieces but are not defences.
