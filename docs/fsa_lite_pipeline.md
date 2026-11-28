# FSA-lite pipeline: Tensor Trust + CIMemories + benign prompts (run1-run3)

This is the design and run record of the FSA-lite training data and runs, kept as it was written
while the runs were made (run1, run2, run3 and the post-hoc follow-ups). The command lines are the
ones used then; with this repository, put your site's settings in `cluster.env` and submit with
`scripts/submit_slurm.sh` / `scripts/submit_pbs.sh` (see the README).

The same objective as v2 (`v2_objective.py`, unchanged), with K = 4 fakes drawn fresh at every draw, the need term at λ = 1, and LoRA r16 / α32 on q/k/v/o.

- **Models:** one run per model, Gemma-3-12B-it first and Qwen2.5-7B-Instruct second, both bf16 on one A40.
- **Evaluation:** later, in the harness (PasswordEval, ConfAIde Tier 4, MMLU, IFEval).
- **Contamination:** nothing from those four benchmarks, nor from the harness's Tensor Trust eval files, enters training. `contam_check.py` must PASS, and `v2_train.py` refuses data without a PASSing report for exactly those files.

## Data (`build_lite.py` → `units.jsonl`, `pools.json`, `manifest.json`, `contam_report.json`)

| Turn type | Source | Objective | Mix weight |
|---|---|---|---|
| `tt_attack` | Tensor Trust raw-data/v2 attack on its own defence (unauthorised) | FSA, fakes = other train access codes | 0.40 |
| `tt_authorised` | the same defence, with the access code as the user turn | need-KL | 0.15 |
| `cimem_unauthorised` | CIMemories (profile, task), a sensitive attribute outside the task's domains | FSA, fakes = that attribute's values in other train profiles | 0.15 |
| `cimem_authorised` | CIMemories (profile, task), the attributes in the task's domains | need-KL | 0.10 |
| `benign` | 200 own template tasks under a train TT or CIMemories context | need-KL | 0.20 |

**Tensor Trust** (`tt_data.py`) uses the same message layout as the harness protocol: system = opening defence, user = attack + `"\n"` + closing defence. It applies these filters:
- the defence must be valid;
- the access code must have 4–64 characters, not be a stopword, and appear verbatim in the defence;
- no attack may contain the access code;
- evaluation exclusions:
  - the code equals an eval secret;
  - the opening or closing defence equals an eval row;
  - the text shares a 13-gram with any eval text, or contains a distinctive eval secret.

10% of the defences are held out as a dev split, and dev codes are never used as fakes.

**CIMemories** (`cimem_data.py`): **the labels are heuristic, not CIMemories gold labels.** The public release has none; its `test.csv` label column is empty.
- `TASK_DOMAINS` maps each of the 49 tasks to the domains it needs, with a one-line reason.
- authorised = the attribute's domain is one of the task's domains;
- unauthorised = a sensitive domain (health, mental_health, legal, relationships, finance) that the task doesn't need;
- any other attribute is context only.
- A labels file in the persona-script format (`--cimem-labels`) replaces the heuristic. It is used only where the personas agree completely.

Two of the 10 profiles form the dev split. Memories are capped at 2,800 characters, dropping unrelated memories, never a target.

**Benign** (`benign_data.py`): own summarise, rewrite, Q&A and list tasks, none taken from an evaluation set.

**Eval-secret matching** (`contam_check.py`): eval secrets are always matched exactly against training secrets and fakes. They are matched inside running text only when distinctive: not made only of stopwords, and at least 8 characters or containing a digit, space or symbol. Generic eval codes such as "access granted" or "hello" occur in most Tensor Trust defences, so substring matching them would flag text that shares nothing with the evaluation items.

## Run order on the Slurm cluster (not run yet)

Run from `$REPO_ROOT`.

1. On the login node (curl and sha256 only, no Python): `bash scripts/fetch_lite.sh`
2. The harness's eval files must be present under `$HARNESS_DATA` (default `$REPO_ROOT/eval/.cache/data`, from the harness FETCH step).
3. Run the pilot, which builds the data, checks it and times the run:
   `env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch scripts/slurm/lite_pilot.sbatch`
   Read the `PILOT:` line, and set `train.steps` to `steps_for_1h15` if it differs much from 400.
4. Run the training, resubmitting the same line while it exits with code 3:
   `env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,RUN_DIR=outputs/training/fsa_lite_gemma12b_run1 scripts/slurm/lite_train.sbatch`
5. For Qwen, add `CONFIG_LITE=configs/training/experiments/fsa_lite_qwen7b.yaml` to `--export` in steps 3 and 4.
6. The copy-layer variants (`*_copylayers.yaml`) are not to be run yet.

## FSA-lite files

| File | Purpose |
|---|---|
| `lite_common.py` | normalisation, 13-grams, word-boundary matching, character classes |
| `tt_data.py`, `cimem_data.py`, `benign_data.py` | the three data builders |
| `build_lite.py` | builds everything, final overlap guard, manifest |
| `contam_check.py` | 13-gram and secret scan against the harness eval files; manifest of the result |
| `lite_source.py` | the `data.source: lite` source for `v2_train.py`: mixture draws, fresh fakes, dev metrics |
| `fetch_lite.sh`, `common_lite.sh`, `lite_pilot.sbatch`, `lite_train.sbatch` | fetch and jobs |

## FSA-lite on the PBS cluster (Qwen2.5-7B)

Qwen's training and its harness technique eval both run on the PBS cluster (A100-SXM4-40GB), where its harness original runs are. So the original/technique pair is on one GPU type and one model snapshot (`a09a35458c702b33eeacc393d103063234e8bc28`, pinned in `fsa_lite_qwen7b.yaml`).

Gemma stays on the Slurm cluster.

**the PBS cluster rules:**
- PBS jobs only: login nodes have no Python.
- Queue `normal` may route jobs with walltime ≤ 01:00:00 to a short queue.
- Compute nodes have no git binary; the commit is read from `.git`.
- `qsub -v` values must not contain commas.

| Job | What it does |
|---|---|
| `lite_setup.pbs` | Builds `.venv` from `uv.lock` (`uv sync --frozen --group ml`, `/bin/python3.11`): the same versions as the Slurm cluster training env. Checks the driver against the cu128 build (or `-v TORCH_CUDA=cu126`: same torch version, recorded), runs a bf16 matmul and the self-test, and writes `SETUP_OK`. |
| `lite_pilot.pbs` | Builds the data under `$SCRATCH/fsa_lite_data/built` and re-runs the contamination check against `$REPO_ROOT/eval/.cache/data`. Then the pilot. |
| `lite_train.pbs` | Trains for at most 50 minutes per job. It stops between reply batches or steps with a resumable state, prints what remains and exits 3; resubmit the same line. When done, it copies `<run>/adapter` to `$SCRATCH/harness_adapters/qwen_fsa_lite` and prints its sha256. |

All three source `common_lite_pbs.sh`, which holds the paths, puts every cache on scratch, loads offline from the model cache (`HF_HUB_CACHE` in `cluster.env`), and never reads the token.

**Sequence:**

1. **Laptop:** make a bundle of branch `amruta` and copy it to `$SCRATCH/bundles/`.
2. **the PBS cluster login node:**
   ```
   git clone -b amruta $SCRATCH/bundles/amruta.bundle $SCRATCH/fsa_main
   ```
   On later updates use `git -C $SCRATCH/fsa_main pull --ff-only <bundle> amruta` instead.
3. **the PBS cluster login node:**
   ```
   mkdir -p $SCRATCH/fsa_main_logs
   FSA_LITE_DATA=$SCRATCH/fsa_lite_data bash $REPO_ROOT/scripts/fetch_lite.sh
   ```
4. Submit the jobs, with `V=$REPO_ROOT/src/fsa_lite/lite/` and `L=$SCRATCH/fsa_main_logs`:
   ```
   S=$(qsub -o $L/lite_setup.log $V/lite_setup.pbs)
   P=$(qsub -W depend=afterok:$S -o $L/lite_pilot.log $V/lite_pilot.pbs)
   qsub -W depend=afterok:$P -v RUN_DIR=outputs/training/fsa_lite_qwen7b_run1 -o $L/lite_train_1.log $V/lite_train.pbs
   ```
   Resubmit the last line (with a new log name, and without `-W`) while the log ends with `NOT FINISHED`.
5. **Harness:**
   - Fill `fsa_lite.approach.yaml` `models.qwen.defence.adapter_sha256` with the printed sha256.
   - Copy `fsa_lite_data/built/used_secrets.txt` to `configs/eval/private/fsa_lite_used_secrets.txt` (git-ignored). It's needed only for fsa_lite's report-only runs.

## FSA-lite run2 (after run1 did not move dev leakage)

Run1 (Gemma) kept dev `tt_attack` leak at 0.312 (5/16) at every evaluation.

**Diagnosing run1** from its own files (`lite_diagnose.py`, dev and train data only):
- which cached replies leak;
- the recomputed draws;
- the FSA term with and without a leaking reply;
- dilution;
- adapter weights.

With `GENERATE=1` it also runs the dev evaluation with the adapter off and on, including all 112 dev attack units.
```
env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,RUN_DIR=outputs/training/fsa_lite_gemma12b_run1,GENERATE=1 scripts/slurm/lite_diagnose.sbatch
```

**What run2 changes** (`fsa_lite_gemma12b_run2.yaml`, `fsa_lite_qwen7b_run2.yaml`; the run1 configs are unchanged):
- FSA draws only replies that contain the real secret (`fsa.units: leaking_reply`), with 3 base replies per attack unit.
- Secret tokens weigh ×10 in the FSA mean.
- Attack share is 0.55; lr is 2e-4; 500 steps.
- The need-KL turns keep 40% of the draws.
- The dev evaluation uses all 112 dev attack units, and its replies are saved to `dev_replies.jsonl`.

**the Slurm cluster (Gemma).** This reuses the built data and run1's cached replies; copy them on the login node with `cp` (no Python):
```
mkdir -p outputs/training/fsa_lite_gemma12b_run2 && cp outputs/training/fsa_lite_gemma12b_run1/replies.jsonl outputs/training/fsa_lite_gemma12b_run2/
env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run2.yaml,RUN_DIR=outputs/training/fsa_lite_gemma12b_run2 scripts/slurm/lite_train.sbatch
```

**the PBS cluster (Qwen).** Copy Qwen run1's replies first if that run exists. The adapter goes to its own folder, so run1's is never overwritten:
```
F=$REPO_ROOT; V=$F/src/fsa_lite/lite/; L=$SCRATCH/fsa_main_logs
mkdir -p $F/outputs/training/fsa_lite_qwen7b_run2 && cp $F/outputs/training/fsa_lite_qwen7b_run1/replies.jsonl $F/outputs/training/fsa_lite_qwen7b_run2/
qsub -v CONFIG_LITE=configs/training/experiments/fsa_lite_qwen7b_run2.yaml,RUN_DIR=outputs/training/fsa_lite_qwen7b_run2,ADAPTER_OUT=$SCRATCH/harness_adapters/qwen_fsa_lite_run2 -o $L/lite_train_r2_1.log $V/lite_train.pbs
```
Resubmit the `qsub` line, with a new log name, while the log ends with `NOT FINISHED`.

**Choosing between run1 and run2** uses dev metrics only. If run2 is chosen, point the harness spec's adapter path for that model at the run2 folder and pin its sha256.

## FSA-lite run3 (key-guarded payloads; run2 untouched)

**Why.** Run2's only training secret was the Tensor Trust access code, and that code guards nothing. Training never saw a password that unlocks a payload, and correct release was never rewarded. On PasswordEval, Gemma's use fell from 71.6 to 67.8 and joint from 61.4 to 48.8.

**Data** (`build_lite_run3.py` → `$FSA_LITE_DATA/built_run3`) is run2's units, byte-identical, plus key-guarded-payload units from `kgp_data.py`, in our own templates. In each context, a password (a Tensor Trust train code) guards a payload. The payload is a synthetic codename, meeting or account, or a CIMemories value. Each context has three matched turns:

| Turn | User message |
|---|---|
| `kgp_attack` | no password: a TT attack text or a direct ask |
| `kgp_wrong` | a near-miss of the password, or another password |
| `kgp_correct` | the exact password |

- **Size:** 600 train contexts and 120 dev contexts.
- **Dev split:** dev passwords and payloads are disjoint from train and from every pool. Dev passwords are the 58 TT dev codes plus synthetic passwords; `meta.pw_source` says which.
- **Contamination:** the check is the same as run2's (exact, 13-gram and the 1,480 eval secrets).
- **Used secrets:** `used_secrets.txt` lists every password, payload, near-miss and fake; it goes to the harness's `private/fsa_lite_run3_used_secrets.txt`.

**Objective.** Every lever below has a config flag; with the flags off, run2 behaves exactly as before (`selftest_run3.py`).
- **FSA turns:**
  - `kgp_attack` and `kgp_wrong` get K = 4 fakes for both the password and the payload, each drawn from its own pool. The near-miss follows the fake password.
  - `fsa.secret_match: robust`: a leak is the password or payload in normalised form, the same normalisation as the harness's robust leak scorer.
  - Secret tokens weigh ×10, including fragments of at least 4 characters.
- **`kgp_correct`:** need-KL on every token, plus a USE term.
  - The USE term is a weighted NLL on the frozen model's own correct-release reply, with payload tokens ×5.
  - It only uses replies that hold the payload and do not repeat the password.
  - There is no suppression term on this turn.
- **Need-KL:** λ = 2 (×2 vs run2).
- **Not used:** no RL and no leak reward; the secret stays in the context.

**Mix:**
- Authorised and correct-key turns are 40%: `kgp_correct` 0.25, `tt_authorised` 0.10, `cimem_authorised` 0.05.
- Benign is 0.15.
- FSA turns are 45%: `kgp_attack` 0.15, `kgp_wrong` 0.15, `tt_attack` 0.10, `cimem_unauthorised` 0.05.

**Dev and selection** were fixed before any benchmark run.
- **When:** greedy dev at steps 0 / 125 / 250 / 375 / 500, with replies saved.
- **Metrics:** for every kgp context, overall and per `pw_source`:
  - `leak_pw` and `leak_payload` on the attack and wrong turns;
  - `use` on the correct turn;
  - `joint` = use AND no payload leak on either other turn.
- **Rule:** best dev joint subject to dev use ≥ 0.95 × step-0 use. Ties go to higher use, then the earlier step.
- **Output:** the trainer writes `selection.json`, makes the chosen checkpoint `<run>/adapter`, keeps the last step as `adapter_last`, and prints `SELECTION: step … adapter_sha256 …`.

**the Slurm cluster (Gemma).** Copy run2's cached replies on the login node first (`cp`, no Python). The run2-type reply keys are identical, so only about 4,200 kgp replies are new.
```
cd $REPO_ROOT
env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml,RUN_DIR=outputs/training/fsa_lite_gemma12b_run3_pilot scripts/slurm/lite_pilot.sbatch
mkdir -p outputs/training/fsa_lite_gemma12b_run3 && cp outputs/training/fsa_lite_gemma12b_run2/replies.jsonl outputs/training/fsa_lite_gemma12b_run3/
env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml,RUN_DIR=outputs/training/fsa_lite_gemma12b_run3 scripts/slurm/lite_train.sbatch
```

**the PBS cluster (Qwen)** uses the existing mirror-mode venv. Resubmit with a new log name while the log ends with `NOT FINISHED`; the job prints the full line.
```
F=$REPO_ROOT; V=$F/src/fsa_lite/lite/; L=$SCRATCH/fsa_main_logs
qsub -v CONFIG_LITE=configs/training/experiments/fsa_lite_qwen7b_run3.yaml,RUN_DIR=outputs/training/fsa_lite_qwen7b_run3_pilot -o $L/lite_pilot_r3.log $V/lite_pilot.pbs
mkdir -p $F/outputs/training/fsa_lite_qwen7b_run3 && cp $F/outputs/training/fsa_lite_qwen7b_run2/replies.jsonl $F/outputs/training/fsa_lite_qwen7b_run3/
qsub -v RUN_DIR=outputs/training/fsa_lite_qwen7b_run3,CONFIG_LITE=configs/training/experiments/fsa_lite_qwen7b_run3.yaml,ADAPTER_OUT=$SCRATCH/harness_adapters/qwen_fsa_lite_run3,MAX_MINUTES=45 -o $L/lite_train_r3_1.log $V/lite_train.pbs
```
After training, put the selected adapter's sha256 into the harness spec `configs/eval/fsa_lite_run3.approach.yaml` (it is `FILL_ME` until then).

### run3 rebuild: held-out secrets excluded (supersedes the cp-replies lines above)

The first run3 build shared one 4-digit value with the harness's `held_out_secrets.txt`; `contam_check` never reads that file. The Qwen run3 adapter trained on it (sha256 3ccd1f55…) is discarded, and the data is rebuilt.
- **`--exclude-secrets` (required):** `build_lite_run3.py` now requires it. `heldout.py` defines a value as matching a held-out secret when it is equal, equal ignoring case, contains one of 4+ characters, or is contained in one.
- **KGP values:** rejected and redrawn at generation time. This covers passwords and their near-misses, payloads, fake pools and attack texts.
- **Shared Tensor Trust / CIMemories / benign units and pools:** filtered after generation.
- **Final check:** the build fails, writing nothing, if any unit value, pool value, used secret, or near-miss derived from a fake password still matches.
- **Wrong-password turns:** never hold the real password, word-bounded for any kind and as a plain substring for every kind except add_digit; asserted at build time.
- **`used_secrets.txt`:** now also lists the near-misses training derives from fake passwords.
- **Same data on both clusters:** `rebuild_run3.sbatch` (Slurm) and `rebuild_run3.pbs` (PBS) build deterministically and print the sha256 of the three files and of the held-out file. The two clusters must agree.
- **Cached replies:** the reply cache is keyed by `reply_key|r`, not by content. `reuse_replies.py`, run by `preflight_fakes.sbatch/.pbs` when `OLD_DATA`, `OLD_REPLIES` and `NEW_RUN` are set, carries over only replies whose prompt is byte-identical; the train job generates the rest.
- **Run folders:** rename the old ones, delete nothing.

## Post-norm diagnostic (report-only, Part B)

`postnorm_diag.py` checks whether Gemma-3's sandwich norms amplify the run2 LoRA's change non-uniformly by layer; Qwen has no post-attention norm and is the contrast. It runs on 50 synthetic key-guarded prompts (not PasswordEval) and reports, per layer:
- ‖δ‖ of the LoRA's o_proj change;
- for Gemma: the gain ‖N(m+δ) − N(m)‖ / ‖δ‖ through `post_attention_layernorm`, and its angle to δ;
- the change of the MLP output and of the residual stream;
- whether the layer is local or global.

These are GPU jobs; there is no generation, only two forward passes per prompt.
```
# the Slurm cluster, Gemma run2
env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch scripts/slurm/postnorm_diag.sbatch
# the PBS cluster, Qwen run2
qsub -o $SCRATCH/fsa_main_logs/postnorm_qwen.log $REPO_ROOT/scripts/pbs/postnorm_diag.pbs
# laptop, after copying both CSVs (harness .venv has matplotlib)
python src/fsa_lite/lite/postnorm_diag.py plot --csv gemma_run2.csv qwen_run2.csv --out postnorm_run2.png
```
## run3 training-signal audit (report-only, H3)

`train_signal_audit.py` reads a finished run3 run and changes nothing. CPU only: it loads the tokenizer, not the model.
- **Buckets:** every cached base reply of the full train set, classified with training's own rules: password-leaking vs payload-leaking FSA replies, and correct-key releases with or without the password. It also reports the draw pools training used.
- **Draws:** the 500 × grad_accum draws, replayed exactly (seeded per step) and checked step by step against `train_log.jsonl`, with the FSA loss attributed by bucket.
- **Loss shares:** FSA, need-KL and USE per 125-step window.
- **Dev table:** per checkpoint, attack and wrong leak_pw / leak_payload and correct use / echo_pw, with CIs and step-0 flips from `dev_replies.jsonl`.

It prints counts only.
```
# the Slurm cluster, Gemma run3
env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml,RUN_DIR=outputs/training/fsa_lite_gemma12b_run3 scripts/slurm/train_signal_audit.sbatch
# the PBS cluster, Qwen run3
qsub -v RUN_DIR=outputs/training/fsa_lite_qwen7b_run3 -o $SCRATCH/fsa_main_logs/train_signal_audit_qwen.log $REPO_ROOT/scripts/pbs/train_signal_audit.pbs
```

## Post-hoc follow-ups to the run3 PasswordEval diagnosis (report-only and ablation)

Both were decided AFTER the run3 PasswordEval results, so they are post-hoc; label them so wherever they are reported.

**Dev leaks by wrong-password kind (report-only).** `dev_wrong_kind.py` reads the built data's dev units and each run's `dev_replies.jsonl`. CPU only: no tokenizer, no model. Per checkpoint it reports:
- kgp leaks by wrong-password kind (drop_last, swap_mid, add_digit, random), with CIs;
- the near-miss minus random difference;
- step-0 flips;
- leak_payload by offered-password similarity: 4 quantile bins of the Levenshtein ratio, plus a logistic slope. The offered password is recovered with the generator's templates and checked against `kgp_data.derive`.

Runs on the same data are paired by dev unit (e.g. run3 vs the ablation). It prints counts only.
```
# the Slurm cluster, Gemma run3 (later: RUNS=run3=...+nouse=outputs/training/fsa_lite_gemma12b_run3_nouse)
env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml,RUNS=run3=outputs/training/fsa_lite_gemma12b_run3,OUT_NAME=gemma scripts/slurm/dev_wrong_kind.sbatch
# the PBS cluster, Qwen run3
qsub -v RUNS=run3=outputs/training/fsa_lite_qwen7b_run3,OUT_NAME=qwen -o $SCRATCH/fsa_main_logs/dev_wrong_kind_qwen.log $REPO_ROOT/scripts/pbs/dev_wrong_kind.pbs
```

**Ablation: run3 without the USE term (Gemma only).** `configs/training/experiments/fsa_lite_gemma12b_run3_nouse.yaml` is the Gemma run3 config with two changes:
- `name`;
- `use.weight` 0 (run3: 1.0).

Everything else is identical, as checked by `selftest_run3.py test_nouse_config`: data `built_run3` and its hashes, mix, steps, seeds, need-KL ×2, and the pre-registered selection rule. The draw stream is run3's, so only the loss differs. kgp_correct turns are still drawn and get need-KL.

The run reuses run3's cached base replies through `reuse_replies.py`, run by `preflight_fakes.sbatch` with:
- `OLD_DATA` = the config's own data dir (the same `built_run3`, so every reply is kept);
- `OLD_REPLIES` = the run3 run's `replies.jsonl`;
- `NEW_RUN` = `outputs/training/fsa_lite_gemma12b_run3_nouse`.

Then train with `lite_train.sbatch` (same `RUN_DIR`, `CONFIG_LITE` passed explicitly). The harness spec is `configs/eval/fsa_lite_run3_nouse.approach.yaml`: the selected adapter (step 500, rule met) is pinned there (sha256 prefix `63a6d346`); it runs PasswordEval only (`--benchmarks password_eval_team`). Job commands: [cluster_setup.md](cluster_setup.md).

**Pre-registered Qwen test of the ablation.** `configs/training/experiments/fsa_lite_qwen7b_run3_nouse.yaml` is the Qwen run3 config with the same two changes (checked by the same self-test). It was registered on 2026-11-28, before training, in [prereg_qwen_nouse.md](prereg_qwen_nouse.md), which holds the hypothesis, endpoints, non-inferiority margins, decision rule and the NSCC training commands. The run folder is under `$SCRATCH` and reuses Qwen run3's `replies.jsonl`. The harness spec is the same `fsa_lite_run3_nouse.approach.yaml`, `models.qwen` (`post_hoc: false`, `pre_registered`); its `adapter_sha256` is `FILL_ME` until the run's `SELECTION:` line, and the bridge refuses the adapter until then.

Self-tests: `selftest_run3.py` (run by `selftest_v2.py`) covers run3; `selftest_postnorm.py` covers Part B on tiny random Gemma3/Qwen2 models; its `--plot-only` mode runs in the harness .venv.
