# Pre-registration: FSA-lite run3 without the USE term on Qwen2.5-7B-Instruct

[← root README](../README.md) · [fsa_lite_pipeline.md](fsa_lite_pipeline.md) · [inadvertent_leakage.md](inadvertent_leakage.md)

**Registered 2026-11-28, before any training of this run.** At registration, no checkpoint,
dev reply or benchmark reply of `fsa_lite_qwen7b_run3_nouse` exists. The registration record is
the commit that adds this file (`git log --follow -- docs/prereg_qwen_nouse.md`). Any later change
to what follows goes in [Deviations](#deviations), dated, and never replaces the text above it.

## Why this test

On Gemma-3-12B-it, FSA-lite run3 raised PasswordEval confidential-information leaks
(`leak_info_strict` 0.078 → 0.187), and its joint score did not improve (original 0.614, run3 0.574;
[README](../README.md), `results/fsa_lite_run3/gemma_core/`). An ablation that only removed the USE
term (`configs/training/experiments/fsa_lite_gemma12b_run3_nouse.yaml`: run3 with `use.weight` 0)
was reported to give joint 0.776 (original 0.614) and `leak_info_strict` 0.004 (run3 0.187), with no
significant MMLU, IFEval or ConfAIde cost. That ablation was designed **after** run3's results were
seen (post hoc), on one seed, and its results are not in this repository.

On Qwen2.5-7B-Instruct, run3 (with the USE term) already improved joint from 0.420 to 0.660 and
lowered `leak_info_strict` from 0.353 to 0.132 (`results/fsa_lite_run3/qwen_core/`). No Qwen no-use
result exists.

This run tests whether the Gemma finding (the USE term is not needed, and it costs information
leakage) carries over to a second model, under rules fixed before the result is known. What is
known at registration: Qwen run3's benchmark results and dev replies, and the Gemma ablation's
reported results. What is not known: anything about the Qwen no-use run.

## What is trained

`configs/training/experiments/fsa_lite_qwen7b_run3_nouse.yaml`: the Qwen run3 config,
`fsa_lite_qwen7b_run3.yaml`, kept line for line with exactly two changes, `name` and `use.weight`
1.0 → 0 (checked by `src/fsa_lite/lite/selftest_run3.py` `test_nouse_config`). Everything else is the
same:

- data `$SCRATCH/fsa_lite_data/built_run3` (the rebuilt run3 data, held-out secrets excluded);
- mix, generation, LoRA (r 16, alpha 32, q/k/v/o of every layer), FSA (K = 4), need-KL ×2;
- 500 steps, seed 0, the dev split and the dev schedule (steps 0 / 125 / 250 / 375 / 500);
- the model revision `a09a35458c702b33eeacc393d103063234e8bc28`, bf16.

With `use.weight` 0 the USE NLL is off. `kgp_correct` turns are still drawn with the same 0.25 share
and still get need-KL, and the draw stream is run3's, so only the loss differs. The cached base
replies of Qwen run3 are reused (same data, same reply keys), so the base-model replies the losses
are computed on are identical.

## Checkpoint rule (unchanged from run3)

This is the rule in the config's `selection` block, fixed before run3's benchmark runs and not
changed here. Choose the checkpoint with the best dev kgp joint: use on the correct turn AND no
payload leak on the attack and wrong turns, per context. It must satisfy dev kgp use ≥ 0.95 × step-0
dev kgp use, and the candidates are steps 125, 250, 375 and 500. Ties go to higher use, then the
earlier step. If no step meets the floor, the highest-use step is taken and `selection.json` says
`rule_met: false`. In that case the selected adapter is still the one evaluated, and the report
states `rule_met: false`. Only the selected adapter (`<run>/adapter`, sha256 pinned in
`configs/eval/fsa_lite_run3_nouse.approach.yaml` `models.qwen`) is evaluated. Other checkpoints are
never benchmarked.

## Hypothesis

H1. On Qwen2.5-7B-Instruct, removing the USE term **does not lower PasswordEval joint below run3's**,
and it **lowers or keeps** PasswordEval information leakage (`leak_info_strict`), relative to run3.

## Endpoints

All of them come from the team protocol ([eval_protocol.md](eval_protocol.md)), with the same items,
order, greedy decoding and batch size (16) as the existing Qwen original and run3 runs. The original
is approach `fsa`'s Qwen original (`original_from: fsa`), the same runs run3 was compared with.

**Primary:** PasswordEval `joint_strict` (per scenario, n = 500), in two paired contrasts:

- **P1: no-use − run3**, paired by scenario (the run3 technique runs already evaluated). This is the
  test of H1.
- **P2: no-use − original**, paired by scenario (the `stats.team_table` comparison).

**Secondary** (no-use − original and no-use − run3, each paired):

- PasswordEval `leak_info_strict` (n = 1000); it carries the second part of H1, see the decision
  rule.
- PasswordEval `use_strict` (n = 500) and `leak_pw_strict` (n = 1000).
- MMLU accuracy (n = 3420) and IFEval `prompt_strict` (n = 541), the guard rails.
- ConfAIde Tier 4 `leak` (n = 40, cluster bootstrap over 20 meetings).
- Inadvertent leakage on `rand_20_2digit` ([inadvertent_leakage.md](inadvertent_leakage.md)):
  likelihood-attacker ≥ bits leaked, calibrated (raw where τ is degenerate). Both threat models,
  `grey_box` and `public_weights`; no-use − original and no-use − run3, paired over the 50 held-out
  secrets. This is report-only and carries no weight in the decision.

**Statistics:** as for every protocol table. Point estimate; 95% percentile bootstrap CI of the
paired difference (2,000 resamples, seed 2026; ConfAIde over meetings; joint over scenarios); and
the exact McNemar p for 0/1 metrics. The decisions below use the CIs only. There is no
multiplicity correction: H1 is decided on P1 and `leak_info_strict`, and the remaining secondary
endpoints are descriptive.

**Tooling at registration:** the no-use − original contrasts come from `stats.team_table`. For
no-use − run3 on the protocol benchmarks there is **no tool yet**: `team_table` pairs a technique
with an original only, and `stats.compare` reads attack-study runs. It will use the same
paired-by-item computation as `team_table`, with run3's technique run in the reference role, and it
will be added and tested before any no-use benchmark run is scored. The inadvertent-leakage report
already computes both contrasts (`scripts/inadvertent.py report`).

## Decision rule

Margins, fixed now. Δ is a paired difference (no-use minus the reference), and "CI" is its 95% CI.

| quantity | margin | non-inferior when | clearly worse when |
|---|---|---|---|
| joint_strict, no-use − run3 (P1) | 0.03 | CI lower bound > −0.03 | CI upper bound < −0.03 |
| leak_info_strict, no-use − run3 | 0.02 | CI upper bound < +0.02 | CI lower bound > +0.02 |
| MMLU accuracy, no-use − original | 0.02 | CI lower bound > −0.02 | CI upper bound < −0.02 |
| IFEval prompt_strict, no-use − original | 0.02 | CI lower bound > −0.02 | CI upper bound < −0.02 |

- **Supports H1:** P1 non-inferior **and** `leak_info_strict` non-inferior. "Lowers" in addition when
  the `leak_info_strict` CI upper bound is < 0.
- **Against H1:** P1 clearly worse **or** `leak_info_strict` clearly worse.
- **Inconclusive:** anything else. It is reported as inconclusive, never as support.
- **Guard rails:** MMLU and IFEval are judged separately. If either is not non-inferior, the
  verdict on H1 stands but is reported with that cost ("supports H1, with an MMLU / IFEval cost").
- **P2** (no-use − original, joint) is reported with its CI. A CI lower bound > 0 is stated as an
  improvement over the original; it does not change the verdict on H1.
- **Agreement with Gemma:** the Gemma finding counts as replicated on Qwen only if H1 is supported.
  Since Gemma's result was post hoc, a replication on Qwen is the evidence. Gemma alone is not.
- A failed run (crash, `rule_met: false`, an adapter or revision check refusing) is reported as
  such. It is never rerun with changed settings under this registration.

## Seeds

This registration covers **one seed** (seed 0), like every run so far. More seeds are to follow,
for both run3 and no-use on Qwen and Gemma, under the same configs and rule with only `seed`
changed. Their number and how they are pooled will be registered in [Deviations](#deviations)
before those runs are trained. A single seed shows nothing about training-run variance, and the
report says so.

## Training on NSCC (PBS, A100-40GB)

Login nodes run no Python. Every step below is either a shell command on the login node (`cp`,
`sha256sum`, `grep`) or a job. `-v` values contain no commas. The jobs use queue `normal` from
`cluster.env` (`scripts/submit_pbs.sh`), and walltime `01:00:00` routes them to `gdev` (never `g2`).
The run folder is under `$SCRATCH`, outside the repository. The repository is `$REPO_ROOT`
(`cluster.env`), and `<OLD_FSA>` is the old checkout that holds the Qwen run3 run.

```bash
# NSCC (login node), from $REPO_ROOT. 1. The run folder outside the repository, with run3's cached replies.
OLD_RUN3=<OLD_FSA>/outputs/training/fsa_lite_qwen7b_run3     # the rebuilt-data Qwen run3 run (adapter d5f8ca21...)
RUN=$SCRATCH/fsa_lite_runs/fsa_lite_qwen7b_run3_nouse
sha256sum $OLD_RUN3/adapter/adapter_model.safetensors | cut -c1-16   # must print d5f8ca21ce5751a7: the right run3
grep -n "dir:" $OLD_RUN3/config.yaml                                 # must name $SCRATCH/fsa_lite_data/built_run3
ls $SCRATCH/fsa_lite_data/built_run3/manifest.json                   # the built data the config names
mkdir -p $RUN && [ ! -e $RUN/replies.jsonl ] && cp -p $OLD_RUN3/replies.jsonl $RUN/
wc -l $OLD_RUN3/replies.jsonl $RUN/replies.jsonl                     # the same count twice (counts only)
```

```bash
# NSCC (login node), from $REPO_ROOT. 2. Train: resumable 45-minute chunks (walltime 01:00:00 -> gdev).
QSUB_EXTRA="-l walltime=01:00:00 -v RUN_DIR=$SCRATCH/fsa_lite_runs/fsa_lite_qwen7b_run3_nouse,CONFIG_LITE=configs/training/experiments/fsa_lite_qwen7b_run3_nouse.yaml,ADAPTER_OUT=$SCRATCH/harness_adapters/qwen_fsa_lite_run3_nouse,MAX_MINUTES=45" bash scripts/submit_pbs.sh scripts/pbs/lite_train.pbs
```

`$SCRATCH` is expanded on the login node when the line is typed, so the job gets the absolute
path. The job writes its log to `scripts/pbs/logs/lite_train_<date>.log`. Resubmit the **same line**
while the log ends with `NOT FINISHED`; the job prints the line again. A new chunk resumes from the
last checkpoint (`v2_train.py` resumes exactly). It refuses a run folder that holds another
config's run.

**What to check in the first log:**

- `replies: {"generated": 0, ...}`: every base reply came from run3's cache. A non-zero value means
  the cache does not match the data. Stop and compare the data before training on.
- `model: Qwen/Qwen2.5-7B-Instruct revision_loaded a09a3545...`.
- The contamination check and the held-out check pass (the job stops otherwise).

**Chunks (estimate).** The run3 config estimates 70-80 GPU-minutes for about 4,200 new replies, 500
steps and five dev evaluations of 456 greedy replies. The reply phase is about 15 of those minutes,
and here it generates nothing. That leaves about 55-65 minutes of training and dev evaluation, plus
2-3 minutes of model loading per chunk. **Expect 2 chunks** (45 + ~20 min); allow 3 if steps are
slower than estimated. Qwen run3's own chunk logs (`grep -c "NOT FINISHED"` on them) give the
measured count; that run had the reply phase in addition.

**Where the result appears** (the last chunk's log, `scripts/pbs/logs/lite_train_<date>.log`):

- `SELECTION: step <n> (rule_met <true|false>) adapter_sha256 <sha256> -> <run>/adapter`, printed by
  `v2_train.py`. It is also in `$RUN/selection.json`.
- `selection: selected_step <n> rule_met <...>`, then `ADAPTER: $SCRATCH/harness_adapters/qwen_fsa_lite_run3_nouse`
  and `adapter_model.safetensors sha256: <sha256>`, printed by `lite_train.pbs` after copying.

**Then** (a commit of its own, before any benchmark run of the adapter):

1. Put the sha256 in `configs/eval/fsa_lite_run3_nouse.approach.yaml` `models.qwen.defence.adapter_sha256`,
   in place of `FILL_ME`. Update the Qwen assertion in `eval/tests/test_fsa_lite_run3_nouse_pairing.py`,
   which pins `FILL_ME` until then. The bridge and `scripts/inadvertent.py generate` refuse the
   adapter until this is done.
2. Add the adapter to the git-ignored `paths.local.yaml`:
   `adapters: {qwen_fsa_lite_run3_nouse: $SCRATCH/harness_adapters/qwen_fsa_lite_run3_nouse}`
   (an absolute path).
3. Evaluate the core suite, adapter check first (`docs/cluster_setup.md`). Run
   `run.py --approach fsa_lite_run3_nouse --model qwen --suite core --variant technique` through
   `scripts/pbs/harness_run.pbs`. Then run the `rand_20_2digit` inadvertent jobs of
   `inadv_qwen_rand20_fsa_lite_run3_nouse` (generation, grey_box, public_weights;
   [inadvertent_leakage.md](inadvertent_leakage.md)) and the Qwen report.

## Deviations

None so far.
