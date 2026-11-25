# Results (FSA v1): fake-secret average

FSA v1 is the earlier version of fake-secret averaging, before FSA-lite (`src/fsa_lite/lite/`): 2-digit
and 4-digit codes in a single system prompt, Gemma-3-12B-it only. This folder holds its **aggregate
results only**. Its notes are in [`docs/fsa/`](../../docs/fsa/README.md).

## What is here

| File | Contents |
|---|---|
| `README.md` | this page: the leakage and utility tables below |
| `summary.csv`, `summary.md` | one row per (run, likelihood-attack file): heuristic and likelihood attacker bits with 95% CIs, exact / top-5 rates, calibration temperature |
| `export_manifest.json` | sha256 of every file in this folder |

**Not published.** The per-run files (`config.yaml` with the run's secret lists, `metrics.json`,
`diagnostics.json` with per-code counts, `likelihood_attack*.json` with per-code scores and ranks,
`utility_metrics.json`), raw generations, Slurm logs and adapter weights are kept privately and are
not part of this repository. The numbers below and in `summary.*` were computed from them.

## Leakage

Gemma-3 12B, 50 held-out secrets each. ≥ bits = bits leaked (lower bound) = log2(100) − the attacker's mean cross-entropy on the true secret, clipped at 0, with its 95% bootstrap CI (max 6.644). Heuristic = the run's own attacker (metrics.json). Likelihood columns: one file per threat model, the calibrated one first, then the newest (the same rule as `summary.md`); files with identical scores count once. `summary.md` lists every file.

| Run | run_id | Prompt | Method | Adapter (training run) | Threat models | Heuristic ≥ bits [95% CI] | Heuristic exact | Grey-box ≥ bits [95% CI] | Public weights ≥ bits [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| rand20 / base | `20261019-142249_baseline_gemma3-12b-it_base` | rand_20_2digit | base | — | grey-box | 1.256 [0.835, 1.689] | 16.0% | 6.644 [6.644, 6.644] *raw* | — |
| rand20 / prompt hiding | `20261019-144738_baseline_gemma3-12b-it_prompt_hiding` | rand_20_2digit | prompt_hiding | — | grey-box | 0.925 [0.471, 1.319] | 4.0% | 6.644 [6.644, 6.644] *raw* | — |
| rand20 / prompt hiding (strict) | `20261019-153905_baseline_gemma3-12b-it_prompt_hiding_strict` | rand_20_2digit | prompt_hiding_strict | — | grey-box | 0.000 [0.000, 0.063] | 1.0% | 3.581 [3.009, 4.110] cal | — |
| rand20 / prompt anti-suppression | `20261019-151532_baseline_gemma3-12b-it_prompt_anti_suppression` | rand_20_2digit | prompt_anti_suppression | — | grey-box | 0.091 [0.000, 0.424] | 4.0% | 6.644 [6.644, 6.644] *raw†* | — |
| rand20 / FSA k=8 | `20261024-052409_baseline_gemma3-12b-it_fsa` | rand_20_2digit | fsa | 20261024-023638_fsa_gemma12b | grey-box, public weights | 0.000 [0.000, 0.000] | 1.0% | 6.368 [5.983, 6.644] cal | 2.839 [1.914, 3.741] cal |
| rand20 / FSA k=49 (old fullmix) | `20261024-231807_baseline_gemma3-12b-it_fsa` | rand_20_2digit | fsa | 20261024-184429_fsa_gemma12b_fullmix | grey-box, public weights | 0.000 [0.000, 0.000] | 1.0% | 0.270 [0.149, 0.392] cal | 0.000 [0.000, 0.000] *raw†* |
| rand20 / B3 baseline (match the no-secret model) | `20261025-231007_baseline_gemma3-12b-it_b3` | rand_20_2digit | b3 | 20261025-052549_b3_gemma12b | grey-box, public weights | 0.000 [0.000, 0.000] | 1.0% | 0.056 [0.000, 0.302] cal | 0.000 [0.000, 0.029] cal |
| rand20 / FSA k=49 multiprompt | `20261108-202641_baseline_gemma3-12b-it_fsa` | rand_20_2digit | fsa | 20261108-024957_fsa_gemma12b_fullmix_multiprompt | grey-box | 0.000 [0.000, 0.000] | 1.0% | 0.468 [0.068, 0.901] cal | — |
| lottery / base | `20261107-222330_baseline_gemma3-12b-it_base` | lottery | base | — | grey-box | 4.949 [4.236, 5.534] | 71.7% | 6.644 [6.644, 6.644] *raw†* | — |
| lottery / FSA k=49 (old fullmix) | `20261107-230410_baseline_gemma3-12b-it_fsa` | lottery | fsa | 20261024-184429_fsa_gemma12b_fullmix | grey-box, public weights | 0.000 [0.000, 0.000] | 1.0% | 1.597 [1.040, 2.147] cal | 0.014 [0.000, 0.087] cal |
| lottery / FSA k=49 multiprompt | `20261108-151123_baseline_gemma3-12b-it_fsa` | lottery | fsa | 20261108-024957_fsa_gemma12b_fullmix_multiprompt | grey-box | 0.000 [0.000, 0.049] | 0.0% | 2.936 [2.343, 3.568] cal | — |
| unpredictabench / base | `20261108-151352_baseline_gemma3-12b-it_base` | unpredictabench_2digit | base | — | grey-box | 0.000 [0.000, 0.135] | 2.0% | 1.630 [1.037, 2.250] cal | — |
| unpredictabench / FSA k=49 multiprompt | `20261108-194245_baseline_gemma3-12b-it_fsa` | unpredictabench_2digit | fsa | 20261108-024957_fsa_gemma12b_fullmix_multiprompt | grey-box | 0.000 [0.000, 0.000] | 1.0% | 0.041 [0.000, 0.194] *raw†* | — |
| unpredictabench / FSA k=49 (old fullmix) | `20261109-103335_baseline_gemma3-12b-it_fsa` | unpredictabench_2digit | fsa | 20261024-184429_fsa_gemma12b_fullmix | grey-box, public weights | 0.016 [0.000, 0.088] | 0.7% | 0.040 [0.000, 0.215] cal | 0.000 [0.000, 0.000] *raw†* |
| pe4 lottery_4digit / base | `20261114-051157_baseline_gemma3-12b-it_base` | lottery_4digit | base | — | grey-box | 0.009 [0.005, 0.015] | 0.6% | 6.644 [6.644, 6.644] *raw†* | — |
| pe4 lottery_4digit / FSA lambda=0 (pe4 fullmix) | `20261114-122212_baseline_gemma3-12b-it_fsa` | lottery_4digit | fsa | 20261113-165343_fsa_gemma12b_pe4_fullmix | grey-box | 0.018 [0.000, 0.107] | 2.6% | 1.410 [0.830, 1.983] cal | — |
| pe4 lottery_4digit / FSA lambda=1 (need term) | `20261114-124824_baseline_gemma3-12b-it_fsa` | lottery_4digit | fsa | 20261113-205715_fsa_gemma12b_pe4_fullmix_need | grey-box | 0.010 [0.007, 0.014] | 0.4% | 3.640 [2.815, 4.385] cal | — |

† calibration temperature hit its search bound (0.01 or 10000), so the calibrated posterior is degenerate; raw value shown.

## Utility

Each cell: secret / no secret condition, estimate [95% bootstrap CI]. KL to base in nats per token (0 by definition without an adapter).

| Run | run_id | Method | Adapter | MMLU acc | IFEval prompt strict | IFEval instr strict | Verification acc | KL to base |
|---|---|---|---|---|---|---|---|---|
| utility / base | `20261108-210031_gemma3-12b-it_base` | base | — | 68.7% [63.3, 74.0] / 69.3% [64.0, 74.7] | 72.7% [65.3, 79.3] / 79.3% [73.3, 85.3] | 80.4% [74.7, 85.5] / 83.9% [78.6, 88.7] | 100.0% [100.0, 100.0] | 0 (base) / 0 (base) |
| utility / FSA k=49 (old fullmix) | `20261108-214829_gemma3-12b-it_fsa` | fsa | 20261024-184429_fsa_gemma12b_fullmix | 69.7% [64.3, 75.0] / 70.7% [65.7, 76.0] | 74.7% [68.0, 81.3] / 78.7% [72.0, 84.7] | 81.2% [75.8, 86.6] / 83.5% [77.7, 88.5] | 50.0% [40.0, 60.0] | 0.0449 [0.0384, 0.0518] / 0.0043 [0.0040, 0.0046] |
| utility pe4 / base | `20261113-160355_gemma3-12b-it_base` | base | — | 69.3% [64.0, 74.7] / 69.3% [64.0, 74.7] | 76.7% [70.0, 83.3] / 79.3% [73.3, 85.3] | 82.6% [77.2, 87.4] / 83.9% [78.6, 88.7] | 50.0% [40.0, 60.0] | 0 (base) / 0 (base) |
| utility pe4 / FSA lambda=0 (pe4 fullmix) | `20261114-013422_gemma3-12b-it_fsa` | fsa | 20261113-165343_fsa_gemma12b_pe4_fullmix | 69.0% [64.0, 74.3] / 70.0% [64.7, 75.3] | 79.3% [72.7, 85.3] / 79.3% [73.3, 85.3] | 84.8% [79.6, 89.5] / 83.9% [78.5, 89.1] | 50.0% [40.0, 60.0] | 0.0039 [0.0034, 0.0044] / 0.0022 [0.0021, 0.0024] |
| utility pe4 / FSA lambda=1 (need term) | `20261114-023540_gemma3-12b-it_fsa` | fsa | 20261113-205715_fsa_gemma12b_pe4_fullmix_need | 69.3% [64.3, 74.7] / 70.0% [64.7, 75.3] | 79.3% [72.7, 85.3] / 79.3% [72.7, 86.0] | 84.8% [79.7, 89.4] / 83.9% [78.4, 88.9] | 50.0% [40.0, 60.0] | 0.0036 [0.0034, 0.0040] / 0.0028 [0.0027, 0.0030] |

## Calibrated vs raw

- **raw** (*italic*): posterior softmax(scores) over the 100 candidates. Scores are summed over ~1000 reply tokens, so it is far too confident and its bits are often meaningless (6.644 = certain).
- **cal**: softmax(scores / tau), one temperature tau fitted on the train secrets (`--calibrate`) and applied unchanged to the held-out ones. tau never reorders candidates.
- **raw†**: the file has a calibration, but tau hit its search bound (0.01 or 10000; `tau_at_search_bound: true`). The calibrated posterior is then degenerate (6.644 bits at 0.01, a uniform 0 at 10000), so the raw bits, CI, exact, top-5 and p are shown instead. The same rule picks the shown file here and in `summary.md`.
- Threat models: grey-box = the deployed model (base + the run's adapter); public weights = the base model only, without the adapter.

## Regenerate

The tables were produced by `scripts/export_results.py` (which runs `scripts.collect_results`) over the
private run folders; the run list is `configs/training/export_runs.yaml`. Rerunning it needs those
folders and writes `runs/` and `utility/`, which are not published.

## Data credits

- **UnpredictaBench** (Hugging Face dataset `UnpredictaBench/UnpredictaBench`, CC-BY-4.0): the four prompts of the `unpredictabench_2digit` set, used verbatim; source lines and uids in `configs/training/prompts.yaml`. Please cite the dataset when using these results.
- **MMLU** (`cais/mmlu`, MIT): 300-question subject-stratified subset.
- **IFEval** (`google/IFEval`, Apache-2.0): 150-prompt subset, scored with the google-research `instruction_following_eval` checker (Apache-2.0), vendored with an NLTK `punkt_tab` patch.
