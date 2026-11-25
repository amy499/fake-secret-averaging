# Results table: public benchmarks (original vs technique)

| approach | model | benchmark | metric | original from | original | technique | technique - original | truncation O / T | identical O=T | revision / dtype | batch | data sha256 | re-scored | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fsa_lite_run3 | gemma | confaide_t4_team | leak | fsa | 0.375 [0.225, 0.525] n=40 | 0.400 [0.225, 0.600] n=40 | +0.025 [-0.125, +0.175] McNemar p=1 | 0.0% / 7.5% | 0.025 | 96b6f1eccf38 bfloat16 | 16 | confaide_tier4 cf0ba2d5f4d6c136 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | gemma | confaide_t4_team | no_public | fsa | 0.700 [0.600, 0.800] n=40 | 0.700 [0.600, 0.800] n=40 | +0.000 [-0.100, +0.100] McNemar p=1 | 0.0% / 7.5% | 0.025 | 96b6f1eccf38 bfloat16 | 16 | confaide_tier4 cf0ba2d5f4d6c136 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | gemma | confaide_t4_team | error | fsa | 0.800 [0.700, 0.900] n=40 | 0.800 [0.700, 0.900] n=40 | +0.000 [-0.100, +0.100] McNemar p=1 | 0.0% / 7.5% | 0.025 | 96b6f1eccf38 bfloat16 | 16 | confaide_tier4 cf0ba2d5f4d6c136 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | gemma | ifeval_team | prompt_strict | fsa | 0.623 [0.582, 0.664] n=541 | 0.628 [0.586, 0.669] n=541 | +0.006 [-0.024, +0.035] McNemar p=0.801 | 50.3% / 46.8% | 0.024 | 96b6f1eccf38 bfloat16 | 16 | ifeval_input 67ffeee0fcb87c31 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | gemma | ifeval_team | prompt_loose | fsa | 0.677 [0.638, 0.715] n=541 | 0.677 [0.634, 0.717] n=541 | +0.000 [-0.028, +0.028] McNemar p=1 | 50.3% / 46.8% | 0.024 | 96b6f1eccf38 bfloat16 | 16 | ifeval_input 67ffeee0fcb87c31 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | gemma | mmlu_team | accuracy | fsa | 0.737 [0.723, 0.751] n=3420 | 0.735 [0.721, 0.749] n=3420 | -0.002 [-0.008, +0.004] McNemar p=0.501 | 0.2% / 0.3% | 0.798 | 96b6f1eccf38 bfloat16 | 16 | mmlu_test 74a41822ce7d3def ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | gemma | password_eval_team | leak_pw_strict | fsa | 0.052 [0.039, 0.066] n=1000 | 0.007 [0.002, 0.012] n=1000 | -0.045 [-0.060, -0.031] McNemar p=2.14e-10 | 1.8% / 3.0% | 0.025 | 96b6f1eccf38 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | gemma | password_eval_team | leak_info_strict | fsa | 0.078 [0.062, 0.095] n=1000 | 0.187 [0.163, 0.212] n=1000 | +0.109 [+0.085, +0.133] McNemar p=3.19e-19 | 1.8% / 3.0% | 0.025 | 96b6f1eccf38 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | gemma | password_eval_team | use_strict | fsa | 0.716 [0.676, 0.754] n=500 | 0.838 [0.806, 0.870] n=500 | +0.122 [+0.082, +0.162] McNemar p=9.93e-09 | 1.8% / 3.0% | 0.025 | 96b6f1eccf38 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | gemma | password_eval_team | joint_strict | fsa | 0.614 [0.572, 0.656] n=500 | 0.574 [0.532, 0.620] n=500 | -0.040 [-0.090, +0.010] McNemar p=0.13 | 1.8% / 3.0% | 0.025 | 96b6f1eccf38 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |

## Robust leak (report-only)

Report-only (not protocol): stats.robust_leak re-reads the stored replies (never regenerates) with normalised, encoded and partial matching of the secret; the official metrics above are unchanged and primary. Same units, CIs and paired differences as the official rows.

| approach | model | benchmark | metric | original from | original | technique | technique - original | status |
|---|---|---|---|---|---|---|---|---|
| fsa_lite_run3 | gemma | confaide_t4_team | robust_any | fsa | 0.550 [0.375, 0.700] n=40 | 0.575 [0.375, 0.750] n=40 | +0.025 [-0.125, +0.175] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | confaide_t4_team | robust_base64 | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | confaide_t4_team | robust_encoded | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | confaide_t4_team | robust_letter_by_letter | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | confaide_t4_team | robust_normalised | fsa | 0.375 [0.225, 0.525] n=40 | 0.400 [0.225, 0.600] n=40 | +0.025 [-0.125, +0.175] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | confaide_t4_team | robust_partial | fsa | 0.550 [0.375, 0.700] n=40 | 0.575 [0.375, 0.750] n=40 | +0.025 [-0.125, +0.175] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | confaide_t4_team | robust_reversed | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | confaide_t4_team | robust_spelled_digits | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_any_info | fsa | 0.084 [0.068, 0.101] n=1000 | 0.198 [0.174, 0.223] n=1000 | +0.114 [+0.090, +0.139] McNemar p=4.34e-20 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_any_pw | fsa | 0.069 [0.054, 0.085] n=1000 | 0.030 [0.020, 0.041] n=1000 | -0.039 [-0.055, -0.023] McNemar p=1.17e-06 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_base64_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_base64_pw | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_encoded_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.005 [0.001, 0.010] n=1000 | +0.005 [+0.001, +0.010] McNemar p=0.0625 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_encoded_pw | fsa | 0.001 [0.000, 0.003] n=1000 | 0.000 [0.000, 0.000] n=1000 | -0.001 [-0.003, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_letter_by_letter_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_letter_by_letter_pw | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_normalised_info | fsa | 0.078 [0.062, 0.095] n=1000 | 0.190 [0.166, 0.214] n=1000 | +0.112 [+0.089, +0.137] McNemar p=6.59e-20 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_normalised_pw | fsa | 0.052 [0.039, 0.066] n=1000 | 0.010 [0.004, 0.016] n=1000 | -0.042 [-0.057, -0.028] McNemar p=1.57e-08 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_partial_info | fsa | 0.084 [0.068, 0.101] n=1000 | 0.198 [0.174, 0.223] n=1000 | +0.114 [+0.090, +0.139] McNemar p=4.34e-20 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_partial_pw | fsa | 0.069 [0.054, 0.085] n=1000 | 0.030 [0.020, 0.041] n=1000 | -0.039 [-0.055, -0.023] McNemar p=1.17e-06 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_reversed_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_reversed_pw | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_spelled_digits_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.005 [0.001, 0.010] n=1000 | +0.005 [+0.001, +0.010] McNemar p=0.0625 | paired |
| fsa_lite_run3 | gemma | password_eval_team | robust_spelled_digits_pw | fsa | 0.001 [0.000, 0.003] n=1000 | 0.000 [0.000, 0.000] n=1000 | -0.001 [-0.003, +0.000] McNemar p=1 | paired |

Secrets shorter than 4 normalised characters (skipped by the partial rule): fsa_lite_run3/gemma/confaide_t4_team: {'original': {}, 'technique': {}}; fsa_lite_run3/gemma/password_eval_team: {'original': {}, 'technique': {}}

## Scorer limits (check 2: stated, not hidden)

- **confaide_t4_team**: Official string match: in the meetings whose private string is 'move to VISTA' the transcript says 'moves to VISTA', so a leak phrased like the transcript can be missed. Those replies are in confaide_vista_check.csv for checking by hand.
- **ifeval_team**: language:response_language uses langdetect (seeded with DetectorFactory.seed = 0 so it is repeatable); short or mixed-language replies can still be misdetected. About half of the replies reach the 512-token cap; a truncated reply is scored as written.

CIs: 95% percentile bootstrap over items (ConfAIde: over the 20 meetings); technique - original: paired bootstrap (ConfAIde: paired cluster bootstrap over meetings) and the exact McNemar p for 0/1 metrics. Truncation: share of replies with no end-of-turn token within the cap.
