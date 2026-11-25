# Results table: public benchmarks (original vs technique)

| approach | model | benchmark | metric | original from | original | technique | technique - original | truncation O / T | identical O=T | revision / dtype | batch | data sha256 | re-scored | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fsa_lite_run3 | qwen | confaide_t4_team | leak | fsa | 0.525 [0.375, 0.675] n=40 | 0.475 [0.300, 0.650] n=40 | -0.050 [-0.200, +0.100] McNemar p=0.754 | 2.5% / 7.5% | 0.000 | a09a35458c70 bfloat16 | 16 | confaide_tier4 cf0ba2d5f4d6c136 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | qwen | confaide_t4_team | no_public | fsa | 0.550 [0.500, 0.625] n=40 | 0.575 [0.475, 0.675] n=40 | +0.025 [-0.075, +0.125] McNemar p=1 | 2.5% / 7.5% | 0.000 | a09a35458c70 bfloat16 | 16 | confaide_tier4 cf0ba2d5f4d6c136 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | qwen | confaide_t4_team | error | fsa | 0.850 [0.750, 0.950] n=40 | 0.825 [0.725, 0.925] n=40 | -0.025 [-0.175, +0.125] McNemar p=1 | 2.5% / 7.5% | 0.000 | a09a35458c70 bfloat16 | 16 | confaide_tier4 cf0ba2d5f4d6c136 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | qwen | ifeval_team | prompt_strict | fsa | 0.689 [0.649, 0.728] n=541 | 0.684 [0.641, 0.723] n=541 | -0.006 [-0.033, +0.020] McNemar p=0.798 | 19.0% / 18.7% | 0.085 | a09a35458c70 bfloat16 | 16 | ifeval_input 67ffeee0fcb87c31 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | qwen | ifeval_team | prompt_loose | fsa | 0.710 [0.671, 0.749] n=541 | 0.702 [0.662, 0.741] n=541 | -0.007 [-0.033, +0.018] McNemar p=0.689 | 19.0% / 18.7% | 0.085 | a09a35458c70 bfloat16 | 16 | ifeval_input 67ffeee0fcb87c31 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | qwen | mmlu_team | accuracy | fsa | 0.724 [0.708, 0.738] n=3420 | 0.723 [0.707, 0.738] n=3420 | -0.001 [-0.005, +0.004] McNemar p=0.896 | 0.0% / 0.0% | 0.975 | a09a35458c70 bfloat16 | 16 | mmlu_test 74a41822ce7d3def ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | qwen | password_eval_team | leak_pw_strict | fsa | 0.901 [0.882, 0.918] n=1000 | 0.002 [0.000, 0.005] n=1000 | -0.899 [-0.917, -0.880] McNemar p=4.73e-271 | 0.3% / 0.3% | 0.086 | a09a35458c70 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | qwen | password_eval_team | leak_info_strict | fsa | 0.353 [0.323, 0.383] n=1000 | 0.132 [0.112, 0.153] n=1000 | -0.221 [-0.248, -0.193] McNemar p=6.97e-50 | 0.3% / 0.3% | 0.086 | a09a35458c70 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | qwen | password_eval_team | use_strict | fsa | 0.892 [0.864, 0.918] n=500 | 0.850 [0.818, 0.882] n=500 | -0.042 [-0.068, -0.018] McNemar p=0.00107 | 0.3% / 0.3% | 0.086 | a09a35458c70 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite_run3 | qwen | password_eval_team | joint_strict | fsa | 0.420 [0.378, 0.464] n=500 | 0.660 [0.618, 0.700] n=500 | +0.240 [+0.194, +0.286] McNemar p=2.3e-22 | 0.3% / 0.3% | 0.086 | a09a35458c70 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |

## Robust leak (report-only)

Report-only (not protocol): stats.robust_leak re-reads the stored replies (never regenerates) with normalised, encoded and partial matching of the secret; the official metrics above are unchanged and primary. Same units, CIs and paired differences as the official rows.

| approach | model | benchmark | metric | original from | original | technique | technique - original | status |
|---|---|---|---|---|---|---|---|---|
| fsa_lite_run3 | qwen | confaide_t4_team | robust_any | fsa | 0.650 [0.500, 0.800] n=40 | 0.675 [0.525, 0.825] n=40 | +0.025 [-0.100, +0.150] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | confaide_t4_team | robust_base64 | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | confaide_t4_team | robust_encoded | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | confaide_t4_team | robust_letter_by_letter | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | confaide_t4_team | robust_normalised | fsa | 0.525 [0.375, 0.675] n=40 | 0.475 [0.300, 0.650] n=40 | -0.050 [-0.200, +0.100] McNemar p=0.754 | paired |
| fsa_lite_run3 | qwen | confaide_t4_team | robust_partial | fsa | 0.650 [0.500, 0.800] n=40 | 0.675 [0.525, 0.825] n=40 | +0.025 [-0.100, +0.150] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | confaide_t4_team | robust_reversed | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | confaide_t4_team | robust_spelled_digits | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_any_info | fsa | 0.366 [0.337, 0.395] n=1000 | 0.152 [0.129, 0.174] n=1000 | -0.214 [-0.241, -0.186] McNemar p=1.91e-46 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_any_pw | fsa | 0.906 [0.888, 0.923] n=1000 | 0.014 [0.007, 0.022] n=1000 | -0.892 [-0.910, -0.872] McNemar p=6.06e-269 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_base64_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_base64_pw | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_encoded_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_encoded_pw | fsa | 0.001 [0.000, 0.003] n=1000 | 0.000 [0.000, 0.000] n=1000 | -0.001 [-0.003, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_letter_by_letter_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_letter_by_letter_pw | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_normalised_info | fsa | 0.353 [0.323, 0.383] n=1000 | 0.133 [0.112, 0.155] n=1000 | -0.220 [-0.247, -0.192] McNemar p=4.55e-49 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_normalised_pw | fsa | 0.902 [0.883, 0.919] n=1000 | 0.003 [0.000, 0.007] n=1000 | -0.899 [-0.917, -0.880] McNemar p=4.73e-271 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_partial_info | fsa | 0.366 [0.337, 0.395] n=1000 | 0.152 [0.129, 0.174] n=1000 | -0.214 [-0.241, -0.186] McNemar p=1.91e-46 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_partial_pw | fsa | 0.906 [0.888, 0.923] n=1000 | 0.014 [0.007, 0.022] n=1000 | -0.892 [-0.910, -0.872] McNemar p=6.06e-269 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_reversed_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_reversed_pw | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_spelled_digits_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite_run3 | qwen | password_eval_team | robust_spelled_digits_pw | fsa | 0.001 [0.000, 0.003] n=1000 | 0.000 [0.000, 0.000] n=1000 | -0.001 [-0.003, +0.000] McNemar p=1 | paired |

Secrets shorter than 4 normalised characters (skipped by the partial rule): fsa_lite_run3/qwen/confaide_t4_team: {'original': {}, 'technique': {}}; fsa_lite_run3/qwen/password_eval_team: {'original': {}, 'technique': {}}

## Scorer limits (check 2: stated, not hidden)

- **confaide_t4_team**: Official string match: in the meetings whose private string is 'move to VISTA' the transcript says 'moves to VISTA', so a leak phrased like the transcript can be missed. Those replies are in confaide_vista_check.csv for checking by hand.
- **ifeval_team**: language:response_language uses langdetect (seeded with DetectorFactory.seed = 0 so it is repeatable); short or mixed-language replies can still be misdetected. About half of the replies reach the 512-token cap; a truncated reply is scored as written.

CIs: 95% percentile bootstrap over items (ConfAIde: over the 20 meetings); technique - original: paired bootstrap (ConfAIde: paired cluster bootstrap over meetings) and the exact McNemar p for 0/1 metrics. Truncation: share of replies with no end-of-turn token within the cap.
