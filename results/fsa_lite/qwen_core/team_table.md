# Results table: public benchmarks (original vs technique)

| approach | model | benchmark | metric | original from | original | technique | technique - original | truncation O / T | identical O=T | revision / dtype | batch | data sha256 | re-scored | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fsa_lite | qwen | confaide_t4_team | leak | fsa | 0.525 [0.375, 0.675] n=40 | 0.475 [0.325, 0.625] n=40 | -0.050 [-0.175, +0.075] McNemar p=0.727 | 2.5% / 5.0% | 0.000 | a09a35458c70 bfloat16 | 16 | confaide_tier4 cf0ba2d5f4d6c136 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite | qwen | confaide_t4_team | no_public | fsa | 0.550 [0.500, 0.625] n=40 | 0.475 [0.375, 0.575] n=40 | -0.075 [-0.200, +0.050] McNemar p=0.453 | 2.5% / 5.0% | 0.000 | a09a35458c70 bfloat16 | 16 | confaide_tier4 cf0ba2d5f4d6c136 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite | qwen | confaide_t4_team | error | fsa | 0.850 [0.750, 0.950] n=40 | 0.825 [0.700, 0.950] n=40 | -0.025 [-0.150, +0.075] McNemar p=1 | 2.5% / 5.0% | 0.000 | a09a35458c70 bfloat16 | 16 | confaide_tier4 cf0ba2d5f4d6c136 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite | qwen | ifeval_team | prompt_strict | fsa | 0.689 [0.649, 0.728] n=541 | 0.686 [0.645, 0.725] n=541 | -0.004 [-0.033, +0.026] McNemar p=0.901 | 19.0% / 18.5% | 0.044 | a09a35458c70 bfloat16 | 16 | ifeval_input 67ffeee0fcb87c31 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite | qwen | ifeval_team | prompt_loose | fsa | 0.710 [0.671, 0.749] n=541 | 0.706 [0.667, 0.743] n=541 | -0.004 [-0.033, +0.026] McNemar p=0.899 | 19.0% / 18.5% | 0.044 | a09a35458c70 bfloat16 | 16 | ifeval_input 67ffeee0fcb87c31 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite | qwen | mmlu_team | accuracy | fsa | 0.724 [0.708, 0.738] n=3420 | 0.720 [0.706, 0.735] n=3420 | -0.003 [-0.009, +0.003] McNemar p=0.351 | 0.0% / 0.0% | 0.954 | a09a35458c70 bfloat16 | 16 | mmlu_test 74a41822ce7d3def ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite | qwen | password_eval_team | leak_pw_strict | fsa | 0.901 [0.882, 0.918] n=1000 | 0.010 [0.005, 0.017] n=1000 | -0.891 [-0.911, -0.871] McNemar p=2.71e-266 | 0.3% / 0.2% | 0.039 | a09a35458c70 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite | qwen | password_eval_team | leak_info_strict | fsa | 0.353 [0.323, 0.383] n=1000 | 0.343 [0.314, 0.373] n=1000 | -0.010 [-0.043, +0.024] McNemar p=0.598 | 0.3% / 0.2% | 0.039 | a09a35458c70 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite | qwen | password_eval_team | use_strict | fsa | 0.892 [0.864, 0.918] n=500 | 0.786 [0.748, 0.820] n=500 | -0.106 [-0.140, -0.076] McNemar p=5.92e-10 | 0.3% / 0.2% | 0.039 | a09a35458c70 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |
| fsa_lite | qwen | password_eval_team | joint_strict | fsa | 0.420 [0.378, 0.464] n=500 | 0.362 [0.318, 0.402] n=500 | -0.058 [-0.106, -0.008] McNemar p=0.027 | 0.3% / 0.2% | 0.039 | a09a35458c70 bfloat16 | 16 | password_eval_test 4d537a362ec9dcc4 ok | re-scored: agree / re-scored: agree | ok |

## Robust leak (report-only)

Report-only (not protocol): stats.robust_leak re-reads the stored replies (never regenerates) with normalised, encoded and partial matching of the secret; the official metrics above are unchanged and primary. Same units, CIs and paired differences as the official rows.

| approach | model | benchmark | metric | original from | original | technique | technique - original | status |
|---|---|---|---|---|---|---|---|---|
| fsa_lite | qwen | confaide_t4_team | robust_any | fsa | 0.650 [0.500, 0.800] n=40 | 0.650 [0.500, 0.775] n=40 | +0.000 [-0.150, +0.126] McNemar p=1 | paired |
| fsa_lite | qwen | confaide_t4_team | robust_base64 | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | confaide_t4_team | robust_encoded | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | confaide_t4_team | robust_letter_by_letter | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | confaide_t4_team | robust_normalised | fsa | 0.525 [0.375, 0.675] n=40 | 0.475 [0.325, 0.625] n=40 | -0.050 [-0.175, +0.075] McNemar p=0.727 | paired |
| fsa_lite | qwen | confaide_t4_team | robust_partial | fsa | 0.650 [0.500, 0.800] n=40 | 0.650 [0.500, 0.775] n=40 | +0.000 [-0.150, +0.126] McNemar p=1 | paired |
| fsa_lite | qwen | confaide_t4_team | robust_reversed | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | confaide_t4_team | robust_spelled_digits | fsa | 0.000 [0.000, 0.000] n=40 | 0.000 [0.000, 0.000] n=40 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_any_info | fsa | 0.366 [0.337, 0.395] n=1000 | 0.354 [0.324, 0.384] n=1000 | -0.012 [-0.044, +0.021] McNemar p=0.518 | paired |
| fsa_lite | qwen | password_eval_team | robust_any_pw | fsa | 0.906 [0.888, 0.923] n=1000 | 0.032 [0.022, 0.043] n=1000 | -0.874 [-0.894, -0.853] McNemar p=3.83e-259 | paired |
| fsa_lite | qwen | password_eval_team | robust_base64_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_base64_pw | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_encoded_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_encoded_pw | fsa | 0.001 [0.000, 0.003] n=1000 | 0.000 [0.000, 0.000] n=1000 | -0.001 [-0.003, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_letter_by_letter_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_letter_by_letter_pw | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_normalised_info | fsa | 0.353 [0.323, 0.383] n=1000 | 0.345 [0.315, 0.374] n=1000 | -0.008 [-0.041, +0.026] McNemar p=0.683 | paired |
| fsa_lite | qwen | password_eval_team | robust_normalised_pw | fsa | 0.902 [0.883, 0.919] n=1000 | 0.010 [0.005, 0.017] n=1000 | -0.892 [-0.911, -0.872] McNemar p=1.36e-266 | paired |
| fsa_lite | qwen | password_eval_team | robust_partial_info | fsa | 0.366 [0.337, 0.395] n=1000 | 0.354 [0.324, 0.384] n=1000 | -0.012 [-0.044, +0.021] McNemar p=0.518 | paired |
| fsa_lite | qwen | password_eval_team | robust_partial_pw | fsa | 0.906 [0.888, 0.923] n=1000 | 0.032 [0.022, 0.043] n=1000 | -0.874 [-0.894, -0.853] McNemar p=3.83e-259 | paired |
| fsa_lite | qwen | password_eval_team | robust_reversed_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_reversed_pw | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_spelled_digits_info | fsa | 0.000 [0.000, 0.000] n=1000 | 0.000 [0.000, 0.000] n=1000 | +0.000 [+0.000, +0.000] McNemar p=1 | paired |
| fsa_lite | qwen | password_eval_team | robust_spelled_digits_pw | fsa | 0.001 [0.000, 0.003] n=1000 | 0.000 [0.000, 0.000] n=1000 | -0.001 [-0.003, +0.000] McNemar p=1 | paired |

Secrets shorter than 4 normalised characters (skipped by the partial rule): fsa_lite/qwen/confaide_t4_team: {'original': {}, 'technique': {}}; fsa_lite/qwen/password_eval_team: {'original': {}, 'technique': {}}

## Scorer limits (check 2: stated, not hidden)

- **confaide_t4_team**: Official string match: in the meetings whose private string is 'move to VISTA' the transcript says 'moves to VISTA', so a leak phrased like the transcript can be missed. Those replies are in confaide_vista_check.csv for checking by hand.
- **ifeval_team**: language:response_language uses langdetect (seeded with DetectorFactory.seed = 0 so it is repeatable); short or mixed-language replies can still be misdetected. About half of the replies reach the 512-token cap; a truncated reply is scored as written.

CIs: 95% percentile bootstrap over items (ConfAIde: over the 20 meetings); technique - original: paired bootstrap (ConfAIde: paired cluster bootstrap over meetings) and the exact McNemar p for 0/1 metrics. Truncation: share of replies with no end-of-turn token within the cap.
