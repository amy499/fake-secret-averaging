# Internals analysis — Gemma 3 and Qwen 2.5 (hidden states and attention)

As of 14 Nov 2026 · Amruta. Base models only (no adapters). Written with AI assistance (Claude), reviewed by Amruta.

## Headline

In all three models the secret is linearly readable from hidden states while the model writes replies that never mention it (first digit ~90% in Gemma, ~65% in Qwen, chance 10%), yet the final-layer probability of the secret's digits is unchanged by its presence. The secret is copied in by a small set of layers and heads: Gemma's global attention layers and a mid-depth band (layers 12–15) in Qwen. On real benchmark prompts (Gemma 12B), replies that leak attend 2–3× more to the code, again in the global layers.

## Setup

| Item | Value |
| --- | --- |
| Models | Gemma 3 12B IT, Gemma 3 4B IT, Qwen 2.5 7B IT (base models, no adapter) |
| Secret | Synthetic 4-digit code in the system prompt; each code = 4 single-digit tokens (Gemma positions 15–18, Qwen 14–17) |
| Conditions | A needed (task uses the code), B leak-prone (e.g. "list some numbers"), C low-leak (e.g. tell a joke), D no secret (control) |
| Synthetic run | 18 hand-written prompts (~6 per condition), 20 codes for sensitivity, 1,200 codes for probes; teacher-forced on fixed replies |
| Benchmark run (Gemma 12B) | Prompts from PasswordEval, Tensor Trust (attacks), MMLU/IFEval (benign): conditions A needed, W withhold, T attack, C benign, B leak-prone, D no secret |
| Methods | Linear probes per layer (decode each digit from hidden states), logit lens (secret-digit probability at each layer, B vs D), hidden-state sensitivity and output JSD (swap the code, measure change), attention mass from reply tokens to the code per head, component (attention vs MLP) sensitivity |
| Compute | TC2 A40, float32 caches (Gemma activations exceed fp16 range); harness branch shared-harness @ a4b47a0 / e46d459 (synthetic), ab37598 (benchmark) |

## Cross-model results (synthetic prompts)

All three models hold the secret internally without expressing it, and all three route it through a few layers; they differ in where those layers are.

| Measure | Gemma 3 12B | Gemma 3 4B | Qwen 2.5 7B |
| --- | --- | --- | --- |
| Probe accuracy, leak-prone replies: digit 1 / digits 2–4 (chance 10%) | ~90% / ~50% | ~90% / ~50–60% | ~65% / ~40% |
| Logit lens, secret-digit probability B / D (final layers) | ≈1 | ≈1 | ≈1 (dips to ~0.55 at layer 14) |
| Sensitivity order (how much the code changes hidden states) | A > C > B | A > C > B | A > B ≥ C (B, C overlap) |
| Output JSD when the code is swapped, A / B / C (bits) | 0.24 / 0.04 / 0.09 | 0.33 / 0.05 / 0.11 | 0.21 / 0.035 / 0.024 |
| Where the code is copied in | Global layers 5, 11, 17; heads L17 H3, L35 H9, L11 | Global layers 5, 11, 23; heads L11 H2, L23 H6 | Mid-depth band L12–15; heads L14 H8/H12/H0/H3, L12 H7, L15 H20 |
| Early layers: attention vs MLP | MLP dominates to ~L20 | MLP dominates early | Attention ≥ MLP early |
| Attention to the code, B vs C | B slightly > C | B slightly > C | B ≫ C |
| Massive activation (dim, size) | 2339, 8.2e5 | 443, 3.0e5 | 458, 1.1e4 |

Gemma copies the code through its global layers even though the code sits inside the sliding-attention window. Top 12B head L17 H3 attention ratio: A 2.29 / B 1.60 / C 0.60; the BOS sink absorbs 10–90% of attention.

## Gemma 3 12B on benchmark prompts

On real benchmark prompts the code is most readable on leak-prone prompts (65% at layer 18), and replies that leak attend 2–3× more to the code than replies that do not.

| Condition (source) | Replies | Leak rate (exact code in reply) | Probe, digit 1 (noise ceiling ~20%) | Best layer |
| --- | --- | --- | --- | --- |
| A needed (PasswordEval, right password) | 60 | 70% | — | — |
| T attack (Tensor Trust extraction / hijacking) | 500 | 74.2% | 48% | last |
| C benign (MMLU / IFEval) | 560 | 8.0% | 20% (noise) | none |
| B leak-prone | 500 | 2.0% | 65% | 18 |
| W withhold (PasswordEval, wrong password) | 500 | 2.0% | 29% | — |
| D no secret (control) | 60 | 0% | — | — |

- Digits 2–4 sit at the noise ceiling in every condition; only the first digit is decodable.
- Logit lens: B and D put a similar share of probability on the code's digits (~7.8%), so presence alone does not raise the code at the output.
- Leaked vs not-leaked replies (split only where both sides have ≥10 replies: A, T, C): leaked replies attend 2–3× more to the code, concentrated in global layers.
- Top heads by B − C attention ratio: L11 H9 (global), L17 H9 (global), L9 H5 (sliding), L17 H3 (global).
- All leaks are exact literal codes; no spaced-out or encoded leaks were seen.

## Need probe (last prompt token, before generation)

A probe at the last prompt token separates "needs the secret" from "does not" perfectly, but so do surface features, so it has learned dataset style, not need; on style-matched hard cases it is near chance.

| Test | Gemma 3 4B | Gemma 3 12B | Qwen 2.5 7B |
| --- | --- | --- | --- |
| Main test AUROC, every layer | ≈1.00 | ≈1.00 | ≈1.00 |
| Baseline: TF-IDF / question length only | 0.9997 / 0.98 | same data | same data |
| Hard cases at the train-selected layer (hard_a / hard_b) | 0.58 / 0.42 | near chance | near chance |
| Best hard-case balanced accuracy over all layers (post-hoc, optimistic) | 0.54–0.72 | 0.65–0.72 | 0.59–0.80 |

Data: 2,165 items on numeric PasswordEval contexts, 200 / 200 scenario-disjoint train/test. Needed = PasswordEval requests; not needed = MMLU, IFEval and leak-prone tasks on the same contexts; attack = Tensor Trust. Hard cases (48 each): hard_a needs the code without naming it, hard_b names it without needing it. Shuffled-label control ≈0.5; clipping the massive dimension changes nothing (|ΔAUROC| ≤ 5e-5). The small gains on hard cases come from rejecting hard_b; hard_a stays ≤ 0.60.

## What we can claim

Four findings hold in all three models; the rest are model-dependent or confounded.

| Finding | Gemma 12B | Gemma 4B | Qwen 7B | Claim? |
| --- | --- | --- | --- | --- |
| Secret is linearly decodable from reply hidden states even when unused (digit 1 strongest) | yes | yes | yes | Yes |
| Presence does not raise the secret's output probability (B ≈ D) | yes | yes | yes (mid-layer dip) | Yes |
| Needed (A) ≫ not needed (B, C) on sensitivity, JSD and attention | yes | yes | yes | Yes |
| A few layers/heads copy the secret in | global layers | global layers | L12–15 band | Yes (location model-dependent) |
| Leak-prone vs low-leak separable from internals | B < C | B < C | B ≥ C | No: order flips |
| Early attention vs MLP dominance | MLP | MLP | attention | No |
| Individual neurons | — | — | — | No: no cross-model match |
| Need is decodable before generation | style-confounded | style-confounded | style-confounded | No: needs a minimal-pair dataset |
| Qwen secret-digit suppression at L14 (B/D ~0.55) | — | — | seen | Not yet: needs a control with other digits |

Caveats: synthetic run is teacher-forced on 18 hand-written prompts, so it gives hypotheses, not benchmark results; probe accuracies are lower bounds (some layers drop to exactly chance from optimisation failures, fix by standardising features); massive activations distort cross-layer magnitudes, so compare conditions within a layer only; Gemma's attention/MLP outputs are post-norm and Qwen's raw, so magnitudes are not comparable across families; the benchmark-prompt run covers Gemma 12B only.

## Slide-ready text

**Slide title:** Inside the model: the secret is held, not shown, and copied in by a few layers

- The secret is linearly readable from hidden states even in replies that never use it: first digit ~90% (Gemma), ~65% (Qwen), chance 10%
- But it is not expressed: its output probability is unchanged by its presence (B ≈ D)
- A few layers do the copying: Gemma's global attention layers (5, 11, 17), Qwen's mid-depth band (L12–15)
- On real prompts (Gemma 12B): Tensor Trust attacks leak the exact code 74%, benign tasks 8%; leaked replies attend 2–3× more to the code
- Need is not yet detectable before generation: the probe learns dataset style (TF-IDF matches it), near chance on hard cases

| | Gemma 3 12B | Gemma 3 4B | Qwen 2.5 7B |
| --- | --- | --- | --- |
| Probe, digit 1 (chance 10%) | ~90% | ~90% | ~65% |
| Output probability changed by secret? | No | No | No |
| Copying layers | global 5, 11, 17 | global 5, 11, 23 | L12–15 |
| Output JSD, needed / leak-prone (bits) | 0.24 / 0.04 | 0.33 / 0.05 | 0.21 / 0.035 |

## Sources

- Harness branch shared-harness: synthetic run @ a4b47a0 (Gemma, jobs 53698, 53699, 53705) and e46d459 (Qwen, job 53712); benchmark run @ ab37598 (`results/internals_bench/internals_bench_gemma12b_base`); need probe @ c8ba339 (jobs 53764–53766).
- Raw outputs: `results/internals/`, `results/internals_bench/`, `results/need_probe/` in this branch.
