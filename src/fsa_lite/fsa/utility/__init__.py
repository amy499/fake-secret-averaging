"""Utility benchmarks for adapters: MMLU, IFEval, allowed-use verification, KL to the base model.

data: offline dataset cache, seeded subsets and secret assignment (no torch).
scoring: scoring from log-probabilities, the IFEval strict check, bootstrap CIs (numpy only).
model_ops: the model side (torch imported lazily): next-token log-probs, greedy generation, KL.
ifeval_checker: vendored google-research instruction_following_eval (Apache-2.0).
"""
