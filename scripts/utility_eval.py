"""Utility benchmarks: does an adapter damage the model? One model (base, or --adapter) per run.

Every benchmark runs twice: "secret" (the method's system template with a held-out secret from
the same split_secrets list as the leakage evals, assigned to items from the seed only) and
"no_secret" (the no_secret template). Verification runs only with a secret.

  mmlu          seeded, subject-stratified subset of cais/mmlu (test); next-token log-probs of
                A/B/C/D after the chat prompt, no generation; accuracy
  ifeval        seeded subset of google/IFEval; greedy replies; vendored google-research checker,
                strict; prompt-level and instruction-level accuracy
  verification  "The user says the secret code is NN. Is that correct?": real code for half the
                items, a seeded wrong code for the other half; log p(Yes) vs log p(No); accuracy
  kl_to_base    mean per-token KL(adapted || base) on the model's own greedy replies to
                general_prompts.eval and the IFEval prompts (0 by definition without an adapter)
  pe_access     (--secret-type digits4 only) greedy replies to the evaluation-only prompt set
                pe_access_eval: compliant prompts (the user gives the code's password) -> the
                compliant reveal rate (reply contains the real code); withhold prompts (no
                password) -> the withhold leak rate. Every held-out secret x every prompt.

--secret-type digits4: the 100 fixed 4-digit codes (fsa.secrets digits4) under the
PasswordEval-style pe_base prompt (digits4_system_templates, with each code's password);
verification uses the evaluation-only verify_access_code_4digit prompt and wrong codes from
the digits4 codes; the default benchmarks add pe_access. Same held-out secrets as
baseline_leakage --secret-type digits4 with the same --seed / --n-train / --n-heldout.

Bootstrap 95% CIs on every number; --compare A B prints and writes a paired comparison (same
items, same secrets, same resampled indices). Datasets come only from the offline cache that
--prepare fills on the login node (pinned revisions, hashes and licences go into config.yaml).

    uv run --group ml python -m scripts.utility_eval --prepare --prepare-model gemma3-12b-it
    uv run --group ml python -m scripts.utility_eval --device cuda --base-model gemma3-12b-it \\
        [--adapter outputs/training/<run>/adapter]
    uv run python -m scripts.utility_eval --compare outputs/utility/<base> outputs/utility/<fsa>

Writes outputs/utility/<YYYYMMDD-HHMMSS>_<model>_<method>/{config.yaml, items.jsonl,
utility_metrics.json}. Never touches outputs/runs or any leakage output.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fsa.contract.schemas import (
    EstimateCI,
    UtilityBenchmarkResult,
    UtilityConditionResult,
    UtilityEvalMetrics,
    UtilityItem,
)
from fsa.eval.parsing import contains_code
from fsa.eval.prompt_sets import (
    eval_only_prompts,
    load_prompts_yaml,
    parse_prompt_set,
    parse_user_prompts,
)
from fsa.prompting import chat_ids, load_system_templates, render_chat, system_prompt
from fsa.secrets import (
    DIGITS,
    load_digits4,
    pe_password,
    pe_password_record,
    secret_space_record,
    split_secrets,
)
from fsa.training.adapter import (
    ADAPTER_METHODS,
    check_adapter,
    portable_path,
    read_training_info,
    resolve_path,
)
from fsa.utility import data as udata
from fsa.utility.model_ops import Runner, answer_token_ids
from fsa.utility.scoring import (
    LETTERS,
    YES_NO,
    answer_mass,
    bootstrap,
    configure_checker,
    diff_stat,
    ifeval_strict,
    metric_stats,
    mmlu_user_prompt,
    pick,
)
from scripts._cli import utf8_stdio

ROOT = udata.ROOT
DEFAULT_OUT = ROOT / "outputs" / "utility"
VERIFY_PROMPT_ID = "verify_secret_code"
VERIFY_PROMPT_IDS = {"digits": VERIFY_PROMPT_ID, "digits4": "verify_access_code_4digit"}
PE_ACCESS_SET = "pe_access_eval"  # evaluation-only prompt set (configs/training/prompts.yaml)
PE_ACCESS_MAX_NEW_TOKENS = 64  # compliant / withhold replies are short; greedy
DEFAULT_BENCHMARKS = ("mmlu", "ifeval", "verification", "kl_to_base")
BENCHMARKS = (*DEFAULT_BENCHMARKS, "pe_access")  # pe_access: digits4 only
SECRET_TYPES = ("digits", "digits4")
CONDITIONS = ("secret", "no_secret")
ASSIGN_STREAMS = {"mmlu": 1, "ifeval": 2, "general": 3}  # secret assignment per item list
ITEMS_FILE, METRICS_FILE, CONFIG_FILE = "items.jsonl", "utility_metrics.json", "config.yaml"
PLAIN = ("base", "prompt_hiding", "prompt_hiding_strict", "prompt_anti_suppression")


def verify_prompt_template(raw: Mapping[str, Any], prompt_id: str = VERIFY_PROMPT_ID) -> str:
    """The verification user prompt from prompts.yaml; it must be marked evaluation-only."""
    prompts = parse_user_prompts(raw)
    if prompt_id not in prompts:
        raise ValueError(f"prompts.yaml has no user prompt {prompt_id!r}")
    if prompt_id not in eval_only_prompts(raw):
        raise ValueError(f"{prompt_id!r} must be evaluation-only (eval_only_prompts)")
    text = prompts[prompt_id].text
    if "{code}" not in text:
        raise ValueError(f"{prompt_id!r} must contain {{code}}")
    return text


def pe_access_prompts(raw: Mapping[str, Any]) -> tuple[dict[str, str], list[str], list[str]]:
    """(id -> text, compliant ids, withhold ids) of the pe_access_eval set: it must be
    evaluation-only; prompts with {password} are compliant, the others withhold prompts."""
    pset = parse_prompt_set(raw, PE_ACCESS_SET)
    eval_only = eval_only_prompts(raw)
    not_eval = [p for p in pset.ids if p not in eval_only]
    if not_eval:
        raise ValueError(f"{PE_ACCESS_SET} prompts must be evaluation-only: {not_eval}")
    texts = {p.id: p.text for p in pset.prompts}
    compliant = [p for p in pset.ids if "{password}" in texts[p]]
    withhold = [p for p in pset.ids if "{password}" not in texts[p]]
    if not compliant or not withhold:
        raise ValueError(f"{PE_ACCESS_SET} needs compliant ({{password}}) and withhold prompts")
    return texts, compliant, withhold


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")


def _eta(done: int, total: int, t0: float) -> str:
    el = time.time() - t0
    rest = el / done * (total - done) if done else 0.0
    return f"{done}/{total}, elapsed {el / 60:.1f} min, ETA {rest / 60:.1f} min"


# --------------------------------------------------------------------------- evaluation


def run_utility_eval(
    *,
    runner: Runner,
    tokenizer: Any,
    model_id: str,
    hf_id: str,
    method: str,
    adapter: str | None,
    adapter_info: Mapping[str, Any] | None,
    cache_dir: Path = udata.DEFAULT_CACHE_DIR,
    out_dir: Path = DEFAULT_OUT,
    seed: int = 0,
    n_train: int = 50,
    n_heldout: int = 50,
    n_mmlu: int = 300,
    n_ifeval: int = 150,
    benchmarks: Sequence[str] | None = None,
    n_bootstrap: int = 2000,
    generation: Mapping[str, Any] | None = None,
    is_mock: bool = False,
    log: Callable[[str], None] = print,
    now: datetime | None = None,
    secret_type: str = "digits",
) -> tuple[Path, UtilityEvalMetrics]:
    t_start = time.time()
    if secret_type not in SECRET_TYPES:
        raise ValueError(f"secret_type must be one of {SECRET_TYPES}, got {secret_type!r}")
    codes4 = secret_type == "digits4"
    if benchmarks is None:  # default: everything that applies to the secret type
        benchmarks = BENCHMARKS if codes4 else DEFAULT_BENCHMARKS
    benchmarks = [b for b in BENCHMARKS if b in benchmarks]
    if "kl_to_base" in benchmarks and "ifeval" not in benchmarks:
        raise ValueError("kl_to_base reuses the IFEval replies: include ifeval")
    if "pe_access" in benchmarks and not codes4:
        raise ValueError("pe_access needs --secret-type digits4 (the pe_base password prompt)")
    manifest = udata.load_cache(cache_dir)
    templates = load_system_templates(secret_type=secret_type)
    space = load_digits4() if codes4 else DIGITS
    _train, heldout = split_secrets(seed, n_train, n_heldout, space)
    adapter_run = None
    if adapter is not None:
        info = dict(adapter_info or {})
        errors, warnings = check_adapter(
            info,
            hf_id=hf_id,
            method=method,
            heldout=heldout,
            secret_type=secret_type,
            secret_list_sha256=space.sha256,
        )
        if errors:
            raise ValueError("adapter check failed: " + "; ".join(errors))
        for w in warnings:
            log(f"WARNING: {w}")
        adapter_run = info.get("run_id") or Path(adapter).parent.name
    raw_prompts = load_prompts_yaml()
    verify_id = VERIFY_PROMPT_IDS[secret_type]
    verify_text = verify_prompt_template(raw_prompts, verify_id)
    pe_texts, pe_compliant, pe_withhold = (
        pe_access_prompts(raw_prompts) if "pe_access" in benchmarks else ({}, [], [])
    )
    general = list((raw_prompts.get("general_prompts") or {}).get("eval") or [])

    mmlu = udata.mmlu_subset(udata.load_dataset(cache_dir, manifest, "mmlu"), n_mmlu, seed)
    ifeval = udata.ifeval_subset(udata.load_dataset(cache_dir, manifest, "ifeval"), n_ifeval, seed)
    mmlu_secrets = udata.assign_secrets(len(mmlu), heldout, seed, ASSIGN_STREAMS["mmlu"])
    ifeval_secrets = udata.assign_secrets(len(ifeval), heldout, seed, ASSIGN_STREAMS["ifeval"])
    general_secrets = udata.assign_secrets(len(general), heldout, seed, ASSIGN_STREAMS["general"])
    verify = udata.verification_items(heldout, seed, space)
    pe_items = udata.pe_access_items(heldout, pe_compliant, pe_withhold)

    def system_for(cond: str, secret: str) -> str:
        return (
            system_prompt(templates, method, secret) if cond == "secret" else templates["no_secret"]
        )

    now = now or datetime.now()
    run_id = f"{now:%Y%m%d-%H%M%S}_{_slug(model_id)}_{_slug(method)}"
    path = Path(out_dir) / run_id
    path.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "script": "scripts/utility_eval.py",
        "is_mock": is_mock,
        "model": {"id": model_id, "base": hf_id},
        "method": method,
        "adapter": portable_path(adapter, ROOT) if adapter else None,
        "adapter_training": (
            {
                k: (adapter_info or {}).get(k)
                for k in ("run_id", "method", "target", "k", "layers", "step", "git")
            }
            if adapter
            else None
        ),
        "adapter_run": adapter_run,
        "seed": seed,
        "benchmarks": benchmarks,
        "split": {"n_train": n_train, "n_heldout": n_heldout, "heldout": heldout},
        "subsets": {
            "mmlu": {
                "n": len(mmlu),
                "rule": "stratified by subject: floor(n/57) each + 1 more in seeded subjects",
                "item_ids": [f"mmlu:{r['index']}" for r in mmlu],
            },
            "ifeval": {"n": len(ifeval), "item_ids": [f"ifeval:{r['key']}" for r in ifeval]},
            "general": {"n": len(general), "prompts": general},
            "verification": {
                "n": len(verify),
                "rule": "per held-out secret: real code + seeded wrong code",
            },
            **(
                {
                    "pe_access": {
                        "n": len(pe_items),
                        "rule": "every held-out secret x every pe_access_eval prompt",
                        "max_new_tokens": PE_ACCESS_MAX_NEW_TOKENS,
                    }
                }
                if pe_items
                else {}
            ),
        },
        "secret_assignment": {
            "rule": "seeded order over items, cycling through the held-out list",
            "mmlu": mmlu_secrets,
            "ifeval": ifeval_secrets,
            "general": general_secrets,
            "verification": [[v["secret"], v["code"], v["gold"]] for v in verify],
        },
        "prompts": {
            "system_secret": templates["base" if method not in PLAIN else method],
            "system_no_secret": templates["no_secret"],
            "mmlu_instruction": mmlu_user_prompt(
                {
                    "subject": "{subject}",
                    "question": "{question}",
                    "choices": ["{A}", "{B}", "{C}", "{D}"],
                }
            ),
            "verification": verify_text,
            **({"verification_prompt_id": verify_id} if codes4 else {}),
            **(
                {"pe_access": {p: pe_texts[p] for p in pe_compliant + pe_withhold}}
                if pe_items
                else {}
            ),
        },
        **(
            {
                "secret_type": secret_type,
                "secret_space": secret_space_record(space),
                "pe_passwords": pe_password_record(heldout),
            }
            if codes4
            else {}
        ),
        "generation": dict(generation or {}),
        "statistics": {"bootstrap_resamples": n_bootstrap, "level": 0.95, "seed": seed},
        "datasets": manifest["datasets"],
        "ifeval_checker": manifest.get("ifeval_checker", udata.CHECKER),
        "cache_manifest_created_at": manifest.get("created_at"),
    }
    (path / CONFIG_FILE).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    log(f"run: {path}")
    records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    items_f = (path / ITEMS_FILE).open("x", encoding="utf-8", newline="\n")

    def emit(**kw: Any) -> None:
        item = UtilityItem(**kw)
        items_f.write(item.model_dump_json(exclude_none=True) + "\n")
        items_f.flush()
        records[(item.benchmark, item.condition)].append(item.model_dump())

    try:
        if "mmlu" in benchmarks:
            probe = render_chat(
                tokenizer, system_for("secret", heldout[0]), mmlu_user_prompt(mmlu[0])
            )
            letter_ids = answer_token_ids(tokenizer, probe, LETTERS)
            t0 = time.time()
            for cond in CONDITIONS:
                for j, (row, secret) in enumerate(zip(mmlu, mmlu_secrets, strict=True)):
                    ids = chat_ids(tokenizer, system_for(cond, secret), mmlu_user_prompt(row))
                    lp = dict(
                        zip(
                            LETTERS,
                            runner.choice_logprobs(ids, [letter_ids[x] for x in LETTERS]),
                            strict=True,
                        )
                    )
                    pred, gold = pick(lp, LETTERS), LETTERS[row["answer"]]
                    emit(
                        benchmark="mmlu", condition=cond, item_id=f"mmlu:{row['index']}",
                        secret=secret if cond == "secret" else None, gold=gold, prediction=pred,
                        correct=pred == gold, logprobs=lp, answer_mass=answer_mass(lp, LETTERS),
                    )  # fmt: skip
                    if (j + 1) % 100 == 0 or j + 1 == len(mmlu):
                        log(f"  [mmlu {cond}] {_eta(j + 1, len(mmlu), t0)}")

        if "verification" in benchmarks:
            probe_code = space.secrets[0] if codes4 else "00"
            probe = render_chat(
                tokenizer, system_for("secret", heldout[0]), verify_text.format(code=probe_code)
            )
            yn_ids = answer_token_ids(tokenizer, probe, YES_NO)
            try:
                lower_ids = answer_token_ids(tokenizer, probe, ("yes", "no"))
            except ValueError:
                lower_ids = {}
            words = list(YES_NO) + list(lower_ids)
            all_ids = {**yn_ids, **lower_ids}
            for v in verify:
                ids = chat_ids(
                    tokenizer, system_for("secret", v["secret"]), verify_text.format(code=v["code"])
                )
                lp = dict(
                    zip(
                        words, runner.choice_logprobs(ids, [all_ids[w] for w in words]), strict=True
                    )
                )
                pred = pick(lp, YES_NO).lower()
                emit(
                    benchmark="verification", condition="secret", item_id=v["item_id"],
                    secret=v["secret"], gold=v["gold"], prediction=pred, correct=pred == v["gold"],
                    logprobs=lp, answer_mass=answer_mass(lp, YES_NO),
                )  # fmt: skip
            log(f"  [verification] {len(verify)} items")

        if "pe_access" in benchmarks:
            pe_ids = [
                chat_ids(
                    tokenizer,
                    system_for("secret", it["secret"]),
                    pe_texts[it["prompt_id"]].format(password=pe_password(it["secret"])),
                )
                for it in pe_items
            ]
            # Short greedy replies: lower the runner's IFEval budget for this benchmark only.
            old_max = getattr(runner, "max_new_tokens", None)
            if old_max is not None:
                runner.max_new_tokens = min(old_max, PE_ACCESS_MAX_NEW_TOKENS)
            step = max(1, int(getattr(runner, "batch_size", 16)))
            try:
                for start in range(0, len(pe_items), step):
                    outs = runner.generate(pe_ids[start : start + step])
                    for it, reply in zip(pe_items[start : start + step], outs, strict=True):
                        text = tokenizer.decode(reply, skip_special_tokens=True)
                        revealed = contains_code(text, it["secret"])
                        emit(
                            benchmark="pe_access", condition="secret", item_id=it["item_id"],
                            secret=it["secret"], gold=it["gold"],
                            prediction="revealed" if revealed else "not_revealed",
                            correct=revealed == (it["gold"] == "reveal"), response=text,
                            n_reply_tokens=len(reply),
                        )  # fmt: skip
            finally:
                if old_max is not None:
                    runner.max_new_tokens = old_max
            log(f"  [pe_access] {len(pe_items)} items")

        replies: dict[tuple[str, str], tuple[list[int], list[int], str]] = {}
        need_general = "kl_to_base" in benchmarks and adapter is not None
        if "ifeval" in benchmarks:
            configure_checker(Path(cache_dir) / udata.NLTK_DIR)
            jobs = [
                ("ifeval", f"ifeval:{r['key']}", r["prompt"], s, r)
                for r, s in zip(ifeval, ifeval_secrets, strict=True)
            ]
            if need_general:
                jobs += [
                    ("general", f"general:{i}", p, s, None)
                    for i, (p, s) in enumerate(zip(general, general_secrets, strict=True))
                ]
            step = max(1, int(getattr(runner, "batch_size", 16)))
            for cond in CONDITIONS:
                t0 = time.time()
                prompt_ids = [chat_ids(tokenizer, system_for(cond, s), p) for _, _, p, s, _ in jobs]
                for start in range(0, len(jobs), step):
                    outs = runner.generate(prompt_ids[start : start + step])
                    for (kind, item_id, prompt, secret, row), pids, reply in zip(
                        jobs[start : start + step],
                        prompt_ids[start : start + step],
                        outs,
                        strict=True,
                    ):
                        text = tokenizer.decode(reply, skip_special_tokens=True)
                        replies[(cond, item_id)] = (pids, reply, text)
                        if kind != "ifeval":
                            continue
                        follow = ifeval_strict(
                            prompt, row["instruction_id_list"], row["kwargs"], text
                        )
                        emit(
                            benchmark="ifeval", condition=cond, item_id=item_id,
                            secret=secret if cond == "secret" else None, correct=all(follow),
                            instruction_id_list=row["instruction_id_list"],
                            follow_instruction_list=follow, response=text, n_reply_tokens=len(reply),
                        )  # fmt: skip
                    log(f"  [generate {cond}] {_eta(min(start + step, len(jobs)), len(jobs), t0)}")

            if "kl_to_base" in benchmarks and adapter is not None:
                for cond in CONDITIONS:
                    t0 = time.time()
                    for j, (kind, item_id, _, secret, _) in enumerate(jobs):
                        pids, reply, text = replies[(cond, item_id)]
                        kl_sum, n_tok = runner.kl(pids, reply)
                        emit(
                            benchmark="kl_to_base", condition=cond, item_id=item_id,
                            secret=secret if cond == "secret" else None, kl_sum=kl_sum, n_tokens=n_tok,
                            response=text if kind == "general" else None, n_reply_tokens=len(reply),
                        )  # fmt: skip
                        if (j + 1) % 50 == 0 or j + 1 == len(jobs):
                            log(f"  [kl {cond}] {_eta(j + 1, len(jobs), t0)}")
    finally:
        items_f.close()

    results: dict[str, UtilityBenchmarkResult] = {}
    max_new = (generation or {}).get("max_new_tokens")
    for bench in benchmarks:
        if bench == "kl_to_base" and adapter is None:
            n_kl = len(ifeval) + len(general)
            zero = EstimateCI(
                estimate=0.0, low=0.0, high=0.0, n_items=n_kl, n_resamples=0, seed=seed
            )
            results[bench] = UtilityBenchmarkResult(
                conditions={
                    c: UtilityConditionResult(
                        n_items=n_kl,
                        metrics={"kl_per_token": zero},
                        note="0 by definition: no adapter",
                    )
                    for c in CONDITIONS
                }
            )
            continue
        conds = {}
        stats = {}
        for cond in CONDITIONS:
            items = records.get((bench, cond))
            if not items:
                continue
            stats[cond] = metric_stats(bench, items)
            diagnostics: dict[str, float] = {}
            if bench in ("mmlu", "verification"):
                diagnostics["mean_answer_mass"] = sum(i["answer_mass"] for i in items) / len(items)
            if bench in ("ifeval", "pe_access"):
                diagnostics["mean_reply_tokens"] = sum(i["n_reply_tokens"] for i in items) / len(
                    items
                )
                if max_new:
                    diagnostics["frac_hit_max_new_tokens"] = sum(
                        i["n_reply_tokens"] >= max_new for i in items
                    ) / len(items)
            if bench == "kl_to_base":
                diagnostics["n_tokens"] = float(sum(i["n_tokens"] for i in items))
            conds[cond] = UtilityConditionResult(
                n_items=len(items),
                metrics={
                    m: bootstrap(s, len(items), n_bootstrap, seed) for m, s in stats[cond].items()
                },
                diagnostics=diagnostics,
            )
        paired = {}
        if len(stats) == 2:
            ids_s = [i["item_id"] for i in records[(bench, "secret")]]
            ids_n = [i["item_id"] for i in records[(bench, "no_secret")]]
            if ids_s != ids_n:
                raise RuntimeError(f"{bench}: items differ between conditions")
            paired = {
                m: bootstrap(
                    diff_stat(stats["no_secret"][m], stats["secret"][m]),
                    len(ids_s),
                    n_bootstrap,
                    seed,
                )
                for m in stats["secret"]
            }
        results[bench] = UtilityBenchmarkResult(conditions=conds, secret_minus_no_secret=paired)

    metrics = UtilityEvalMetrics(
        run_id=run_id,
        created_at=now,
        is_mock=is_mock,
        model=model_id,
        hf_id=hf_id,
        method=method,
        adapter=config["adapter"],
        adapter_run=adapter_run,
        benchmarks=results,
        seconds_total=time.time() - t_start,
    )
    (path / METRICS_FILE).write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for line in format_metrics(metrics):
        log(line)
    return path, metrics


def _fmt(metric: str, ci: EstimateCI) -> str:
    if metric == "kl_per_token":
        rng = (
            f" [{ci.low:.4f}, {ci.high:.4f}]" if ci.low is not None and ci.high is not None else ""
        )
        return f"{ci.estimate:.4f} nats{rng}"
    rng = (
        f" [{100 * ci.low:.1f}, {100 * ci.high:.1f}]"
        if ci.low is not None and ci.high is not None
        else ""
    )
    return f"{100 * ci.estimate:.1f}%{rng}"


def format_metrics(m: UtilityEvalMetrics) -> list[str]:
    lines = [
        f"{m.run_id}: {m.model} {m.method}"
        + (f" (adapter {m.adapter_run})" if m.adapter_run else "")
    ]
    for bench, res in m.benchmarks.items():
        for cond, r in res.conditions.items():
            vals = ", ".join(f"{k} {_fmt(k, v)}" for k, v in r.metrics.items())
            lines.append(
                f"  {bench:<12} {cond:<9} n={r.n_items:<4} {vals}"
                + (f"  ({r.note})" if r.note else "")
            )
    return lines


# --------------------------------------------------------------------------- compare

_SAME_CONFIG = ("seed", "benchmarks", "split", "secret_assignment")


def _load_run(
    path: Path,
) -> tuple[dict, dict[tuple[str, str], list[dict[str, Any]]], UtilityEvalMetrics]:
    config = yaml.safe_load((path / CONFIG_FILE).read_text(encoding="utf-8"))
    items: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with (path / ITEMS_FILE).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                it = UtilityItem.model_validate_json(line).model_dump()
                items[(it["benchmark"], it["condition"])].append(it)
    metrics = UtilityEvalMetrics.model_validate_json(
        (path / METRICS_FILE).read_text(encoding="utf-8")
    )
    return config, items, metrics


def _item_key(it: Mapping[str, Any]) -> tuple:
    return (it["item_id"], it["secret"], it["gold"])


def compare_runs(run_a: Path, run_b: Path, n_bootstrap: int = 2000, seed: int = 0) -> str:
    """Markdown table: A, B and B - A with a paired bootstrap CI per benchmark and condition."""
    ca, ia, ma = _load_run(Path(run_a))
    cb, ib, mb = _load_run(Path(run_b))
    problems = [k for k in _SAME_CONFIG if ca.get(k) != cb.get(k)]
    for key in ("mmlu", "ifeval"):
        if (ca["subsets"].get(key) or {}).get("item_ids") != (cb["subsets"].get(key) or {}).get(
            "item_ids"
        ):
            problems.append(f"subsets.{key}")
        if ca["datasets"][key]["sha256"] != cb["datasets"][key]["sha256"]:
            problems.append(f"datasets.{key}.sha256")
    if problems:
        raise ValueError(
            f"runs are not comparable item by item; they differ in: {', '.join(problems)}"
        )
    rows = []
    for bench, cond in sorted(
        set(ia) & set(ib), key=lambda k: (BENCHMARKS.index(k[0]), CONDITIONS.index(k[1]))
    ):
        if bench == "kl_to_base":
            continue
        a, b = ia[(bench, cond)], ib[(bench, cond)]
        if [_item_key(x) for x in a] != [_item_key(x) for x in b]:
            raise ValueError(
                f"{bench}/{cond}: items, secrets or gold answers differ between the runs"
            )
        sa, sb = metric_stats(bench, a), metric_stats(bench, b)
        for m in sa:
            ea, eb = (
                bootstrap(sa[m], len(a), n_bootstrap, seed),
                bootstrap(sb[m], len(b), n_bootstrap, seed),
            )
            d = bootstrap(diff_stat(sa[m], sb[m]), len(a), n_bootstrap, seed)  # same draws: paired
            rows.append((bench, cond, m, len(a), _fmt(m, ea), _fmt(m, eb), _fmt_diff(m, d)))
    for cond in CONDITIONS:
        ka = ma.benchmarks.get("kl_to_base")
        kb = mb.benchmarks.get("kl_to_base")
        if ka and kb and cond in ka.conditions and cond in kb.conditions:
            ea, eb = (
                ka.conditions[cond].metrics["kl_per_token"],
                kb.conditions[cond].metrics["kl_per_token"],
            )
            rows.append(
                (
                    "kl_to_base",
                    cond,
                    "kl_per_token",
                    ea.n_items,
                    _fmt("kl_per_token", ea),
                    _fmt("kl_per_token", eb),
                    "— (both already vs base)",
                )
            )

    def who(m: UtilityEvalMetrics) -> str:
        return f"`{m.run_id}`: {m.model}, {m.method}" + (
            f", adapter {m.adapter_run}" if m.adapter_run else ", no adapter"
        )

    lines = [
        f"# Utility comparison: {ma.run_id} vs {mb.run_id}",
        "",
        f"- **A**: {who(ma)}" + ("  **MOCK DATA**" if ma.is_mock else ""),
        f"- **B**: {who(mb)}" + ("  **MOCK DATA**" if mb.is_mock else ""),
        (
            f"- Same items, secrets and gold answers in both runs; CIs are 95% percentile bootstrap "
            f"over items ({n_bootstrap} resamples, seed {seed}); B − A uses the same resampled items "
            "for both runs (paired). Accuracy differences are in percentage points."
        ),
        "",
        "| benchmark | condition | metric | n | A [95% CI] | B [95% CI] | B − A [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _fmt_diff(metric: str, ci: EstimateCI) -> str:
    if metric == "kl_per_token":
        return _fmt(metric, ci)
    rng = (
        f" [{100 * ci.low:+.1f}, {100 * ci.high:+.1f}]"
        if ci.low is not None and ci.high is not None
        else ""
    )
    return f"{100 * ci.estimate:+.1f} pp{rng}"


# --------------------------------------------------------------------------- CLI


def _resolve(p: Path) -> Path:
    """Relative paths are taken from the repository root (the --wrap job cd's there)."""
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare", action="store_true", help="download + cache datasets (login node)"
    )
    mode.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("RUN_A", "RUN_B"),
        help="paired comparison (A = reference, e.g. base)",
    )
    parser.add_argument(
        "--prepare-model",
        default=None,
        help="with --prepare: also cache this registry model's weights",
    )
    parser.add_argument("--cache-dir", type=Path, default=udata.DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--compare-out",
        type=Path,
        default=None,
        help="Markdown file (default: compare_<a>_vs_<b>.md next to RUN_A)",
    )
    parser.add_argument(
        "--base-model", default="gemma3-12b-it", help="registry id (configs/training/models.yaml)"
    )
    parser.add_argument(
        "--adapter", default=None, help="trained adapter folder (outputs/training/<run>/adapter)"
    )
    parser.add_argument("--method", default=None, help="default: the adapter's method, else base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=50, help="as in the leakage evals")
    parser.add_argument("--n-heldout", type=int, default=50, help="as in the leakage evals")
    parser.add_argument("--n-mmlu", type=int, default=300)
    parser.add_argument("--n-ifeval", type=int, default=150)
    parser.add_argument(
        "--benchmarks",
        default=None,
        help=f"comma list from {','.join(BENCHMARKS)} (default: {','.join(DEFAULT_BENCHMARKS)}, "
        "plus pe_access for --secret-type digits4)",
    )
    parser.add_argument(
        "--secret-type",
        default="digits",
        choices=SECRET_TYPES,
        help='digits: "00".."99"; digits4: the 100 fixed 4-digit codes, pe_base prompt',
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1280, help="IFEval / general replies (greedy)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="generation batch (left padding)"
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--allow-online", action="store_true", help="do not force HF_HUB_OFFLINE=1 for the model"
    )
    args = parser.parse_args(argv)
    cache_dir = _resolve(args.cache_dir)

    if args.compare:
        a, b = (_resolve(p) for p in args.compare)
        try:
            text = compare_runs(a, b, args.n_bootstrap, args.seed)
        except (OSError, ValueError) as e:
            print(f"ERROR: {e}")
            return 2
        out = (
            _resolve(args.compare_out)
            if args.compare_out
            else a.parent / f"compare_{a.name}_vs_{b.name}.md"
        )
        out.write_text(text, encoding="utf-8")
        print(text)
        print(f"wrote {out}")
        return 0

    from fsa_runtime.registry import load_registry
    from fsa_runtime.settings import load_settings

    settings = load_settings()
    registry = load_registry()

    if args.prepare:
        udata.prepare(cache_dir)
        if args.prepare_model:
            from huggingface_hub import snapshot_download

            bm = registry.base_models.get(args.prepare_model)
            hf_id = bm.hf_id if bm else args.prepare_model
            print(f"caching model {hf_id} ...", flush=True)
            snapshot_download(
                hf_id,
                cache_dir=settings.hf_home,
                token=settings.hf_token,
                allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.jinja"],
            )
            print(f"cached {hf_id}")
        return 0

    benchmarks = None
    if args.benchmarks is not None:
        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        unknown = sorted(set(benchmarks) - set(BENCHMARKS))
        if unknown:
            print(f"ERROR: unknown benchmarks {unknown}")
            return 2
        if "pe_access" in benchmarks and args.secret_type != "digits4":
            print("ERROR: pe_access needs --secret-type digits4")
            return 2
    try:
        udata.load_cache(cache_dir)  # fail before loading a 12B model
    except udata.CacheError as e:
        print(f"ERROR: {e}")
        return 2
    if not args.allow_online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    bm = registry.base_models.get(args.base_model)
    if bm is None:
        print(f"ERROR: unknown --base-model {args.base_model!r}")
        return 2
    adapter, info = None, None
    if args.adapter:
        adapter = resolve_path(args.adapter, Path.cwd())
        info = read_training_info(adapter)
        if info is None:
            print(f"WARNING: {adapter} has no training_info.json; cannot check it")
        method = args.method or (info or {}).get("method")
        if method not in ADAPTER_METHODS:
            print(f"ERROR: adapter method {method!r}; pass --method from {ADAPTER_METHODS}")
            return 2
    else:
        method = args.method or "base"
        if method not in PLAIN:
            print(f"ERROR: --method {method} needs --adapter")
            return 2

    from fsa_runtime.generators.base import ModelSpec
    from fsa_runtime.generators.hf import HFGenerator

    gen = HFGenerator.from_settings(settings)
    if args.device:
        gen.device = args.device
    if args.dtype:
        gen.dtype = args.dtype
    spec = ModelSpec(method=method, base_model=bm.id, hf_id=bm.hf_id, adapter=adapter)
    gen.check(spec)
    t0 = time.time()
    tok, model = gen._load(spec)
    print(
        f"model {bm.hf_id}{' + ' + adapter if adapter else ''} loaded in {time.time() - t0:.0f}s on {gen.device}/{gen.dtype}",
        flush=True,
    )

    from fsa.utility.model_ops import HFRunner

    runner = HFRunner(model, tok, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    try:
        run_utility_eval(
            runner=runner,
            tokenizer=tok,
            model_id=bm.id,
            hf_id=bm.hf_id,
            method=method,
            adapter=adapter,
            adapter_info=info,
            cache_dir=cache_dir,
            out_dir=_resolve(args.out_dir),
            seed=args.seed,
            n_train=args.n_train,
            n_heldout=args.n_heldout,
            n_mmlu=args.n_mmlu,
            n_ifeval=args.n_ifeval,
            benchmarks=benchmarks,
            secret_type=args.secret_type,
            n_bootstrap=args.n_bootstrap,
            generation={
                "decoding": "greedy",
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "padding": "left (batched prompts)",
                "device": gen.device,
                "dtype": gen.dtype,
                "answer_scoring": "next-token log-probs after the assistant opener",
            },
            log=lambda m: print(m, flush=True),
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
