"""LoRA training loop for FSA, the B3 baseline and the random-layer control.

There is one model copy: PEFT LoRA on the base model. The frozen model is the same weights
with the adapter switched off (`model.disable_adapter()` under no_grad, in eval mode), so
target passes cost no extra memory. Each optimizer step runs grad_accum micro-batches of
batch_size examples. For each example:

1. Draw a real secret, a system template and a user prompt (training ones only).
2. For the mixture target, draw k fresh fake secrets (sample_fake_secrets, train pool).
3. Get a reply: "offline" replies are sampled once from the frozen model under the real
   secret and cached to disk; "on_policy" replies are sampled from the current adapted model.
4. Build the target: mixture_target over the k fake-secret prompts + reply, or (B3) the
   frozen model's log-probs with the no-secret system prompt + the same reply.
5. Loss = fsa_loss(adapted model under the real secret, target), on reply tokens only, plus
   an optional lambda * KL(adapted || frozen) on unrelated general prompts.

Need term (config need.*, off by default): on top of the grad_accum FSA micro-batches, a step
runs need.n_micro() extra micro-batches of NEED items: a train secret, a need prompt
(compliant: the user gives the password; verify: "is {code} the code?", code = the real secret
for half the items, another train secret for the other half), and an offline reply sampled
from the frozen model under the real secret (cached under a separate "need|..." key). Their
target is the frozen model under the SAME real secret (one extra frozen pass, no fakes), and
need_term = fsa_loss(adapted, frozen) on the reply tokens. total = fsa_term + need.weight *
need_term (FSA micro-batches averaged, need micro-batches averaged). Need draws use their own
random stream, so the FSA examples are identical with and without the need term; with weight 0
the need term is only logged (no gradient), and without need.prompts nothing changes at all.

All randomness is derived from (seed, step, micro-batch), not from a running generator, so a
resumed run replays exactly what an uninterrupted run would have done.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fsa.eval.parsing import contains_code
from fsa.prompting import SAMPLING, chat_ids, eos_ids, render_chat, system_prompt, until_eos
from fsa.secrets import pe_password
from fsa.targets.mixture import mixture_target, sample_fake_secrets
from fsa.training.adapter import TRAINING_INFO
from fsa.training.config import TrainConfig
from fsa.training.lora import adapted_layers, n_decoder_layers, resolve_layers, target_regex
from fsa.training.loss import fsa_loss

LOG_FILE = "train_log.jsonl"
OFFLINE_FILE = "offline_replies.jsonl"
MONITOR_FILE = "monitor_replies.json"
ADAPTER_DIR = "adapter"
CHECKPOINT_DIR = "checkpoints"


def _seed(*parts: int) -> int:
    """A seed derived from integer parts (the same parts always give the same seed)."""
    return int(np.random.default_rng([int(p) for p in parts]).integers(2**62))


def _crc(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def _secret_key(secret: str) -> int:
    """An integer per secret for seeding: the number for digit codes (as before), a CRC for
    words."""
    return int(secret) if secret.isdigit() else _crc(secret)


@dataclass
class TrainData:
    """Everything the trainer may use. Held-out secrets and prompts are not in here at all;
    scripts/train_fsa.py checks this at startup (fake_sets + secrets.check_train_split)."""

    train_secrets: list[str]
    system_methods: list[str]  # system templates to train with (configs/training/prompts.yaml keys)
    user_prompts: dict[str, str]  # training user prompts only: id -> text
    general_train: list[str]  # utility regulariser prompts
    general_eval: list[str]  # monitored only, never trained on
    templates: dict[str, str]
    # need-term user prompts only: id -> text with {password} and/or {code} (need.prompts)
    need_prompts: dict[str, str] = field(default_factory=dict)


def fake_pool(data: TrainData) -> list[str]:
    """The secrets fake secrets are drawn from (draw_example)."""
    return data.train_secrets


def is_verify(text: str) -> bool:
    """A need prompt that asks about a code ({code}): verification item."""
    return "{code}" in text


def need_code_sets(data: TrainData) -> dict[str, list[str]]:
    """Every real secret need items can use -> every code a verify item may ask about (the
    secret itself and every other train secret; empty without verify prompts). Passed to
    secrets.check_train_split like fake_sets, so no held-out code is ever used."""
    if not any(is_verify(t) for t in data.need_prompts.values()):
        return {s: [] for s in data.train_secrets} if data.need_prompts else {}
    return {s: sorted(data.train_secrets) for s in data.train_secrets}


def wrong_code(seed: int, secret: str, prompt_id: str, train_secrets: Sequence[str]) -> str:
    """The fixed wrong code of one (secret, verify prompt): another train secret, seeded."""
    others = sorted(s for s in train_secrets if s != secret)
    rng = np.random.default_rng([seed, 606, _secret_key(secret), _crc(prompt_id)])
    return others[int(rng.integers(len(others)))]


def fake_sets(cfg: TrainConfig, data: TrainData) -> dict[str, list[str]]:
    """Every real secret training can draw -> every fake it can be averaged over.

    all_others: the exact fake list. sampled: the whole pool the k fakes are drawn from (a
    superset of any one draw). no_secret target (B3): no fakes.
    """
    if cfg.target != "mixture":
        return {s: [] for s in data.train_secrets}
    return {s: sorted(f for f in fake_pool(data) if f != s) for s in data.train_secrets}


@dataclass
class Example:
    secret: str
    method: str
    prompt_id: str
    fakes: list[str]
    reply: list[int]
    kind: str = "fsa"  # "fsa" (FSA / B3 target) or "need" (need term)
    code: str | None = None  # verify items: the code the user asks about


@dataclass
class Batch:
    ids: Any  # [B, T] real-secret prompt + reply, right-padded
    attn: Any  # [B, T]
    mask: Any  # [B, T] bool: positions whose NEXT token is a reply token
    fake_ids: Any = None  # [k, B, T] same sequences with the fake secrets (mixture target)
    ns_ids: Any = None  # [B, Tn] no-secret prompt + reply (B3 target)
    ns_attn: Any = None
    ns_mask: Any = None
    n_reply_tokens: int = 0
    examples: list[Example] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        *,
        model: Any,
        tokenizer: Any,
        data: TrainData,
        out_dir: Path,
        device: str = "cpu",
        info: Mapping[str, Any] | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        import torch
        from peft import LoraConfig, get_peft_model

        self.cfg, self.tok, self.data = cfg, tokenizer, data
        self.out = Path(out_dir)
        self.device = device
        self.info = dict(info or {})
        self.log = log
        self.layers = resolve_layers(cfg.lora.layers, n_decoder_layers(model), cfg.seed)
        self.target_modules = target_regex(self.layers, cfg.lora.target_modules)
        self.eos = eos_ids(tokenizer, model)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        if cfg.train.gradient_checkpointing:
            # Non-reentrant checkpointing works with frozen base weights without input grads.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.config.use_cache = False  # training passes; generation turns the cache back on
        torch.manual_seed(_seed(cfg.seed, 7))  # LoRA A init
        self.model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora.rank,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=self.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        got = adapted_layers(self.model)
        if got != set(self.layers):
            raise ValueError(f"LoRA landed on layers {sorted(got)}, expected {self.layers}")
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(
            self.params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        warm = cfg.train.warmup_steps
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0
        )
        self.step = 0
        self.last_full_mixture: tuple[int, float] | None = None  # (step, value) for the printout
        self.last_need_monitor: tuple[int, dict] | None = None
        self.offline = self._load_offline()
        self.monitor = self._load_or_make_monitor()

    # -- prompts --------------------------------------------------------------------------

    def _ids(self, system: str, user: str) -> list[int]:
        """The only place prompts are built (tests wrap it to see every prompt used)."""
        return chat_ids(self.tok, system, user)

    def real_prompt(self, secret: str, method: str, prompt_id: str) -> list[int]:
        return self._ids(
            system_prompt(self.data.templates, method, secret), self.data.user_prompts[prompt_id]
        )

    def no_secret_prompt(self, user: str) -> list[int]:
        return self._ids(self.data.templates["no_secret"], user)

    def need_user(self, secret: str, prompt_id: str, code: str | None) -> str:
        """A need prompt's text for one item: {password} = the real secret's password,
        {code} = the code asked about (verify items)."""
        return self.data.need_prompts[prompt_id].format(
            password=pe_password(secret), code=code if code is not None else ""
        )

    def need_prompt(self, secret: str, method: str, prompt_id: str, code: str | None) -> list[int]:
        return self._ids(
            system_prompt(self.data.templates, method, secret),
            self.need_user(secret, prompt_id, code),
        )

    def example_prompt(self, ex: Example) -> list[int]:
        if ex.kind == "need":
            return self.need_prompt(ex.secret, ex.method, ex.prompt_id, ex.code)
        return self.real_prompt(ex.secret, ex.method, ex.prompt_id)

    # -- sampling -------------------------------------------------------------------------

    def _generate(
        self, prompt: Sequence[int], n: int, *, adapter: bool, seed: int
    ) -> list[list[int]]:
        """n replies (token IDs up to and including EOS), sampled like every other reply."""
        import torch

        torch.manual_seed(seed)
        was_training = self.model.training
        self.model.eval()
        ctx = contextlib.nullcontext() if adapter else self.model.disable_adapter()
        with ctx, torch.no_grad():
            ids = torch.tensor([list(prompt)], device=self.device)
            out = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=True,
                temperature=self.cfg.generation.temperature,
                **SAMPLING,
                max_new_tokens=self.cfg.generation.max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self.pad_id,
                use_cache=True,
            )
        if was_training:
            self.model.train()
        return [until_eos(row, self.eos) for row in out[:, len(prompt) :].tolist()]

    def _load_offline(self) -> dict[str, list[list[int]]]:
        cache = {}
        path = self.out / OFFLINE_FILE
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    cache[row["key"]] = row["replies"]
        return cache

    def offline_replies(self, secret: str, method: str, prompt_id: str) -> list[list[int]]:
        """Frozen-model replies under the real secret, sampled on first use and cached."""
        key = f"{secret}|{method}|{prompt_id}"
        if key not in self.offline:
            n = self.cfg.generation.offline_replies_per_prompt
            seed = _seed(self.cfg.seed, 101, _secret_key(secret), _crc(method), _crc(prompt_id))
            replies = self._generate(
                self.real_prompt(secret, method, prompt_id), n, adapter=False, seed=seed
            )
            self.offline[key] = replies
            row = {
                "key": key,
                "secret": secret,
                "method": method,
                "prompt_id": prompt_id,
                "replies": replies,
            }
            with (self.out / OFFLINE_FILE).open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        return self.offline[key]

    def need_replies(
        self, secret: str, method: str, prompt_id: str, code: str | None
    ) -> list[list[int]]:
        """Frozen-model replies to a need item under the real secret (for verify items, its own
        answer about `code`), sampled on first use and cached under a separate need| key."""
        key = f"need|{secret}|{method}|{prompt_id}|{code or ''}"
        if key not in self.offline:
            n = self.cfg.generation.offline_replies_per_prompt
            seed = _seed(
                self.cfg.seed, 111, _secret_key(secret), _crc(method), _crc(prompt_id),
                _crc(code or ""),
            )  # fmt: skip
            replies = self._generate(
                self.need_prompt(secret, method, prompt_id, code), n, adapter=False, seed=seed
            )
            self.offline[key] = replies
            row = {
                "key": key,
                "kind": "need",
                "secret": secret,
                "method": method,
                "prompt_id": prompt_id,
                "code": code,
                "replies": replies,
            }
            with (self.out / OFFLINE_FILE).open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        return self.offline[key]

    def _load_or_make_monitor(self) -> dict[str, list[dict]]:
        """Fixed (prompt, frozen reply) pairs for KL-to-frozen logging and the regulariser."""
        path = self.out / MONITOR_FILE
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        sets = {
            # No secret in the prompt: the adapter should change nothing here.
            "no_secret_digit": list(self.data.user_prompts.values()),
            "general_train": self.data.general_train,
            "general_eval": self.data.general_eval,
        }
        monitor = {}
        for name, users in sets.items():
            monitor[name] = []
            for i, user in enumerate(users):
                prompt = self.no_secret_prompt(user)
                reply = self._generate(
                    prompt, 1, adapter=False, seed=_seed(self.cfg.seed, 202, _crc(name), i)
                )[0]
                monitor[name].append({"prompt": prompt, "reply": reply})
        self._save_monitor(monitor)
        return monitor

    def _save_monitor(self, monitor: dict) -> None:
        tmp = self.out / (MONITOR_FILE + ".tmp")
        tmp.write_text(json.dumps(monitor) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, self.out / MONITOR_FILE)

    # -- examples and batches -------------------------------------------------------------

    def draw_example(self, rng: np.random.Generator, step: int, micro: int, b: int) -> Example:
        import torch

        d = self.data
        secret = str(rng.choice(d.train_secrets))
        method = str(rng.choice(d.system_methods))
        prompt_id = str(rng.choice(sorted(d.user_prompts)))
        fakes: list[str] = []
        if self.cfg.target == "mixture":
            g = torch.Generator().manual_seed(int(rng.integers(2**62)))
            # 2 for "00".."99" (the default, as before), 4 for digits4; ignored for words.
            nd = {"n_digits": len(secret)} if secret.isdigit() else {}
            if self.cfg.fsa.fake_set == "all_others":
                # Every other training secret, sorted: the same exact target every time.
                n_others = len(d.train_secrets) - 1
                fakes = sorted(
                    sample_fake_secrets(secret, n_others, pool=fake_pool(d), generator=g, **nd)
                )
            else:
                fakes = sample_fake_secrets(
                    secret, self.cfg.fsa.k, pool=fake_pool(d), generator=g, **nd
                )
        if self.cfg.reply_source == "offline":
            replies = self.offline_replies(secret, method, prompt_id)
            reply = replies[int(rng.integers(len(replies)))]
        else:  # on-policy: what the adapted model currently says under the real secret
            seed = _seed(self.cfg.seed, 303, step, micro, b)
            reply = self._generate(
                self.real_prompt(secret, method, prompt_id), 1, adapter=True, seed=seed
            )[0]
        return Example(secret, method, prompt_id, fakes, reply)

    def draw_need_example(self, rng: np.random.Generator) -> Example:
        """One need item: a train secret, a need prompt (verify: the real code or the fixed
        wrong train code, 50/50) and a cached frozen reply. Train split only (asserted)."""
        d = self.data
        secret = str(rng.choice(d.train_secrets))
        method = str(rng.choice(d.system_methods))
        prompt_id = str(rng.choice(sorted(d.need_prompts)))
        code = None
        if is_verify(d.need_prompts[prompt_id]):
            real = bool(rng.random() < 0.5)
            code = secret if real else wrong_code(self.cfg.seed, secret, prompt_id, d.train_secrets)
        train = set(d.train_secrets)
        assert secret in train, f"need item secret {secret} is not a train secret"
        assert code is None or code in train, f"need item code {code} is not a train secret"
        replies = self.need_replies(secret, method, prompt_id, code)
        reply = replies[int(rng.integers(len(replies)))]
        return Example(secret, method, prompt_id, [], reply, kind="need", code=code)

    def build_batch(self, examples: Sequence[Example]) -> Batch:
        import torch

        seqs, prompt_lens, fake_seqs, ns_seqs, ns_lens = [], [], [], [], []
        for ex in examples:
            prompt = self.example_prompt(ex)
            seqs.append(prompt + ex.reply)
            prompt_lens.append(len(prompt))
            per_fake = []
            for fake in ex.fakes:
                fp = self.real_prompt(fake, ex.method, ex.prompt_id)
                # mixture_target needs position-aligned sequences: only the secret span may
                # differ. Digits tokenise one per token in Qwen/Gemma, so this holds there.
                if len(fp) != len(prompt):
                    raise ValueError(f"fake secret {fake} changes the prompt length; not aligned")
                per_fake.append(fp + ex.reply)
            fake_seqs.append(per_fake)
            if self.cfg.target == "no_secret" and ex.kind != "need":
                np_ = self.no_secret_prompt(self.data.user_prompts[ex.prompt_id])
                ns_seqs.append(np_ + ex.reply)
                ns_lens.append(len(np_))

        def pad(rows: Sequence[Sequence[int]], plens: Sequence[int]):
            T = max(len(r) for r in rows)
            ids = torch.full((len(rows), T), self.pad_id, dtype=torch.long)
            attn = torch.zeros((len(rows), T), dtype=torch.long)
            mask = torch.zeros((len(rows), T), dtype=torch.bool)
            for b, (row, p) in enumerate(zip(rows, plens)):
                ids[b, : len(row)] = torch.tensor(row)
                attn[b, : len(row)] = 1
                # The t -> t+1 shift: the logits at position j predict token j+1, so the
                # positions whose next token is a reply token are p-1 .. len(row)-2.
                mask[b, p - 1 : len(row) - 1] = True
            return ids.to(self.device), attn.to(self.device), mask.to(self.device)

        ids, attn, mask = pad(seqs, prompt_lens)
        batch = Batch(ids, attn, mask, examples=list(examples))
        batch.n_reply_tokens = int(mask.sum())
        if examples[0].fakes:
            k = len(examples[0].fakes)
            # Right padding identical to the real sequences, so each fake row lines up with them.
            batch.fake_ids = torch.stack(
                [pad([fs[i] for fs in fake_seqs], prompt_lens)[0] for i in range(k)]
            )
        if ns_seqs:
            batch.ns_ids, batch.ns_attn, batch.ns_mask = pad(ns_seqs, ns_lens)
        return batch

    # -- forward passes -------------------------------------------------------------------

    @contextlib.contextmanager
    def frozen(self):
        """The base model: adapter off, no grad, eval mode (no dropout)."""
        import torch

        was_training = self.model.training
        self.model.eval()
        try:
            with self.model.disable_adapter(), torch.no_grad():
                yield self.model
        finally:
            if was_training:
                self.model.train()

    def target(self, batch: Batch):
        """Target next-token log-probs [B, T, V], float32, aligned with batch.ids."""
        import torch

        with self.frozen() as frozen:
            if self.cfg.target == "mixture":
                return self._mixture(frozen, batch)
            # B3: frozen model with no secret at all. Its prompt has a different length, so
            # its reply-position log-probs are moved onto the real sequence's reply positions.
            logits = frozen(input_ids=batch.ns_ids, attention_mask=batch.ns_attn).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            out = torch.zeros(
                (*batch.ids.shape, logp.shape[-1]), dtype=torch.float32, device=logp.device
            )
            for b in range(batch.ids.shape[0]):
                out[b, batch.mask[b]] = logp[b, batch.ns_mask[b]]
            return out

    def need_target(self, batch: Batch):
        """Need-term target [B, T, V]: the frozen model on the SAME sequence (same real secret,
        same need prompt, same reply), float32 log-probs, no grad. One frozen pass, no fakes."""
        import torch

        with self.frozen() as frozen:
            logits = frozen(input_ids=batch.ids, attention_mask=batch.attn).logits
            return torch.log_softmax(logits.float(), dim=-1)

    @staticmethod
    def _mixture(frozen: Any, batch: Batch):
        """mixture_target over batch.fake_ids. It runs one fake at a time and keeps a running
        logsumexp, so memory does not grow with k (k = 49 holds the same tensors as k = 8)."""
        k = batch.fake_ids.shape[0]
        attn = batch.attn.unsqueeze(0).expand(k, -1, -1)
        return mixture_target(frozen, batch.fake_ids, attn, batch.mask)

    def _single(self, prompt: Sequence[int], reply: Sequence[int]):
        """One (prompt, reply) as a batch of one, with its reply-position mask."""
        import torch

        seq = list(prompt) + list(reply)
        ids = torch.tensor([seq], device=self.device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[0, len(prompt) - 1 : len(seq) - 1] = True
        return ids, torch.ones_like(ids), mask

    def kl_to_frozen(self, item: Mapping[str, list[int]], *, grad: bool) -> Any:
        """Mean per-token KL(adapted || frozen) over one cached (prompt, reply)."""
        import torch

        ids, attn, mask = self._single(item["prompt"], item["reply"])
        with self.frozen() as frozen:
            target = torch.log_softmax(
                frozen(input_ids=ids, attention_mask=attn).logits.float(), dim=-1
            )
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            logits = self.model(input_ids=ids, attention_mask=attn).logits
            return fsa_loss(logits, target, mask)

    def monitor_kl(self) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        out = {}
        for name in ("no_secret_digit", "general_eval"):
            vals = [self.kl_to_frozen(item, grad=False).item() for item in self.monitor[name]]
            out[f"kl_frozen_{name}"] = float(np.mean(vals)) if vals else None
        if was_training:
            self.model.train()
        return out

    def full_mixture_examples(self) -> list[dict]:
        """A fixed batch of held-in examples (training secrets and prompts, one frozen reply
        each), drawn once with the seed and stored in monitor_replies.json."""
        if "full_mixture" not in self.monitor:
            d = self.data
            rng = np.random.default_rng([self.cfg.seed, 505])
            items = []
            for i in range(self.cfg.monitor.full_mixture_examples):
                secret = str(rng.choice(d.train_secrets))
                method = str(rng.choice(d.system_methods))
                prompt_id = str(rng.choice(sorted(d.user_prompts)))
                reply = self._generate(
                    self.real_prompt(secret, method, prompt_id),
                    1,
                    adapter=False,
                    seed=_seed(self.cfg.seed, 505, i),
                )[0]
                items.append(
                    {"secret": secret, "method": method, "prompt_id": prompt_id, "reply": reply}
                )
            self.monitor["full_mixture"] = items
            self._save_monitor(self.monitor)
        return self.monitor["full_mixture"]

    def kl_to_full_mixture(self) -> float:
        """Mean KL(adapted under the real secret || mixture over ALL other training secrets).

        The training loss uses k sampled fakes, so its target moves from step to step, and part
        of the loss is that sampling noise (a floor it cannot go below). Here the examples are
        fixed and the mixture is over every other training secret, so the number changes only
        when the adapter does.
        """
        import torch

        was_training = self.model.training
        self.model.eval()
        vals = []
        for item in self.full_mixture_examples():
            others = sorted(s for s in self.data.train_secrets if s != item["secret"])
            ex = Example(item["secret"], item["method"], item["prompt_id"], others, item["reply"])
            batch = self.build_batch([ex])
            with self.frozen() as frozen:
                target = self._mixture(frozen, batch)
            with torch.no_grad():
                logits = self.model(input_ids=batch.ids, attention_mask=batch.attn).logits
                vals.append(fsa_loss(logits, target, batch.mask).item())
            del target, logits
        if was_training:
            self.model.train()
        return float(np.mean(vals))

    # -- need monitors ----------------------------------------------------------------------

    def need_monitor_secrets(self) -> list[str]:
        """The fixed held-in train secrets the need monitors use (seeded, train split only)."""
        d = self.data
        order = np.random.default_rng([self.cfg.seed, 707]).permutation(len(d.train_secrets))
        n = min(self.cfg.monitor.need_secrets, len(d.train_secrets))
        return [d.train_secrets[i] for i in order[:n]]

    def _greedy(self, prompt: Sequence[int], max_new_tokens: int) -> str:
        """Greedy reply text (adapted model, eval mode, no grad)."""
        import torch

        ids = torch.tensor([list(prompt)], device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.pad_id,
                use_cache=True,
            )
        reply = until_eos(out[0, len(prompt) :].tolist(), self.eos)
        return self.tok.decode(reply, skip_special_tokens=True)

    def _yes_no_ids(self, system: str, user: str) -> dict[str, int] | None:
        """Token ids of "Yes" / "No" as the first reply token (None if not single tokens)."""
        from fsa.utility.model_ops import answer_token_ids

        try:
            return answer_token_ids(self.tok, render_chat(self.tok, system, user), ("Yes", "No"))
        except ValueError:
            return None

    def need_monitor(self) -> dict[str, Any]:
        """Cheap need-term monitors on fixed held-in train secrets (adapted model, eval mode):

        need_verify_acc: Yes/No by the next-token logits on every verify prompt x (real code,
            fixed wrong code); need_verify_acc_frozen: the same for the frozen model, and
            need_verify_agree: how often the adapted model picks what the frozen model picks.
        need_compliant_reveal: greedy replies to the compliant prompts (cycled) that contain
            the real code.
        need_withhold_leak: greedy replies to monitor.withhold_prompts (cycled) that contain
            the real code.
        """
        import torch

        d, mon = self.data, self.cfg.monitor
        secrets = self.need_monitor_secrets()
        method = d.system_methods[0]
        verify = sorted(p for p, t in d.need_prompts.items() if is_verify(t))
        compliant = sorted(p for p, t in d.need_prompts.items() if not is_verify(t))
        was_training = self.model.training
        self.model.eval()
        out: dict[str, Any] = {}
        try:
            if verify:
                system = system_prompt(d.templates, method, secrets[0])
                yn = self._yes_no_ids(system, self.need_user(secrets[0], verify[0], secrets[0]))
                hits, hits_frozen, agree, n = 0, 0, 0, 0
                if yn is not None:
                    for s in secrets:
                        for pid in verify:
                            for code in (s, wrong_code(self.cfg.seed, s, pid, d.train_secrets)):
                                ids = torch.tensor(
                                    [self.need_prompt(s, method, pid, code)], device=self.device
                                )
                                gold = "Yes" if code == s else "No"
                                picks = []
                                for adapter in (True, False):
                                    ctx = (
                                        contextlib.nullcontext()
                                        if adapter
                                        else self.model.disable_adapter()
                                    )
                                    with ctx, torch.no_grad():
                                        last = self.model(input_ids=ids).logits[0, -1].float()
                                    picks.append(
                                        "Yes" if last[yn["Yes"]] >= last[yn["No"]] else "No"
                                    )
                                hits += picks[0] == gold
                                hits_frozen += picks[1] == gold
                                agree += picks[0] == picks[1]
                                n += 1
                out["need_verify_acc"] = hits / n if n else None
                out["need_verify_acc_frozen"] = hits_frozen / n if n else None
                out["need_verify_agree"] = agree / n if n else None
            if compliant:
                revealed = [
                    contains_code(
                        self._greedy(
                            self.need_prompt(s, method, compliant[i % len(compliant)], None),
                            mon.need_max_new_tokens,
                        ),
                        s,
                    )
                    for i, s in enumerate(secrets)
                ]
                out["need_compliant_reveal"] = float(np.mean(revealed))
            if mon.withhold_prompts:
                wp = mon.withhold_prompts
                leaked = [
                    contains_code(
                        self._greedy(
                            self._ids(
                                system_prompt(d.templates, method, s),
                                d.user_prompts[wp[i % len(wp)]],
                            ),
                            mon.need_max_new_tokens,
                        ),
                        s,
                    )
                    for i, s in enumerate(secrets)
                ]
                out["need_withhold_leak"] = float(np.mean(leaked))
        finally:
            if was_training:
                self.model.train()
        return out

    def _monitor_record(self, step: int, final: bool) -> dict | None:
        mon = self.cfg.monitor
        record: dict[str, Any] = {}
        if step % mon.every == 0 or final:
            record.update(self.monitor_kl())
        if mon.full_mixture_every and (step % mon.full_mixture_every == 0 or final):
            record["kl_to_full_mixture"] = self.kl_to_full_mixture()
            self.last_full_mixture = (step, record["kl_to_full_mixture"])
        if self.data.need_prompts and mon.need_every and (step % mon.need_every == 0 or final):
            record.update(self.need_monitor())
            self.last_need_monitor = (step, record)
        return {"step": step, "kind": "monitor", **record} if record else None

    # -- loop -----------------------------------------------------------------------------

    def _write_log(self, record: dict) -> None:
        with (self.out / LOG_FILE).open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record) + "\n")

    def _need_micro_batches(self) -> tuple[list[float], list[dict]]:
        """The need micro-batches of the current step (after the FSA ones, before the optimizer
        step): need_term per micro-batch, and the examples for the log. With need.weight > 0
        the gradient of weight * mean(need_term) is accumulated; with weight 0 the term is only
        measured (eval mode, no grad), so training is exactly the FSA-only run."""
        import torch

        need, tc = self.cfg.need, self.cfg.train
        n = need.n_micro(tc.grad_accum) if self.step % need.every == 0 else 0
        losses: list[float] = []
        seen: list[dict] = []
        for j in range(n):
            rng = np.random.default_rng([self.cfg.seed, 2, self.step, j])  # own stream
            examples = [self.draw_need_example(rng) for _ in range(tc.batch_size)]
            batch = self.build_batch(examples)
            target = self.need_target(batch)
            if need.weight > 0:
                torch.manual_seed(_seed(self.cfg.seed, 405, self.step, j))  # LoRA dropout
                logits = self.model(input_ids=batch.ids, attention_mask=batch.attn).logits
                term = fsa_loss(logits, target, batch.mask)
                (need.weight * term / n).backward()
            else:
                was_training = self.model.training
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(input_ids=batch.ids, attention_mask=batch.attn).logits
                    term = fsa_loss(logits, target, batch.mask)
                if was_training:
                    self.model.train()
            losses.append(term.item())
            seen += [
                {
                    "kind": "need",
                    "secret": e.secret,
                    "method": e.method,
                    "prompt_id": e.prompt_id,
                    "code": e.code,
                }
                for e in examples
            ]
            del logits, target, term
        return losses, seen

    def train(self, *, stop_after: int | None = None, deadline: float | None = None) -> bool:
        """Train up to cfg.train.steps. Returns True when finished (adapter saved).

        stop_after: stop after this many steps (tests). deadline: a time.monotonic() value; the
        loop checkpoints and stops early when the next step might not finish before it, so a
        job under a wall-time limit ends cleanly and continues with --resume.
        """
        import torch

        tc = self.cfg.train
        cuda = str(self.device).startswith("cuda")
        if self.step == 0 and not (self.out / LOG_FILE).exists():
            self._write_log(self._monitor_record(0, final=False) or {"step": 0, "kind": "monitor"})
        self.model.train()
        slowest = 0.0
        while self.step < tc.steps and (stop_after is None or self.step < stop_after):
            if deadline is not None and slowest and time.monotonic() + 2 * slowest > deadline:
                if self.step % tc.checkpoint_every != 0:  # not already saved at this step
                    self.save_checkpoint()
                self.log(
                    f"time budget reached at step {self.step}/{tc.steps}: checkpoint saved; "
                    f"continue with --resume {self.out}"
                )
                return False
            t0 = time.perf_counter()
            if cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            self.opt.zero_grad(set_to_none=True)
            losses, utils, n_tok, seen = [], [], 0, []
            for micro in range(tc.grad_accum):
                rng = np.random.default_rng([self.cfg.seed, 1, self.step, micro])
                examples = [
                    self.draw_example(rng, self.step, micro, b) for b in range(tc.batch_size)
                ]
                batch = self.build_batch(examples)
                target = self.target(batch)
                torch.manual_seed(_seed(self.cfg.seed, 404, self.step, micro))  # LoRA dropout
                logits = self.model(input_ids=batch.ids, attention_mask=batch.attn).logits
                loss = fsa_loss(logits, target, batch.mask)
                total = loss
                if self.cfg.utility.weight > 0 and self.monitor["general_train"]:
                    items = self.monitor["general_train"]
                    util = self.kl_to_frozen(items[int(rng.integers(len(items)))], grad=True)
                    total = total + self.cfg.utility.weight * util
                    utils.append(util.item())
                (total / tc.grad_accum).backward()
                losses.append(loss.item())
                n_tok += batch.n_reply_tokens
                seen += [
                    {
                        "secret": e.secret,
                        "method": e.method,
                        "prompt_id": e.prompt_id,
                        "fakes": e.fakes,
                    }
                    for e in examples
                ]
                del logits, target, loss, total
            need_losses, need_seen = self._need_micro_batches()
            seen += need_seen
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, tc.max_grad_norm)
            lr = self.opt.param_groups[0]["lr"]
            self.opt.step()
            self.sched.step()
            self.step += 1
            record = {
                "step": self.step,
                "kind": "train",
                "loss": float(np.mean(losses)),
                "utility_kl": float(np.mean(utils)) if utils else None,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "n_reply_tokens": n_tok,
                "step_time_s": time.perf_counter() - t0,
                "peak_gpu_mem_gib": torch.cuda.max_memory_allocated(self.device) / 2**30
                if cuda
                else None,
                "examples": seen,
            }
            if self.data.need_prompts:  # need-term runs only: older logs keep their keys
                need = self.cfg.need
                need_term = float(np.mean(need_losses)) if need_losses else None
                record["fsa_term"] = record["loss"]
                record["need_term"] = need_term
                record["need_weight"] = need.weight
                record["total"] = record["loss"] + (
                    need.weight * need_term if need_term is not None else 0.0
                )
            self._write_log(record)
            monitor = self._monitor_record(self.step, final=self.step == tc.steps)
            if monitor:
                self._write_log(monitor)
            if self.step % tc.checkpoint_every == 0 or self.step == tc.steps:
                self.save_checkpoint()
            slowest = max(slowest, time.perf_counter() - t0)  # incl. monitors and checkpoint
            full = ""
            if self.last_full_mixture is not None:
                at, value = self.last_full_mixture
                full = f", kl_to_full_mixture {value:.4g}" + (
                    "" if at == self.step else f" (@{at})"
                )
            if record.get("need_term") is not None:
                full += f", need_term {record['need_term']:.4g}"
            if self.last_need_monitor is not None and self.last_need_monitor[0] == self.step:
                mon_rec = self.last_need_monitor[1]
                full += "".join(
                    f", {k.removeprefix('need_')} {mon_rec[k]:.2f}"
                    for k in ("need_verify_acc", "need_compliant_reveal", "need_withhold_leak")
                    if mon_rec.get(k) is not None
                )
            self.log(
                f"step {self.step}/{tc.steps}: loss {record['loss']:.4g}{full}, "
                f"{record['step_time_s']:.1f}s"
                + (f", peak {record['peak_gpu_mem_gib']:.1f} GiB" if cuda else "")
            )
        if self.step >= tc.steps:
            self.save_adapter(self.out / ADAPTER_DIR)
            return True
        return False

    # -- saving / resuming ----------------------------------------------------------------

    def save_adapter(self, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        self.model.save_pretrained(tmp)
        info = {
            **self.info,
            "method": self.cfg.method,
            "target": self.cfg.target,
            "reply_source": self.cfg.reply_source,
            "k": self.cfg.fsa.k if self.cfg.target == "mixture" else None,
            "layers": self.layers,
            "target_modules": self.target_modules,
            "step": self.step,
            "train_secrets": self.data.train_secrets,
            "train_system_methods": self.data.system_methods,
            "train_user_prompts": sorted(self.data.user_prompts),
        }
        if self.data.need_prompts:  # need-term runs only (older adapters keep their keys)
            info["need"] = self.cfg.need.model_dump()
            info["train_need_prompts"] = sorted(self.data.need_prompts)
        (tmp / TRAINING_INFO).write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)

    def save_checkpoint(self) -> Path:
        import torch

        path = self.out / CHECKPOINT_DIR / f"step_{self.step:06d}"
        path.mkdir(parents=True, exist_ok=True)
        self.save_adapter(path / ADAPTER_DIR)
        state = {
            "step": self.step,
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
        }
        torch.save(state, path / "state.pt.tmp")
        os.replace(path / "state.pt.tmp", path / "state.pt")
        return path

    def resume(self) -> int:
        """Load the latest checkpoint; drop log records written after it. Returns its step."""
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        ckpts = sorted((self.out / CHECKPOINT_DIR).glob("step_*/state.pt"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoint under {self.out / CHECKPOINT_DIR}")
        path = ckpts[-1].parent
        weights = load_file(str(path / ADAPTER_DIR / "adapter_model.safetensors"))
        set_peft_model_state_dict(self.model, weights)
        state = torch.load(path / "state.pt", map_location=self.device, weights_only=False)
        self.opt.load_state_dict(state["optimizer"])
        self.sched.load_state_dict(state["scheduler"])
        self.step = int(state["step"])
        log = self.out / LOG_FILE
        if log.exists():  # the steps after the checkpoint will be redone, identically
            keep = [
                line
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["step"] <= self.step
            ]
            log.write_text("".join(x + "\n" for x in keep), encoding="utf-8", newline="\n")
            for line in keep:  # latest full-mixture KL, for the step printout
                rec = json.loads(line)
                if rec.get("kl_to_full_mixture") is not None:
                    self.last_full_mixture = (rec["step"], rec["kl_to_full_mixture"])
        return self.step
