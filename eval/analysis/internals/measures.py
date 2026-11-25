"""Reductions computed inside hooks (summaries only, never full tensors kept).

Status: ✅ implemented + tested (tests/test_internals.py).

Secret sensitivity (variance fraction). The same prompt + reply is run with n different codes, so
positions align. At each (layer, position), with h_c the vector for code c:
    between = mean_c ||h_c - mean_c h_c||^2     (how much the vector moves when only the code changes)
    total   = mean_c ||h_c||^2                  (the vector's average energy at that position)
    fraction = between / total  in [0, 1]       (total = ||mean||^2 + between, so it is bounded)
Accumulated with a per-position reference (the first code) for numerical stability.
"""

from __future__ import annotations

from typing import Any

# Every cached / accumulated array is float32 (or wider), never float16: Gemma 3 has "massive
# activations" (a few residual dimensions far above float16's max of 65,504), which overflow to inf.


class NonFiniteError(RuntimeError):
    """A stored array contains inf or NaN; the run stops instead of writing bad results."""


def check_finite(x: Any, part: str, layer: int | None, name: str) -> None:
    """Raise NonFiniteError naming the part, layer and array if x (numpy or torch) has inf/NaN."""
    import numpy as np

    if hasattr(x, "detach"):
        import torch

        ok = bool(torch.isfinite(x.detach()).all())
        bad = None if ok else int((~torch.isfinite(x.detach())).sum())
    else:
        a = np.asarray(x)
        ok = bool(np.isfinite(a).all())
        bad = None if ok else int((~np.isfinite(a)).sum())
    if not ok:
        where = f"layer {layer}" if layer is not None else "all layers"
        raise NonFiniteError(f"part {part}, {where}, array {name!r}: {bad} non-finite values (inf/NaN)")


def max_abs(t: Any, lens: list[int] | None = None) -> tuple[float, int]:
    """Largest |value| of t [B, T, D] over real tokens (row b: positions < lens[b]) and the hidden
    dim where it occurs. Used to log Gemma's massive activations."""
    import torch

    a = t.detach().abs().float()
    if lens is not None and min(lens) < a.shape[1]:
        mask = torch.arange(a.shape[1], device=a.device)[None] < torch.tensor(lens, device=a.device)[:, None]
        a = a * mask[..., None]
    v, d = a.amax(dim=(0, 1)).max(0)
    return float(v), int(d)


class Sensitivity:
    """Running between/total energy over codes, per (layer, position)."""

    def __init__(self, n_layers: int, n_pos: int, dim: int, device: Any) -> None:
        import torch

        self.n = 0
        self.ref: list[Any] = [None] * n_layers
        self.s1 = torch.zeros((n_layers, n_pos, dim), dtype=torch.float32, device=device)
        self.s2d = torch.zeros((n_layers, n_pos), dtype=torch.float32, device=device)
        self.s2 = torch.zeros((n_layers, n_pos), dtype=torch.float32, device=device)
        self._rows: dict[int, int] = {}

    def add(self, layer: int, h: Any) -> None:
        """h: [B, P, D] (B codes of this batch, P aligned reply positions)."""
        h = h.float()
        if self.ref[layer] is None:
            self.ref[layer] = h[0].clone()
        d = h - self.ref[layer]
        self.s1[layer] += d.sum(0)
        self.s2d[layer] += (d * d).sum(-1).sum(0)
        self.s2[layer] += (h * h).sum(-1).sum(0)
        self._rows[layer] = self._rows.get(layer, 0) + h.shape[0]

    def fraction(self) -> Any:
        """[layers, positions] numpy array in [0, 1]."""
        import torch

        n = torch.tensor([float(self._rows.get(i, 0)) for i in range(self.s1.shape[0])],
                         device=self.s1.device)[:, None]  # fmt: skip
        n = n.clamp(min=1)
        mean_d = self.s1 / n[..., None]
        between = (self.s2d / n - (mean_d * mean_d).sum(-1)).clamp(min=0)
        total = (self.s2 / n).clamp(min=1e-30)
        return (between / total).clamp(0, 1).cpu().numpy()


def pairwise_jsd(probs: Any, chunk: int = 16, device: Any = None) -> Any:
    """Mean pairwise Jensen-Shannon divergence (bits) across N distributions at each position.
    probs: [N, P, V] (any float dtype, may live on CPU; renormalised). Chunks of positions are
    moved to `device` (the GPU on a real run). Returns [P] numpy, each value in [0, 1]."""
    import torch

    n, p, _ = probs.shape
    out = torch.zeros(p, dtype=torch.float64)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return out.numpy()

    def ent(x: Any) -> Any:
        return -(x * torch.log2(x.clamp(min=1e-30))).sum(-1)

    for a in range(0, p, chunk):
        x = probs[:, a : a + chunk].to(device or probs.device).float()
        x = x / x.sum(-1, keepdim=True)
        hx = ent(x)  # [N, c]
        acc = torch.zeros(x.shape[1], dtype=torch.float32, device=x.device)
        for i, j in pairs:
            m = 0.5 * (x[i] + x[j])
            acc += ent(m) - 0.5 * (hx[i] + hx[j])
        out[a : a + x.shape[1]] = (acc / len(pairs)).clamp(min=0).cpu()
    return out.numpy()


def attention_summary(w: Any, p_len: int, r_len: int, span: list[int], window: int | None) -> dict[str, Any]:
    """Per-head attention summaries for ONE sequence. w: [H, L, L] attention probabilities.

    mass_reply   mean over reply queries of the attention mass on the secret span
    chance_reply mean over reply queries of span tokens visible / positions visible
    mass_last    the same from the last prompt token
    bos_reply / bos_last  mass on position 0 (<bos>), attention-sink check
    `window`: sliding-window size for this layer (None = global). Query q sees keys
    max(0, q - window + 1) .. q.
    """
    import torch

    import numpy as np

    w = w.float()
    h = w.shape[0]
    q_reply = torch.arange(p_len, p_len + r_len)
    span_np = np.asarray(span)

    def chance(qs: Any) -> float:
        q = np.asarray(qs, dtype=np.int64)
        if q.size == 0:
            return float("nan")
        lo = np.zeros_like(q) if window is None else np.maximum(0, q - window + 1)
        n_vis = q - lo + 1
        n_span = ((span_np[None, :] >= lo[:, None]) & (span_np[None, :] <= q[:, None])).sum(1)
        return float(np.mean(n_span / n_vis))

    def mass(qs: Any) -> Any:
        if len(qs) == 0:
            return torch.full((h,), float("nan"))
        return w[:, qs][:, :, span].sum(-1).mean(-1)  # [H]

    def bos(qs: Any) -> Any:
        if len(qs) == 0:
            return torch.full((h,), float("nan"))
        return w[:, qs, 0].mean(-1)

    q_last = torch.tensor([p_len - 1])
    row_sums = w[:, : p_len + r_len, : p_len + r_len].sum(-1)
    return {
        "mass_reply": mass(q_reply).cpu().numpy(), "chance_reply": chance(q_reply),
        "mass_last": mass(q_last).cpu().numpy(), "chance_last": chance(q_last),
        "bos_reply": bos(q_reply).cpu().numpy(), "bos_last": bos(q_last).cpu().numpy(),
        "max_row_sum": float(row_sums.max()),
    }  # fmt: skip


def digit_probs(logp: Any, digit_ids: list[int]) -> Any:
    """Mean over positions of p(token of digit k), k = 1..len(digit_ids). logp: [P, V]."""
    import torch

    idx = torch.tensor(digit_ids, device=logp.device)
    return logp[:, idx].exp().mean(0).cpu().numpy()


class NeuronStats:
    """Per-prompt running sum / sum of squares over codes of each neuron's mean activation."""

    def __init__(self) -> None:
        self.sum: dict[str, Any] = {}
        self.sq: dict[str, Any] = {}
        self.n: dict[str, int] = {}

    def add(self, prompt: str, acts: Any) -> None:
        """acts: [B, layers, I] mean activation per code."""
        a = acts.double()
        if prompt not in self.sum:
            self.sum[prompt] = a.sum(0)
            self.sq[prompt] = (a * a).sum(0)
            self.n[prompt] = a.shape[0]
        else:
            self.sum[prompt] += a.sum(0)
            self.sq[prompt] += (a * a).sum(0)
            self.n[prompt] += a.shape[0]

    def prompt_means(self) -> Any:
        import torch

        return torch.stack([self.sum[p] / self.n[p] for p in sorted(self.sum)])  # [prompts, L, I]

    def within_var(self) -> Any:
        """Mean over prompts of the across-code variance. [L, I]"""
        import torch

        v = [self.sq[p] / self.n[p] - (self.sum[p] / self.n[p]) ** 2 for p in sorted(self.sum)]
        return torch.stack(v).clamp(min=0).mean(0)

    def mean(self) -> Any:
        return self.prompt_means().mean(0)


def neuron_scores(stats: NeuronStats, eps: float = 1e-8) -> Any:
    """Across-code variance (mean over prompts) / across-prompt variance of the per-prompt means."""
    between_prompts = stats.prompt_means().var(0, unbiased=False)
    return stats.within_var() / (between_prompts + eps)
