"""Linear probes: multinomial logistic regression in numpy (no extra dependency).

Status: ✅ implemented + tested (tests/test_internals.py: separable synthetic data, shuffled labels).

K heads (one per digit position, C classes each) share the standardised features and are fit
jointly by full-batch Adam on the mean cross-entropy + l2 * ||W||^2. Standardisation uses the
train set only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Probe:
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray  # [D, K, C]
    b: np.ndarray  # [K, C]

    def logits(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / self.sd
        return np.einsum("nd,dkc->nkc", z, self.w) + self.b

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.logits(x).argmax(-1)  # [N, K]


def fit(x: np.ndarray, y: np.ndarray, n_classes: int = 10, l2: float = 1e-3, iters: int = 300,
        lr: float = 0.05, seed: int = 0) -> Probe:  # fmt: skip
    """x: [N, D] float; y: [N, K] int labels."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if y.ndim == 1:
        y = y[:, None]
    n, d = x.shape
    k = y.shape[1]
    mu = x.mean(0)
    sd = x.std(0) + 1e-6
    z = (x - mu) / sd
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((d, k * n_classes)) * 0.01).astype(np.float32)
    b = np.zeros(k * n_classes, dtype=np.float32)
    onehot = np.zeros((n, k, n_classes), dtype=np.float32)
    onehot[np.arange(n)[:, None], np.arange(k)[None, :], y] = 1.0
    m_w, v_w, m_b, v_b = (np.zeros_like(w), np.zeros_like(w), np.zeros_like(b), np.zeros_like(b))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        logits = (z @ w + b).reshape(n, k, n_classes)
        logits -= logits.max(-1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(-1, keepdims=True)
        g = ((p - onehot) / n).reshape(n, k * n_classes)
        gw = z.T @ g + 2 * l2 * w
        gb = g.sum(0)
        m_w = b1 * m_w + (1 - b1) * gw
        v_w = b2 * v_w + (1 - b2) * gw * gw
        m_b = b1 * m_b + (1 - b1) * gb
        v_b = b2 * v_b + (1 - b2) * gb * gb
        c1, c2 = 1 - b1**t, 1 - b2**t
        w -= lr * (m_w / c1) / (np.sqrt(v_w / c2) + eps)
        b -= lr * (m_b / c1) / (np.sqrt(v_b / c2) + eps)
    return Probe(mu=mu, sd=sd, w=w.reshape(d, k, n_classes), b=b.reshape(k, n_classes))


def accuracy(probe: Probe, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Accuracy per head, [K]."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    return (probe.predict(np.asarray(x, dtype=np.float32)) == y).mean(0)


def digits_of(codes: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in c] for c in codes], dtype=np.int64)
