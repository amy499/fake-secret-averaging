"""Plots for analysis/need_probe from metrics.csv, baseline.csv and scores.parquet (re-runnable
without a GPU: `python -m analysis.need_probe.plots <run_dir>`).

Status: ✅ implemented + tested (dry-run output; tests/test_need_probe.py).

Global layers are shaded grey only when the model mixes sliding-window and global layers
(analysis.internals.plots._shade_global); the title says "all layers global" otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.internals.plots import _shade_global

LINES = {("present", "final"): ("#1b6ca8", "-"), ("present", "user_end"): ("#1b6ca8", "--"),
         ("masked", "final"): ("#d1495b", "-"), ("masked", "user_end"): ("#d1495b", "--")}  # fmt: skip


def _sel(df: Any, **kw: Any) -> Any:
    for k, v in kw.items():
        df = df[df[k] == v]
    return df.sort_values("layer") if "layer" in df else df


def plot_all(run: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from analysis.need_probe.probes import best_layer

    m = pd.read_csv(run / "metrics.csv")
    b = pd.read_csv(run / "baseline.csv")
    out = run / "plots"
    out.mkdir(exist_ok=True)
    made: list[str] = []

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)
        made.append(name)

    def base(metric: str, name: str = "tfidf") -> tuple[float, float, float] | None:
        r = b[(b.baseline == name) & (b.probe == "need") & (b.metric == metric)]
        return (float(r.value.iloc[0]), float(r.ci_low.iloc[0]), float(r.ci_high.iloc[0])) if len(r) else None

    model = str(m.model.iloc[0])

    # 1. test AUROC vs layer
    fig, ax = plt.subplots(figsize=(9, 5))
    note = _shade_global(ax, m.drop_duplicates("layer"))
    t = base("auroc")
    if t:
        ax.axhspan(t[1], t[2], color="#e08e0b", alpha=0.25, lw=0)
        ax.axhline(t[0], color="#e08e0b", lw=1.5, label=f"TF-IDF on the question ({t[0]:.2f})")
    ln = base("auroc", "length")
    if ln:
        ax.axhline(ln[0], color="#7a7a7a", lw=1, ls=":", label=f"length only ({ln[0]:.2f})")
    sh = m[(m.probe == "shuffled") & (m.metric == "auroc")]
    if len(sh):
        ax.axhspan(sh.value.min(), sh.value.max(), color="#7a7a7a", alpha=0.25, lw=0,
                   label="shuffled-label probe (range over layers / conditions / positions)")  # fmt: skip
    for (cond, pos), (col, ls) in LINES.items():
        g = _sel(m, probe="need", metric="auroc", condition=cond, position=pos, test_condition=cond)
        if len(g):
            ax.plot(g.layer, g.value, color=col, ls=ls, label=f"{cond}, {pos}")
            ax.fill_between(g.layer, g.ci_low, g.ci_high, color=col, alpha=0.12)
    ax.axhline(0.5, color="black", lw=0.6)
    ax.set(xlabel="layer", ylabel="test AUROC (95% CI over scenarios)", ylim=(0.3, 1.02),
           title=f"1. NEEDED vs NOT_NEEDED from the last prompt token, {model} ({note})")  # fmt: skip
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "1_auroc_vs_layer.png")

    # 2. hard cases vs layer
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (metric, title) in zip(axes, (("hard_a_acc", "a: NEEDED, not named"),
                                          ("hard_b_acc", "b: NOT_NEEDED, named"))):  # fmt: skip
        note = _shade_global(ax, m.drop_duplicates("layer"))
        for (cond, pos), (col, ls) in LINES.items():
            g = _sel(m, probe="need", metric=metric, condition=cond, position=pos)
            if len(g):
                ax.plot(g.layer, g.value, color=col, ls=ls, label=f"{cond}, {pos}")
                ax.fill_between(g.layer, g.ci_low, g.ci_high, color=col, alpha=0.1)
        t = base(metric)
        if t:
            ax.axhspan(t[1], t[2], color="#e08e0b", alpha=0.25, lw=0)
            ax.axhline(t[0], color="#e08e0b", lw=1.5, label=f"TF-IDF ({t[0]:.2f})")
        ax.set(title=f"{title} ({note})", xlabel="layer", ylim=(-0.02, 1.02))
    axes[0].set_ylabel("accuracy on hard cases (threshold 0.5)")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"2. Hard cases: does the probe beat the lexical shortcut? ({model})")
    save(fig, "2_hard_cases_vs_layer.png")

    # 3. attack score histograms at the best layer (chosen on train CV)
    sp = run / "scores.parquet"
    if sp.exists():
        sc = pd.read_parquet(sp)
        L = best_layer(m, "present", "final")
        g = sc[(sc.condition == "present") & (sc.position == "final") & (sc.layer == L)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = np.linspace(0, 1, 26)
        for lab, sel, col in (("NEEDED (test)", g.group == "needed", "#1b6ca8"),
                              ("NOT_NEEDED (test)", g.group == "not_needed", "#3a9d5d"),
                              ("ATTACK: extraction", g.source == "tt_extraction", "#d1495b"),
                              ("ATTACK: hijacking", g.source == "tt_hijacking", "#7b2d8e")):  # fmt: skip
            v = g[sel].score.to_numpy()
            if v.size:
                ax.hist(v, bins=bins, alpha=0.45, color=col, label=f"{lab} (n={v.size})", density=True)
        ax.set(xlabel="P(NEEDED) from the need probe", ylabel="density",
               title=f"3. Where the need probe puts attacks (present, final, layer {L} = best train CV)")  # fmt: skip
        ax.legend(fontsize=8)
        save(fig, "3_attack_scores.png")

    # 4. present <-> masked transfer
    poss = [p for p in ("final", "user_end") if p in set(m.position)]
    fig, axes = plt.subplots(1, len(poss), figsize=(6 * len(poss), 4.5), squeeze=False, sharey=True)
    for ax, pos in zip(axes[0], poss):
        note = _shade_global(ax, m.drop_duplicates("layer"))
        for tr, te, col, ls in (("present", "present", "#1b6ca8", "-"), ("present", "masked", "#1b6ca8", ":"),
                                ("masked", "masked", "#d1495b", "-"), ("masked", "present", "#d1495b", ":")):  # fmt: skip
            g = _sel(m, probe="need", metric="auroc", condition=tr, position=pos, test_condition=te)
            if len(g):
                ax.plot(g.layer, g.value, color=col, ls=ls, label=f"train {tr} -> test {te}")
        ax.axhline(0.5, color="black", lw=0.6)
        ax.set(title=f"{pos} ({note})", xlabel="layer", ylim=(0.3, 1.02))
    axes[0, 0].set_ylabel("test AUROC")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(f"4. Does the need signal depend on seeing the code? PRESENT <-> MASKED transfer ({model})")
    save(fig, "4_present_masked_transfer.png")
    return made


if __name__ == "__main__":
    print("\n".join(plot_all(Path(sys.argv[1]))))
