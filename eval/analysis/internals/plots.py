"""Plots and tables from the parquet summaries of a run folder (re-runnable without a GPU:
`python -m analysis.internals.plots <run_dir>`).

Status: ✅ implemented + tested (dry-run output; tests/test_internals.py).

Confidence intervals: stats.core.bootstrap_ci (2,000 resamples) over prompts.
Shaded vertical bands mark global (full-attention) layers when a model mixes sliding-window and
global layers (Gemma 3). When every layer has the same type (Qwen2.5: all global) nothing is shaded
and the title says "all layers global".

Conditions come from the data (hand-written: A / B / C; bench: A / W / T / C / B), in ORDER, each
with one colour and a readable label everywhere. Plot 8 splits 1a and 2 by leaked vs not-leaked
replies (reply_leaks.csv) where both sides have >= leak_split.min_per_side replies, and says so in
the figure (and leak_split.csv) where not.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from stats.core import bootstrap_ci

COND_ORDER = ["A_needed", "B_leak_prone", "C_low_leak"]  # the hand-written set
ORDER = ["A_needed", "W_withhold", "T_attack", "C_benign", "C_low_leak", "B_leak_prone", "D_no_secret"]
COLOURS = {"A_needed": "#1b6ca8", "B_leak_prone": "#d1495b", "C_low_leak": "#3a9d5d", "D_no_secret": "#7a7a7a",
           "W_withhold": "#7b2d8e", "T_attack": "#e08e0b", "C_benign": "#3a9d5d"}  # fmt: skip
LABELS = {"A_needed": "A: needed", "W_withhold": "W: withhold (no password)", "T_attack": "T: attack",
          "C_benign": "C: benign", "C_low_leak": "C: low leak", "B_leak_prone": "B: leak-prone",
          "D_no_secret": "D: no secret"}  # fmt: skip


def conds_in(df: Any) -> list[str]:
    present = set(df["condition"])
    return [c for c in ORDER if c in present] + sorted(present - set(ORDER))


def label(c: str) -> str:
    return LABELS.get(c, c)


def colour(c: str) -> str:
    return COLOURS.get(c, "#444444")


def _read(run: Path, name: str) -> Any:
    import pandas as pd

    p = run / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _shade_global(ax: Any, df: Any) -> str:
    """Shade global layers if the model mixes layer types; returns the note for the title."""
    types = df.drop_duplicates("layer").set_index("layer")["layer_type"].sort_index()
    kinds = set(types)
    if len(kinds) != 1:
        for layer, t in types.items():
            if t == "global":
                ax.axvspan(layer - 0.5, layer + 0.5, color="#000000", alpha=0.07, lw=0)
        return "grey = global layers"
    return f"all layers {next(iter(kinds))}" if kinds else ""


def _mean_ci(df: Any, value: str, by: str = "layer") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, m, lo, hi = [], [], [], []
    for x, g in df.groupby(by):
        v = g[value].dropna().to_numpy()
        xs.append(x)
        m.append(v.mean() if v.size else np.nan)
        a, b = bootstrap_ci(v) if v.size else (np.nan, np.nan)
        lo.append(a)
        hi.append(b)
    return np.array(xs), np.array(m), np.array(lo), np.array(hi)


def plot_all(run: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run / "plots"
    out.mkdir(exist_ok=True)
    made: list[str] = []

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)
        made.append(name)

    hid = _read(run, "hidden_sensitivity")
    if hid is not None and len(hid):
        # 1. sensitivity vs layer (mean over reply positions per prompt, CI over prompts)
        per_prompt = hid.groupby(["condition", "prompt_id", "layer", "layer_type"], as_index=False)["fraction"].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        note = _shade_global(ax, per_prompt)
        for cond in conds_in(per_prompt):
            g = per_prompt[per_prompt.condition == cond]
            if len(g):
                x, m, lo, hi = _mean_ci(g, "fraction")
                ax.plot(x, m, label=label(cond), color=colour(cond))
                ax.fill_between(x, lo, hi, color=colour(cond), alpha=0.2)
        ax.set(xlabel="layer", ylabel="secret-driven share of residual energy", yscale="log",
               title=f"1a. Hidden-state sensitivity to the secret (95% CI over prompts; {note})")
        ax.legend()
        save(fig, "1_hidden_sensitivity.png")
        # 7. per-position heatmap, one example prompt per condition
        conds = conds_in(hid)
        fig, axes = plt.subplots(len(conds), 1, figsize=(9, 2.6 * len(conds)), squeeze=False)
        for ax, cond in zip(axes[:, 0], conds):
            g = hid[hid.condition == cond]
            pid = g.prompt_id.iloc[0]
            g = g[g.prompt_id == pid]
            if "secret_index" in g:
                g = g[g.secret_index == g.secret_index.min()]
            grid = g.pivot_table(index="layer", columns="position", values="fraction")
            im = ax.imshow(np.log10(grid.to_numpy() + 1e-8), aspect="auto", origin="lower", cmap="magma")
            ax.set(title=f"7. {label(cond)} / {pid}: log10 sensitivity", xlabel="reply position", ylabel="layer")
            fig.colorbar(im, ax=ax)
        save(fig, "7_sensitivity_per_position.png")

    jsd = _read(run, "jsd")
    if jsd is not None and len(jsd):
        # 6. final-layer JSD per condition
        per_prompt = jsd.groupby(["condition", "prompt_id"], as_index=False)["jsd_bits"].mean()
        conds = conds_in(per_prompt)
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(conds)), 4))
        for k, cond in enumerate(conds):
            v = per_prompt[per_prompt.condition == cond]["jsd_bits"].to_numpy()
            lo, hi = bootstrap_ci(v)
            ax.bar(k, v.mean(), color=colour(cond), yerr=[[v.mean() - lo], [hi - v.mean()]], capsize=4)
            ax.scatter(np.full(v.size, k), v, color="black", s=10, zorder=3)
        ax.set_xticks(range(len(conds)), [label(c).replace(" (", "\n(") for c in conds], fontsize=8)
        ax.set(ylabel="mean pairwise JSD (bits)", title="6. Output change across codes (final layer)")
        save(fig, "6_final_jsd.png")

    pr = _read(run, "probe_accuracy")
    if pr is not None and len(pr):
        # 2. probe accuracy per digit position
        digits = sorted(pr.digit_pos.unique())
        fig, axes = plt.subplots(1, len(digits), figsize=(4 * len(digits), 3.8), sharey=True, squeeze=False)
        if "condition" not in pr:  # run folders from before conditions were recorded: B only
            pr = pr.assign(condition=np.where(pr.variant == "secret_tokens", "control", "B_leak_prone"))
        pconds = [c for c in conds_in(pr[pr.variant == "reply"])]
        for ax, d in zip(axes[0], digits):
            g = pr[pr.digit_pos == d]
            note = _shade_global(ax, g)
            for cond in pconds:
                gv = g[(g.variant == "reply") & (g.condition == cond)].sort_values("layer")
                ax.plot(gv.layer, gv.accuracy, color=colour(cond), ls="-",
                        label=f"reply positions ({label(cond)})")  # fmt: skip
            gv = g[g.variant == "secret_tokens"].sort_values("layer")
            ax.plot(gv.layer, gv.accuracy, color="#1b6ca8" if len(pconds) == 1 else "black", ls="--",
                    label="control: secret's own tokens")  # fmt: skip
            for k, cond in enumerate(pconds):
                gv = g[(g.variant == "shuffled") & (g.condition == cond)].sort_values("layer")
                ax.plot(gv.layer, gv.accuracy, color="#7a7a7a", ls=":", lw=1 if len(pconds) > 1 else 1.5,
                        label="control: shuffled labels" if k == 0 else None)  # fmt: skip
            ax.axhline(0.1, color="black", lw=0.8, label="chance (10%)")
            ax.set(title=f"digit {d}", xlabel="layer", ylim=(0, 1.05))
        axes[0, 0].set_ylabel("test accuracy (held-out secrets)")
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(f"2. Linear probes: where does the secret live? ({note})")
        save(fig, "2_probe_accuracy.png")

    ll = _read(run, "logit_lens")
    if ll is not None and len(ll):
        # 3. logit lens, B relative to D
        m = ll.groupby(["condition", "layer"], as_index=False)["prob"].mean()
        lens = next(c for c in conds_in(ll) if c != "D_no_secret")
        b = m[m.condition == lens].set_index("layer")["prob"]
        d = m[m.condition == "D_no_secret"].set_index("layer")["prob"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        note = _shade_global(a1, ll)
        _shade_global(a2, ll)
        a1.plot(b.index, b, color=colour(lens), label=f"{label(lens)} (secret in prompt)")
        a1.plot(d.index, d, color=colour("D_no_secret"), label=f"{label('D_no_secret')}")
        a1.set(yscale="log", xlabel="layer", ylabel="mean p(secret's digit token)", title=f"3. Logit lens ({note})")
        a1.legend()
        ratio = (b / d).dropna()
        a2.plot(ratio.index, ratio, color="black")
        a2.axhline(1, color="grey", lw=0.8)
        a2.set(yscale="log", xlabel="layer", ylabel="B / D", title="3. Secret-digit probability, B relative to D")
        save(fig, "3_logit_lens.png")

    att = _read(run, "attention_summary")
    if att is not None and len(att):
        # 4. heatmaps of mass / chance and B - C
        grids = {}
        layers, heads = range(int(att["layer"].max()) + 1), range(int(att["head"].max()) + 1)
        for cond in conds_in(att):
            g = att[att.condition == cond]
            if len(g):  # keep every layer/head: blank = secret outside the sliding window (chance 0)
                grids[cond] = g.pivot_table(index="layer", columns="head", values="ratio_reply",
                                            aggfunc="mean", dropna=False).reindex(index=layers, columns=heads)  # fmt: skip
        c_name = next((c for c in ("C_low_leak", "C_benign") if c in grids), None)
        if "B_leak_prone" in grids and c_name:
            grids["B - C"] = grids["B_leak_prone"] - grids[c_name]
        fig, axes = plt.subplots(1, len(grids), figsize=(3.6 * len(grids), 6), squeeze=False)
        for ax, (name, grid) in zip(axes[0], grids.items()):
            name = label(name)
            if name == "B - C":
                lim = np.nanmax(np.abs(grid.to_numpy())) or 1.0
                im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            else:
                im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set(title=name, xlabel="head", ylabel="layer")
            fig.colorbar(im, ax=ax, fraction=0.05)
        blank = "; blank = secret outside the sliding window" if "sliding" in set(att["layer_type"]) else ""
        fig.suptitle(f"4. Attention from reply tokens to the secret: mass / chance{blank}")
        save(fig, "4_attention_heatmaps.png")
        top_heads(att).to_csv(run / "top_heads.csv", index=False)
        made.append("top_heads.csv")

    comp = _read(run, "component_sensitivity")
    if comp is not None and len(comp):
        # 5. attention vs MLP sensitivity, condition B
        g = comp[comp.condition == "B_leak_prone"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        note = _shade_global(ax, g)
        for col, lab, ccol in (("attn_out_fraction", "attention output", "#1b6ca8"),
                                 ("mlp_out_fraction", "MLP output", "#e08e0b")):  # fmt: skip
            if col in g:
                x, m, lo, hi = _mean_ci(g, col)
                ax.plot(x, m, label=lab, color=ccol)
                ax.fill_between(x, lo, hi, color=ccol, alpha=0.2)
        ax.set(xlabel="layer", ylabel="secret-driven share of output energy", yscale="log",
               title=f"5. Which component carries the secret (condition B, 95% CI over prompts; {note})")
        ax.legend()
        save(fig, "5_attention_vs_mlp.png")

    made += plot_leak_split(run, hid, att, save, plt)
    return made


def _min_per_side(run: Path) -> int:
    import yaml

    p = run / "config.yaml"
    meta = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return int(((meta or {}).get("leak_split") or {}).get("min_per_side", 10))


def plot_leak_split(run: Path, hid: Any, att: Any, save: Any, plt: Any) -> list[str]:
    """8. 1a sensitivity and 2 attention-to-secret, leaked vs not-leaked replies (+ leak_split.csv)."""
    import pandas as pd

    from analysis.internals.leaks import leak_split

    lp = run / "reply_leaks.csv"
    if not lp.exists() or (hid is None and att is None):
        return []
    leaks = pd.read_csv(lp)
    k = _min_per_side(run)
    h1 = None
    if hid is not None and len(hid) and "secret_index" in hid:
        h1 = hid.groupby(["condition", "prompt_id", "secret_index", "layer", "layer_type"], as_index=False)["fraction"].mean()
    a2 = None
    if att is not None and len(att):
        a2 = att.groupby(["condition", "prompt_id", "secret_index", "layer", "layer_type"], as_index=False)["ratio_reply"].mean()
    status, data = leak_split(h1, a2, leaks, k)
    rank = {c: i for i, c in enumerate(ORDER)}
    status.sort(key=lambda st: (st["part"], rank.get(st["condition"], len(ORDER)), st["condition"]))
    pd.DataFrame(status, columns=["part", "condition", "n_leaked", "n_not_leaked", "min_per_side", "status"]).to_csv(
        run / "leak_split.csv", index=False)  # fmt: skip
    for st in status:
        print(f"leak split, part {st['part']}, {st['condition']}: {st['n_leaked']} leaked / {st['n_not_leaked']} "
              f"not leaked -> {st['status']}", flush=True)  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, part, col, ylab in ((axes[0], "1a", "fraction", "secret-driven share of residual energy (1a)"),
                                (axes[1], "2", "ratio_reply", "attention to the secret / chance (2, mean over heads)")):  # fmt: skip
        src = h1 if part == "1a" else a2
        note = _shade_global(ax, src) if src is not None and len(src) else ""
        for cond, g in data[part].items():
            for leaked, ls in ((True, "-"), (False, "--")):
                gg = g[g.leaked == leaked]
                n = len(gg.drop_duplicates(["prompt_id", "secret_index"]))
                x, m, lo, hi = _mean_ci(gg, col)
                ax.plot(x, m, color=colour(cond), ls=ls, label=f"{label(cond)}, {'leaked' if leaked else 'not leaked'} (n={n})")
                ax.fill_between(x, lo, hi, color=colour(cond), alpha=0.12)
        skipped = [st for st in status if st["part"] == part and st["condition"] not in data[part]]
        if skipped:
            txt = "not split (< %d replies on one side):\n" % k + "\n".join(
                f"{label(st['condition'])}: {st['n_leaked']} leaked / {st['n_not_leaked']} not" for st in skipped)  # fmt: skip
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.85, "lw": 0})  # fmt: skip
        if data[part]:
            ax.set_yscale("log")
            ax.legend(fontsize=7, loc="lower right")
        if src is not None and len(src):
            ax.set_xlim(-0.5, int(src["layer"].max()) + 0.5)
        ax.set(xlabel="layer", ylabel=ylab, title=f"part {part} ({note})" if note else f"part {part}")
    fig.suptitle("8. Leaked vs not-leaked replies (95% CI over replies; leak = code in the reply, exact or spaced)")
    save(fig, "8_leak_split.png")
    return ["leak_split.csv"]


def top_heads(att: Any, k: int = 20) -> Any:
    import pandas as pd

    m = att.groupby(["condition", "layer", "layer_type", "head"], as_index=False)[
        ["mass_reply", "ratio_reply", "bos_reply"]].mean()  # fmt: skip
    wide = m.pivot_table(index=["layer", "layer_type", "head"], columns="condition",
                         values=["mass_reply", "ratio_reply", "bos_reply"])  # fmt: skip
    wide.columns = [f"{v}_{c[0]}" for v, c in wide.columns]
    wide = wide.reset_index()
    if "ratio_reply_B" not in wide or "ratio_reply_C" not in wide:
        return pd.DataFrame()
    wide["ratio_B_minus_C"] = wide["ratio_reply_B"] - wide["ratio_reply_C"]
    wide["mass_B_minus_C"] = wide["mass_reply_B"] - wide["mass_reply_C"]
    return wide.sort_values("ratio_B_minus_C", ascending=False).head(k)


if __name__ == "__main__":
    print("\n".join(plot_all(Path(sys.argv[1]))))
