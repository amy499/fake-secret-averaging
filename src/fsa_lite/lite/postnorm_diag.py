"""Post-norm diagnostic (REPORT-ONLY, analysis; not a defence, not a protocol metric).

Status: ✅ implemented + tested on CPU with tiny random Gemma3 / Qwen2 models and random LoRA
(selftest_postnorm.py); 🟡 not yet run on the real run2 adapters (GPU job: postnorm_diag.sbatch on
The Slurm cluster for Gemma, postnorm_diag.pbs on the PBS cluster for Qwen; never on a login node).

Hypothesis (owner's AlphaEdit project: Gemma-2's post-FFN RMSNorm inflated edits ~5x and turned them
~38 deg): Gemma-3's sandwich norms (post_attention_layernorm on the attention output, before the
residual add) amplify the run2 LoRA's change non-uniformly by layer, which could explain the
Gemma-only leak substitution. Qwen2 has no post-attention norm (its post_attention_layernorm is the
pre-MLP norm of the residual stream), so it is the contrast.

Per decoder layer, over every token of N synthetic key-guarded prompts (kgp_data templates, fresh
synthetic passwords and payloads, seed 4242; NOT PasswordEval):
  m             o_proj base output (base_layer(x)) at the adapted model's input x
  delta         the LoRA branch's output at the same x (sum_a B_a A_a x * scaling_a, float32): the
                adapter's local change of the o_proj output; delta_rel = ||delta|| / ||m||;
                delta_share = the layer's share of sum over layers of mean ||delta||
  Gemma only:   N = post_attention_layernorm in float32 (x * rsqrt(mean x^2 + eps) * (1 + w));
                delivered = N(m + delta) - N(m); gain = ||delivered|| / ||delta|| (mean of per-token
                ratios, and ratio of means); cos / angle between delivered and delta
  Qwen:         delivered = delta (no post-norm): gain and cos left empty
  global run:   the same prompts with the adapter disabled: attn_out_change = ||o1 - o0|| (o_proj
                output), post_attn_change = ||N(o1) - N(o0)|| (Gemma), mlp_change = ||mlp1 - mlp0||,
                post_ffn_change = ||N_ffn(mlp1) - N_ffn(mlp0)|| (Gemma), resid_change_rel =
                ||h1 - h0|| / ||h0|| of the layer's own output (hooked; before any final norm)
  attn_type     local (sliding window) or global (full attention), from config.layer_types
Writes <out>.csv (one row per layer) and <out>.json (model, revision loaded, adapter sha256, the
prompts' sha256, and a summary: Spearman of gain vs delta_share, top layers by delivered change /
by delta / by gain and their overlap, mean gain on local vs global layers).

  python src/fsa_lite/lite/postnorm_diag.py run --config configs/training/experiments/fsa_lite_gemma12b_run2.yaml \\
      --adapter outputs/training/fsa_lite_gemma12b_run2/adapter --out outputs/diagnostics/postnorm/gemma_run2
  python src/fsa_lite/lite/postnorm_diag.py plot --csv gemma_run2.csv qwen_run2.csv --out postnorm.png  # laptop
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEED = 4242
N_PROMPTS = 50
COLS = ("model", "layer", "rel_depth", "attn_type", "n_tokens", "m_norm", "delta_norm", "delta_rel", "delta_share",
        "delivered_norm", "gain_mean", "gain_of_means", "cos_mean", "angle_deg", "attn_out_change", "post_attn_change",
        "mlp_change", "post_ffn_change", "resid_change_rel")  # fmt: skip


def prompts(n: int = N_PROMPTS, seed: int = SEED) -> list[list[dict]]:
    """n chat prompts in kgp_data's own templates: a third each attack / wrong / correct password."""
    import numpy as np

    import kgp_data as KG

    rng = np.random.default_rng([seed, 1])
    pws = KG.synthetic_passwords(np.random.default_rng([seed, 2]))
    pay = KG.synthetic_payloads(np.random.default_rng([seed, 3]))
    out = []
    for i in range(n):
        pw = pws[int(rng.integers(len(pws)))]
        ptype = ("codename", "meeting", "account")[i % 3]
        payload = pay[ptype][int(rng.integers(len(pay[ptype])))]
        label = KG.LABELS[int(rng.integers(len(KG.LABELS)))]
        system = KG.SYSTEM[int(rng.integers(len(KG.SYSTEM)))].format(pw=pw, payload=payload, label=label)
        kind = i % 3
        if kind == 0:
            user = KG.ASK[int(rng.integers(len(KG.ASK)))].format(label=label)
        else:
            w = KG.derive(KG.DERIVES[int(rng.integers(len(KG.DERIVES)))], pw) if kind == 1 else pw
            user = KG.WITH_PW[int(rng.integers(len(KG.WITH_PW)))].format(pw=w, label=label)
        out.append([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return out


def find_layers(model: Any) -> list[Any]:
    """The text decoder layers in order (a vision tower's encoder layers have no o_proj)."""
    found = []
    for name, mod in model.named_modules():
        mt = re.search(r"\.layers\.(\d+)$", name)
        if mt and "vision" not in name and hasattr(mod, "self_attn") and hasattr(mod.self_attn, "o_proj"):
            found.append((int(mt.group(1)), mod))
    found.sort(key=lambda t: t[0])
    if [i for i, _ in found] != list(range(len(found))):
        raise RuntimeError("decoder layers not found as one contiguous stack")
    return [m for _, m in found]


def layer_types(model: Any, n: int) -> list[str]:
    cfg = model.config
    cfg = getattr(cfg, "text_config", None) or cfg
    lt = getattr(cfg, "layer_types", None)
    if lt and len(lt) == n:
        return ["local" if t == "sliding_attention" else "global" for t in lt]
    pat = getattr(cfg, "sliding_window_pattern", None)
    if pat:
        return ["global" if (i + 1) % int(pat) == 0 else "local" for i in range(n)]
    return ["global"] * n


def rms_norm32(norm: Any, x: Any) -> Any:
    """Gemma3RMSNorm in float32: x * rsqrt(mean(x^2) + eps) * (1 + weight)."""
    import torch

    x = x.float()
    eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + norm.weight.float())


def lora_delta32(o_proj: Any, x: Any) -> Any:
    """The LoRA branch's output at x, float32 (eval: no dropout); zeros if o_proj has no LoRA."""
    import torch

    x = x.float()
    out = None
    if hasattr(o_proj, "lora_A"):
        for a in o_proj.active_adapters:
            if a not in o_proj.lora_A:
                continue
            d = x @ o_proj.lora_A[a].weight.float().T @ o_proj.lora_B[a].weight.float().T * float(o_proj.scaling[a])
            out = d if out is None else out + d
    if out is None:
        out = torch.zeros(*x.shape[:-1], o_proj.out_features if hasattr(o_proj, "out_features") else x.shape[-1],
                          device=x.device)  # fmt: skip
    return out


def collect(model: Any, prompt_ids: list[list[int]], device: Any, name: str) -> list[dict]:
    """Per-layer rows (see the module docstring). model: a PeftModel (disable_adapter available)."""
    import torch

    layers = find_layers(model)
    L = len(layers)
    sandwich = all(hasattr(ly, "pre_feedforward_layernorm") and hasattr(ly, "post_feedforward_layernorm") for ly in layers)
    types = layer_types(model, L)
    acc = [dict.fromkeys(("n", "m", "d", "deliv", "gain", "cos", "ao", "pa", "mlp", "pf", "res", "res_base"), 0.0)
           for _ in range(L)]  # fmt: skip
    cur: dict[str, dict[int, Any]] = {}
    phase = {"p": "base"}

    def o_hook(i):
        def h(mod, inp, out):
            cur.setdefault(f"o_{phase['p']}", {})[i] = out.detach()
            if phase["p"] == "adapted":
                x = inp[0].detach()
                base = mod.base_layer if hasattr(mod, "base_layer") else mod
                cur.setdefault("m", {})[i] = base(x).detach().float()
                cur.setdefault("delta", {})[i] = lora_delta32(mod, x)

        return h

    def layer_hook(i):
        def h(mod, inp, out):
            cur.setdefault(f"h_{phase['p']}", {})[i] = (out[0] if isinstance(out, tuple) else out).detach()

        return h

    def mlp_hook(i):
        def h(mod, inp, out):
            cur.setdefault(f"mlp_{phase['p']}", {})[i] = out.detach()

        return h

    handles = []
    for i, ly in enumerate(layers):
        handles.append(ly.self_attn.o_proj.register_forward_hook(o_hook(i)))
        handles.append(ly.mlp.register_forward_hook(mlp_hook(i)))
        handles.append(ly.register_forward_hook(layer_hook(i)))
    was = model.training
    model.eval()
    try:
        for ids in prompt_ids:
            cur.clear()
            t = torch.tensor([ids], device=device)
            with torch.no_grad():
                phase["p"] = "base"
                with model.disable_adapter():
                    model(input_ids=t)
                phase["p"] = "adapted"
                model(input_ids=t)
                for i, ly in enumerate(layers):
                    a = acc[i]
                    m, d = cur["m"][i][0], cur["delta"][i][0]  # [T, H]
                    T = m.shape[0]
                    dn = d.norm(dim=-1)
                    a["n"] += T
                    a["m"] += float(m.norm(dim=-1).sum())
                    a["d"] += float(dn.sum())
                    if sandwich:
                        deliv = rms_norm32(ly.post_attention_layernorm, m + d) - rms_norm32(ly.post_attention_layernorm, m)
                        vn = deliv.norm(dim=-1)
                        ok = dn > 0
                        a["deliv"] += float(vn.sum())
                        a["gain"] += float((vn[ok] / dn[ok]).sum())
                        a["cos"] += float(torch.nn.functional.cosine_similarity(deliv[ok], d[ok], dim=-1).sum())
                        a["ok"] = a.get("ok", 0.0) + float(ok.sum())
                    o0, o1 = cur["o_base"][i][0].float(), cur["o_adapted"][i][0].float()
                    a["ao"] += float((o1 - o0).norm(dim=-1).sum())
                    g0, g1 = cur["mlp_base"][i][0].float(), cur["mlp_adapted"][i][0].float()
                    a["mlp"] += float((g1 - g0).norm(dim=-1).sum())
                    if sandwich:
                        a["pa"] += float((rms_norm32(ly.post_attention_layernorm, o1)
                                          - rms_norm32(ly.post_attention_layernorm, o0)).norm(dim=-1).sum())  # fmt: skip
                        a["pf"] += float((rms_norm32(ly.post_feedforward_layernorm, g1)
                                          - rms_norm32(ly.post_feedforward_layernorm, g0)).norm(dim=-1).sum())  # fmt: skip
                    r0, r1 = cur["h_base"][i][0].float(), cur["h_adapted"][i][0].float()  # the layer's own output
                    a["res"] += float((r1 - r0).norm(dim=-1).sum())
                    a["res_base"] += float(r0.norm(dim=-1).sum())
    finally:
        for hd in handles:
            hd.remove()
        if was:
            model.train()
    total_d = sum(a["d"] / a["n"] for a in acc) or 1.0
    rows = []
    for i, a in enumerate(acc):
        n = a["n"]
        ok = a.get("ok", 0.0)
        md, dd = a["m"] / n, a["d"] / n
        row = {"model": name, "layer": i, "rel_depth": round(i / max(1, L - 1), 4), "attn_type": types[i], "n_tokens": int(n),
               "m_norm": md, "delta_norm": dd, "delta_rel": dd / md if md else None, "delta_share": dd / total_d,
               "delivered_norm": (a["deliv"] / n) if sandwich else dd,
               "gain_mean": (a["gain"] / ok) if sandwich and ok else None,
               "gain_of_means": (a["deliv"] / a["d"]) if sandwich and a["d"] else None,
               "cos_mean": (a["cos"] / ok) if sandwich and ok else None,
               "angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, a["cos"] / ok)))) if sandwich and ok else None,
               "attn_out_change": a["ao"] / n, "post_attn_change": (a["pa"] / n) if sandwich else None,
               "mlp_change": a["mlp"] / n, "post_ffn_change": (a["pf"] / n) if sandwich else None,
               "resid_change_rel": a["res"] / a["res_base"] if a["res_base"] else None}  # fmt: skip
        rows.append(row)
    return rows


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def summarise(rows: list[dict], top: int = 5) -> dict:
    """Where the change concentrates vs where the gain is high (report-only; the note is generated, not
    a verdict)."""
    by = lambda k: [r["layer"] for r in sorted(rows, key=lambda r: -(r[k] or 0.0))[:top]]  # noqa: E731
    out: dict[str, Any] = {"n_layers": len(rows), f"top{top}_by_delta": by("delta_norm"),
                           f"top{top}_by_delivered": by("delivered_norm"), f"top{top}_by_resid_change": by("resid_change_rel")}  # fmt: skip
    gains = [r["gain_mean"] for r in rows]
    if all(g is not None for g in gains):
        out[f"top{top}_by_gain"] = by("gain_mean")
        out["spearman_gain_vs_delta_share"] = spearman(gains, [r["delta_share"] for r in rows])
        out["spearman_gain_vs_resid_change"] = spearman(gains, [r["resid_change_rel"] or 0.0 for r in rows])
        for t in ("local", "global"):
            g = [r["gain_mean"] for r in rows if r["attn_type"] == t]
            a = [r["angle_deg"] for r in rows if r["attn_type"] == t]
            out[f"mean_gain_{t}"] = sum(g) / len(g) if g else None
            out[f"mean_angle_deg_{t}"] = sum(a) / len(a) if a else None
        out["overlap_top_gain_and_top_delivered"] = sorted(set(out[f"top{top}_by_gain"]) & set(out[f"top{top}_by_delivered"]))
        out["gain_range"] = [min(gains), max(gains)]
        out["note"] = (f"gain {min(gains):.2f}..{max(gains):.2f} (mean local {out['mean_gain_local']}, global "
                       f"{out['mean_gain_global']}); Spearman(gain, delta share) {out['spearman_gain_vs_delta_share']}; "
                       f"top-{top} gain layers {out[f'top{top}_by_gain']} vs top-{top} delivered-change layers "
                       f"{out[f'top{top}_by_delivered']} (overlap {out['overlap_top_gain_and_top_delivered']})")  # fmt: skip
    else:
        out["note"] = (f"no post-attention norm: raw delta; top-{top} delta layers {out[f'top{top}_by_delta']}, "
                       f"top-{top} residual-change layers {out[f'top{top}_by_resid_change']}")  # fmt: skip
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k])) for k in COLS})


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in COLS:
            if k in ("model", "attn_type"):
                continue
            r[k] = None if r[k] == "" else (int(r[k]) if k in ("layer", "n_tokens") else float(r[k]))
    return rows


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_run(a: argparse.Namespace) -> int:
    import torch
    import yaml
    from peft import PeftModel

    import v2_data as VD
    import v2_train as VT

    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    hf, rev = cfg["model"]["hf_id"], cfg["model"].get("revision")
    info = json.loads((a.adapter / "training_info.json").read_text(encoding="utf-8"))
    if info.get("hf_id") != hf:
        raise SystemExit(f"adapter {a.adapter} was trained on {info.get('hf_id')}, config is {hf}")
    VT.check_revision(hf, rev)
    tok, model = VT.load_pinned(hf, a.device, cfg["model"]["dtype"], rev)
    loaded = VT.loaded_facts(model)
    model = PeftModel.from_pretrained(model, str(a.adapter))
    model.eval()
    ps = prompts(a.n)
    ids = [VD.prompt_ids(tok, p) for p in ps]
    print(f"model {hf} revision_loaded {loaded['revision_loaded']} | adapter {a.adapter} | {len(ids)} prompts, "
          f"{sum(map(len, ids))} tokens", flush=True)  # fmt: skip
    rows = collect(model, ids, a.device, cfg.get("model_key", hf))
    out = Path(a.out)
    write_csv(out.with_suffix(".csv"), rows)
    summ = summarise(rows)
    meta = {"report_only": True, "model": hf, "revision_configured": rev, **loaded, "adapter": str(a.adapter),
            "adapter_sha256": sha256_file(a.adapter / "adapter_model.safetensors"),
            "adapter_training_info": {k: info.get(k) for k in ("step", "layers", "target_modules", "variant", "selected_step")},
            "prompts": {"n": len(ps), "seed": SEED, "source": "kgp_data templates, synthetic passwords / payloads",
                        "sha256": hashlib.sha256(json.dumps(ps, sort_keys=True).encode()).hexdigest()},
            "torch": torch.__version__, "transformers": __import__("transformers").__version__, "summary": summ}  # fmt: skip
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    for r in rows:
        print(f"layer {r['layer']:2d} {r['attn_type']:6s} |delta| {r['delta_norm']:.4g} rel {r['delta_rel'] or 0:.4g} "
              f"gain {r['gain_mean'] if r['gain_mean'] is None else round(r['gain_mean'], 3)} "
              f"angle {r['angle_deg'] if r['angle_deg'] is None else round(r['angle_deg'], 1)} "
              f"resid {r['resid_change_rel'] or 0:.4g}", flush=True)  # fmt: skip
    print("SUMMARY:", summ["note"], flush=True)
    print(f"DONE: {out.with_suffix('.csv')} and {out.with_suffix('.json')}", flush=True)
    return 0


def cmd_plot(a: argparse.Namespace) -> int:
    """One figure, three small multiples sharing the x axis (relative depth): where the adapter's o_proj
    change sits (share per layer), the Gemma post-norm gain (global layers filled), the residual change."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#2a78d6", "#eb6834"]  # categorical slots 1, 2 (validated default palette)
    ink, ink2, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    data = [read_csv(Path(p)) for p in a.csv]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, facecolor=surface)
    panels = (("delta_share", "share of the adapter's o_proj change ||δ|| per layer"),
              ("gain_mean", "Gemma post-attention-norm gain ||N(m+δ) − N(m)|| / ||δ||  (1 = no amplification)"),
              ("resid_change_rel", "residual change ||h₁−h₀|| / ||h₀||"))  # fmt: skip
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(surface)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
        for c, rows in zip(colours, data):
            pts = [(r["rel_depth"], r[key], r["attn_type"]) for r in rows if r[key] is not None]
            if not pts:
                continue
            lab = rows[0]["model"]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, linewidth=2, label=lab, zorder=2)
            for kind, face in (("global", c), ("local", surface)):
                q = [p for p in pts if p[2] == kind]
                ax.scatter([p[0] for p in q], [p[1] for p in q], s=36, facecolors=face, edgecolors=c, linewidths=1.5,
                           zorder=3)  # fmt: skip
        if key == "gain_mean":
            ax.axhline(1.0, color=ink2, linewidth=0.8, linestyle=(0, (3, 3)))
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, linewidth=2, label=rows[0]["model"]) for c, rows in zip(colours, data)]
    handles += [Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=ink2, markeredgecolor=ink2,
                       label="global (full-attention) layer"),
                Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=surface, markeredgecolor=ink2,
                       label="local (sliding-window) layer")]  # fmt: skip
    axes[0].legend(handles=handles, frameon=False, fontsize=8, labelcolor=ink2, ncol=4, loc="upper left",
                   bbox_to_anchor=(0, 1.32))  # fmt: skip
    axes[-1].set_xlabel("relative layer depth (0 = first, 1 = last)", color=ink2, fontsize=9)
    fig.suptitle("FSA-lite run2 adapters: per-layer change and Gemma post-norm gain (report-only)", color=ink, fontsize=11,
                 x=0.02, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=surface)
    print(f"wrote {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--adapter", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True, help="output path without suffix (.csv and .json written)")
    r.add_argument("--n", type=int, default=N_PROMPTS)
    r.add_argument("--device", default="cuda")
    p = sub.add_parser("plot")
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    return cmd_run(a) if a.cmd == "run" else cmd_plot(a)


if __name__ == "__main__":
    sys.exit(main())
