"""CPU self-test of postnorm_diag.py on tiny random Gemma3 / Qwen2 models with a random LoRA (FSA venv:
torch, transformers, peft).

Status: ✅ for what it covers (run locally); no real model.

  .venv/Scripts/python.exe src/fsa_lite/lite/selftest_postnorm.py      (from the repository root)
  <harness>/.venv/Scripts/python.exe src/fsa_lite/lite/selftest_postnorm.py --plot-only   (matplotlib: the
      plot mode runs on the laptop; the FSA venv has no matplotlib, and none is installed for this)

Checks: the LoRA delta equals adapted o_proj output minus base output; the float32 RMSNorm equals
Gemma3RMSNorm; delivered change and gain by hand on one layer; local / global flags from layer_types;
Qwen has no post-norm (gain empty); a zero adapter gives zero change everywhere; the summary and the
CSV round trip; the prompts are kgp templates (no PasswordEval wording) and deterministic; the plot is
written; the job files are GPU jobs that never run Python on a login node.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def job(name: str) -> Path:
    """A job script or shell helper by file name: scripts/slurm/*.sbatch, scripts/pbs/*.pbs, scripts/lib/*.sh."""
    if Path(name).suffix == ".py":
        return HERE / name  # a module of this folder
    sub = {".sbatch": "slurm", ".pbs": "pbs", ".sh": "lib"}[Path(name).suffix]
    return HERE.parents[2] / "scripts" / sub / name
sys.path.insert(0, str(HERE))

import postnorm_diag as PD  # noqa: E402


def tiny(kind: str, zero: bool = False):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    common = dict(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=16, max_position_embeddings=128)  # fmt: skip
    if kind == "gemma":
        m = Gemma3ForCausalLM(Gemma3TextConfig(**common, sliding_window=8))
        for ly in m.model.layers:  # non-trivial norm weights
            for n in (ly.post_attention_layernorm, ly.post_feedforward_layernorm):
                torch.nn.init.normal_(n.weight, std=0.5)
    else:
        m = Qwen2ForCausalLM(Qwen2Config(**common))
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "o_proj"], lora_dropout=0.0,
                                     init_lora_weights=zero))  # fmt: skip
    m.eval()
    return m


def main() -> int:
    import torch

    sys.stdout.reconfigure(encoding="utf-8")
    ids = [[(7 * i + j) % 90 + 3 for j in range(12 + i)] for i in range(4)]
    g = tiny("gemma")
    layers = PD.find_layers(g)
    assert len(layers) == 6 and PD.layer_types(g, 6) == ["local"] * 5 + ["global"]
    # LoRA delta == adapted output - base output; float32 norm == the module
    x = torch.randn(1, 5, 32)
    op = layers[2].self_attn.o_proj
    with torch.no_grad():
        assert torch.allclose(PD.lora_delta32(op, x), op(x).float() - op.base_layer(x).float(), atol=1e-5)
        n = layers[2].post_attention_layernorm
        assert torch.allclose(PD.rms_norm32(n, x), n(x).float(), atol=1e-5)
    rows = PD.collect(g, ids, "cpu", "gemma")
    assert [r["layer"] for r in rows] == list(range(6)) and rows[5]["attn_type"] == "global"
    assert abs(sum(r["delta_share"] for r in rows) - 1.0) < 1e-6
    for r in rows:
        assert r["delta_norm"] > 0 and r["gain_mean"] > 0 and -1e-6 <= r["angle_deg"] <= 180 and r["post_attn_change"] is not None
    # one layer by hand: hook the input of o_proj at layer 1 for the first prompt
    cap = {}
    hd = layers[1].self_attn.o_proj.register_forward_hook(lambda mod, i, o: cap.setdefault("x", i[0].detach()))
    with torch.no_grad():
        g(input_ids=torch.tensor([ids[0]]))
    hd.remove()
    xx = cap["x"][0]
    m = layers[1].self_attn.o_proj.base_layer(xx).float()
    d = PD.lora_delta32(layers[1].self_attn.o_proj, xx)
    N = layers[1].post_attention_layernorm
    dv = PD.rms_norm32(N, m + d) - PD.rms_norm32(N, m)
    one = PD.collect(g, [ids[0]], "cpu", "gemma")[1]
    assert abs(one["gain_mean"] - float((dv.norm(dim=-1) / d.norm(dim=-1)).mean())) < 1e-4
    assert abs(one["delivered_norm"] - float(dv.norm(dim=-1).mean())) < 1e-4
    # Qwen: no post-norm
    q = tiny("qwen")
    qr = PD.collect(q, ids, "cpu", "qwen")
    assert all(r["gain_mean"] is None and r["post_attn_change"] is None and r["delivered_norm"] == r["delta_norm"] for r in qr)
    assert all(r["resid_change_rel"] > 0 for r in qr)
    # zero adapter: nothing changes
    z = PD.collect(tiny("gemma", zero=True), ids, "cpu", "gemma0")
    assert all(r["delta_norm"] == 0 and r["attn_out_change"] == 0 and r["resid_change_rel"] == 0 for r in z)
    # summary, CSV round trip, plot
    s = PD.summarise(rows)
    assert "spearman_gain_vs_delta_share" in s and len(s["top5_by_gain"]) == 5 and "gain" in s["note"]
    assert "no post-attention norm" in PD.summarise(qr)["note"]
    tmp = Path(tempfile.mkdtemp())
    PD.write_csv(tmp / "g.csv", rows)
    PD.write_csv(tmp / "q.csv", qr)
    back = PD.read_csv(tmp / "g.csv")
    assert [r["layer"] for r in back] == list(range(6)) and abs(back[3]["gain_mean"] - rows[3]["gain_mean"]) < 1e-4
    plot_note = "plot: run --plot-only (no matplotlib here)"
    try:
        import matplotlib  # noqa: F401

        test_plot(tmp)
        plot_note = "plot written"
    except ModuleNotFoundError:
        pass
    # prompts: our kgp templates, deterministic; no PasswordEval wording
    ps = PD.prompts()
    assert len(ps) == 50 and ps == PD.prompts() and all(p[0]["role"] == "system" for p in ps)
    assert not any("password_eval" in p[0]["content"].lower() for p in ps)
    # jobs: GPU, guarded, the right adapters
    sb = job("postnorm_diag.sbatch").read_text(encoding="utf-8")
    pb = job("postnorm_diag.pbs").read_text(encoding="utf-8")
    assert "--gres=gpu:1" in sb and "--partition" not in sb and "fsa_lite_gemma12b_run2/adapter" in sb and "--time=00:" in sb
    assert "ngpus=1" in pb and "#PBS -P" not in pb and pb.index("lite_pbs_require_job") < pb.index('"$PY"')
    assert "harness_adapters/qwen_fsa_lite_run2" in pb and "walltime=00:" in pb
    assert b"\r\n" not in (HERE / "postnorm_diag.py").read_bytes() + sb.encode() + pb.encode()
    print("postnorm diagnostic: OK (LoRA delta = adapted - base; float32 norm = Gemma3RMSNorm; gain / delivered by hand; "
          f"local / global flags; Qwen without post-norm; zero adapter = no change; summary, CSV; {plot_note}; prompts kgp, "
          "deterministic; GPU jobs only)")
    return 0


def synthetic_rows(name: str, n: int, gemma: bool) -> list[dict]:
    rows = []
    for i in range(n):
        g = 1.0 + (i % 6 == 5) * 3.0 + 0.1 * i if gemma else None
        rows.append({"model": name, "layer": i, "rel_depth": i / (n - 1), "attn_type": "global" if (i + 1) % 6 == 0 else "local",
                     "n_tokens": 100, "m_norm": 10.0, "delta_norm": 0.1 + 0.01 * i, "delta_rel": 0.01, "delta_share": 1 / n,
                     "delivered_norm": 0.2, "gain_mean": g, "gain_of_means": g, "cos_mean": 0.8 if gemma else None,
                     "angle_deg": 36.0 if gemma else None, "attn_out_change": 0.1, "post_attn_change": 0.2 if gemma else None,
                     "mlp_change": 0.1, "post_ffn_change": 0.1 if gemma else None, "resid_change_rel": 0.01 * (i + 1)})  # fmt: skip
    return rows


def test_plot(tmp: Path) -> None:
    PD.write_csv(tmp / "pg.csv", synthetic_rows("gemma", 48, True))
    PD.write_csv(tmp / "pq.csv", synthetic_rows("qwen", 28, False))
    assert PD.main(["plot", "--csv", str(tmp / "pg.csv"), str(tmp / "pq.csv"), "--out", str(tmp / "p.png")]) == 0
    assert (tmp / "p.png").stat().st_size > 10_000


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        out = Path(tempfile.mkdtemp())
        test_plot(out)
        print(f"postnorm plot: OK ({out / 'p.png'}: 48-layer Gemma and 28-layer Qwen synthetic CSVs)")
        sys.exit(0)
    sys.exit(main())
