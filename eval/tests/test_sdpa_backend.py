"""SDPA backend per benchmark (models/hf.py sdpa_backend; ConfAIde Tier 4 on math SDPA, as the team
protocol's ConfAIde rows): flags set and restored, recorded in summary.json for both variants,
part of the pairing fingerprint, refused by the team table when the variants differ, and greedy
outputs unchanged on the tiny models.

The crash it avoids ("p.attn_bias_ptr is not correctly aligned") is in PyTorch's CUDA
memory-efficient kernel; CPU tests cannot reproduce it. The torch.backends.cuda flags only steer
CUDA kernels, so on CPU the output check shows that the context changes nothing else; on a GPU
machine the same test compares the math backend against the default kernels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from core import protocol as P  # noqa: E402
from core.interfaces import GenConfig, Item  # noqa: E402
from models.hf import sdpa_backend  # noqa: E402
from stats import team_table as TT  # noqa: E402
from stats.rescore import rescore_run  # noqa: E402


def test_context_sets_and_restores_the_flags():
    cuda = torch.backends.cuda
    before = (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.math_sdp_enabled())
    with sdpa_backend("math") as rec:
        assert rec["backend"] == "math"
        assert rec["flags"]["math"] is True and rec["flags"]["flash"] is False and rec["flags"]["mem_efficient"] is False
        assert not cuda.mem_efficient_sdp_enabled()
    assert (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.math_sdp_enabled()) == before
    with sdpa_backend(None) as rec:
        assert rec["backend"] == "default"
    with pytest.raises(ValueError, match="sdpa_backend 'flash'"):
        with sdpa_backend("flash"):
            pass
    with pytest.raises(RuntimeError):  # restored even when the block fails
        with sdpa_backend("math"):
            raise RuntimeError("boom")
    assert (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.math_sdp_enabled()) == before


@pytest.mark.parametrize("name", ["tiny-gemma3", "tiny-qwen2"])
def test_greedy_outputs_unchanged_under_math_sdpa(name):
    from analysis.internals.tiny import TINY

    m = TINY[name]({"device": "cuda" if torch.cuda.is_available() else "cpu", "dtype": "float32",
                    "batch_size": 4, "attn_implementation": "sdpa"})  # fmt: skip
    long_text = " ".join(f"Line {i}: the meeting notes continue here." for i in range(12))
    items = [Item(id=str(k), system_prompt="You are a helpful assistant.", user_prompt=long_text[: 60 + 37 * k])
             for k in range(4)]  # different lengths: a padded batch, lengths not multiples of 8
    msgs = [it.messages() for it in items]
    cfg = GenConfig(max_new_tokens=8)
    with sdpa_backend("default"):
        a = m.generate(msgs, cfg)
    with sdpa_backend("math"):
        b = m.generate(msgs, cfg)
    assert [(g.text, g.n_tokens) for g in a] == [(g.text, g.n_tokens) for g in b]


def test_confaide_uses_math_for_both_variants_and_the_rest_default(tmp_path):
    from tests.test_protocol import _template_spec

    spec = _template_spec(tmp_path, "sdpa")
    out = {v: {d.name: d for d in P.run_protocol(spec["name"], "gemma", v, benchmarks=["confaide_t4_team", "mmlu_team"],
                                                  dry_run=True, output_dir=tmp_path / "out", approach_spec=spec)}
           for v in P.VARIANTS}  # fmt: skip
    for v in P.VARIANTS:
        cf = json.loads((out[v]["confaide_t4_team"] / "summary.json").read_text(encoding="utf-8"))
        mm = json.loads((out[v]["mmlu_team"] / "summary.json").read_text(encoding="utf-8"))
        assert cf["sdpa_backend"]["backend"] == "math" and mm["sdpa_backend"]["backend"] == "default"
        assert "sdpa_backend" in P.SUMMARY_KEYS
    for bench in ("confaide_t4_team", "mmlu_team"):
        so, st = (json.loads((out[v][bench] / "summary.json").read_text(encoding="utf-8")) for v in P.VARIANTS)
        assert so["settings_sha256"] == st["settings_sha256"]


def _edit(d: Path, fn) -> None:
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    fn(s)
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")


def test_team_table_refuses_pairs_with_different_sdpa_backends(tmp_path):
    from tests.test_protocol import _template_spec

    spec = _template_spec(tmp_path, "sdpa_pair")
    dirs = {}
    for v in P.VARIANTS:
        d = P.run_protocol(spec["name"], "gemma", v, benchmarks=["confaide_t4_team"], dry_run=True,
                           output_dir=tmp_path / "out", approach_spec=spec)[0]  # fmt: skip
        rescore_run(d)
        dirs[v] = d

    def rows():
        return [r for r in TT.build(tmp_path / "out")["rows"] if r["benchmark"] == "confaide_t4_team" and r["metric"] != "-"]

    assert rows() and all("SDPA" not in r["status"] for r in rows())
    _edit(dirs["technique"], lambda s: s.update(sdpa_backend={"backend": "default", "flags": None}))
    assert all("not paired: SDPA backend differs (math vs default)" in r["status"] for r in rows())
    # runs made before the field existed count as "default": two such runs still pair on it
    for d in dirs.values():
        _edit(d, lambda s: s.pop("sdpa_backend", None))
    assert all("SDPA" not in r["status"] for r in rows())
