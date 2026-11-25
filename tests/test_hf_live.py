"""Live checks against the real Qwen2.5-0.5B model. Skipped unless the ml group is installed
AND FSA_RUN_HF_TESTS=1 (first run downloads ~1 GB; CPU generation takes a little while).

    $env:FSA_RUN_HF_TESTS = "1"; uv run --group ml pytest tests/test_hf_live.py -v
"""

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    os.environ.get("FSA_RUN_HF_TESTS") != "1", reason="set FSA_RUN_HF_TESTS=1 to run live HF tests"
)

from fsa_runtime.generators.base import ModelSpec
from fsa_runtime.generators.hf import HFGenerator, load_system_templates
from scripts.check_tokenization import analyse

QWEN = ModelSpec(
    method="base", base_model="qwen2.5-0.5b-instruct", hf_id="Qwen/Qwen2.5-0.5B-Instruct"
)
PROMPT = "Generate 20 random 2-digit numbers, separated by commas."


@pytest.fixture(scope="module")
def gen():
    return HFGenerator(templates=load_system_templates(), max_new_tokens=96, batch_size=4)


def test_generates_reproducible_replies(gen):
    a = gen.generate(QWEN, secret="37", prompt=PROMPT, n_samples=2, seed=5)
    b = gen.generate(QWEN, secret="37", prompt=PROMPT, n_samples=2, seed=5)
    assert len(a) == 2 and all(isinstance(r, str) and r for r in a)
    assert a == b


def test_tokenization_check_runs_on_qwen(gen):
    rows, mode = analyse(gen.tokenizer(QWEN.hf_id), load_system_templates()["base"])
    assert len(rows) == 100 and mode >= 1
