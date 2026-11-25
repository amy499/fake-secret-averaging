"""Phase 1 sanity checks: configs parse, and the stub modules import without torch installed."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_configs_parse():
    for path in (ROOT / "configs").rglob("*.yaml"):
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict), path


def test_stubs_importable():
    from fsa.patching.metric import patching_metric
    from fsa.targets.mixture import mixture_target, sample_fake_secrets
    from fsa.training.loss import fsa_loss

    for fn in (sample_fake_secrets, mixture_target, fsa_loss, patching_metric):
        assert callable(fn) and fn.__doc__
