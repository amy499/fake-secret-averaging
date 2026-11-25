import os

import pytest
from pydantic import ValidationError

from fsa_runtime.settings import ROOT, load_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("FSA_"):
            monkeypatch.delenv(key)


def test_defaults_from_app_yaml():
    s = load_settings(env_file=None)
    assert s.backend == "mock" and s.is_mock
    assert s.api_host == "127.0.0.1"  # never 0.0.0.0 by default
    assert s.api_port == 8001
    assert s.cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]
    assert s.runs_dir == (ROOT / "outputs" / "runs").resolve()
    assert s.mock_runs_dir == (ROOT / "mock_data" / "runs").resolve()
    assert s.active_runs_dir == s.mock_runs_dir


def test_env_file_overrides_yaml_and_env_overrides_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FSA_BACKEND=real\nFSA_API_PORT=9000\nFSA_RUNS_DIR=elsewhere/runs\nHF_TOKEN=secret\n",
        encoding="utf-8",
    )
    s = load_settings(env_file=env_file)
    assert s.backend == "real" and s.api_port == 9000
    assert s.runs_dir == (ROOT / "elsewhere" / "runs").resolve()  # relative to the experiment
    assert s.active_runs_dir == s.runs_dir

    monkeypatch.setenv("FSA_API_PORT", "9100")
    monkeypatch.setenv("FSA_CORS_ORIGINS", '["http://localhost:5174"]')
    s = load_settings(env_file=env_file)
    assert s.api_port == 9100 and s.cors_origins == ["http://localhost:5174"]


def test_attacker_is_a_setting(monkeypatch):
    s = load_settings(env_file=None)
    assert s.attacker_hf_id == "Qwen/Qwen2.5-1.5B-Instruct" and s.attacker_adapter is None
    monkeypatch.setenv("FSA_ATTACKER_ADAPTER", "outputs/attacker/lora")
    assert load_settings(env_file=None).attacker_adapter == "outputs/attacker/lora"


def test_absolute_paths_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("FSA_MOCK_RUNS_DIR", str(tmp_path))
    assert load_settings(env_file=None).mock_runs_dir == tmp_path.resolve()


def test_invalid_values_rejected(monkeypatch):
    monkeypatch.setenv("FSA_BACKEND", "gpu")
    with pytest.raises(ValidationError):
        load_settings(env_file=None)
