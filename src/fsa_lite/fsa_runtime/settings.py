"""Dashboard settings: configs/training/app.yaml gives the defaults, .env and FSA_* env vars override them.

Precedence (highest first): process environment > .env file > configs/training/app.yaml. Keys are flat
and prefixed with FSA_ in the environment (FSA_BACKEND, FSA_API_PORT, FSA_RUNS_DIR, ...);
list values such as FSA_CORS_ORIGINS are given as JSON. Relative paths are resolved against
, so nothing depends on the current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]
APP_YAML = ROOT / "configs" / "training" / "app.yaml"
ENV_FILE = ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FSA_", extra="ignore")

    backend: Literal["mock", "real"]
    api_host: str
    api_port: int = Field(ge=1, le=65535)
    cors_origins: list[str]
    runs_dir: Path
    mock_runs_dir: Path
    # scripts/utility_eval.py output (real backend only)
    utility_dir: Path = Field(default=Path("outputs/utility"), validate_default=True)
    seed: int
    default_n_samples: int = Field(ge=1)
    max_n_samples: int = Field(ge=1, le=200)
    default_temperature: float = Field(gt=0, le=2)
    default_prompt: str
    attacker_label: str
    attacker_hf_id: str
    attacker_adapter: str | None = None
    # Real (Hugging Face) backend
    hf_device: str = "cpu"
    hf_dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"  # auto: bf16 on CUDA
    hf_max_new_tokens: int = Field(default=160, ge=8, le=256)  # see hf.MAX_NEW_TOKENS_CAP
    hf_batch_size: int = Field(default=8, ge=1)
    # Standard HF variables (also read from .env); never logged.
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("FSA_HF_TOKEN", "HF_TOKEN"), repr=False
    )
    hf_home: str | None = Field(
        default=None, validation_alias=AliasChoices("FSA_HF_HOME", "HF_HOME")
    )

    @field_validator("hf_token", "hf_home")
    @classmethod
    def _blank_is_none(cls, v: str | None) -> str | None:
        return v or None

    @field_validator("runs_dir", "mock_runs_dir", "utility_dir")
    @classmethod
    def _resolve(cls, p: Path) -> Path:
        return (p if p.is_absolute() else ROOT / p).resolve()

    @property
    def is_mock(self) -> bool:
        return self.backend == "mock"

    @property
    def active_runs_dir(self) -> Path:
        """Mock backend serves mock_data/runs; real backend serves outputs/runs."""
        return self.mock_runs_dir if self.is_mock else self.runs_dir

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # app.yaml values arrive as init kwargs, so they go last (lowest priority).
        return env_settings, dotenv_settings, init_settings


def _yaml_defaults(path: Path) -> dict[str, Any]:
    app = yaml.safe_load(path.read_text(encoding="utf-8"))
    pg = app.get("playground", {})
    attacker = app["attacker"]
    hf = app.get("hf", {})
    return {
        "backend": app["backend"],
        "api_host": app["api"]["host"],
        "api_port": app["api"]["port"],
        "cors_origins": app["api"]["cors_origins"],
        "runs_dir": app["paths"]["runs_dir"],
        "mock_runs_dir": app["paths"]["mock_runs_dir"],
        **({"utility_dir": app["paths"]["utility_dir"]} if "utility_dir" in app["paths"] else {}),
        "seed": app["seed"],
        "default_n_samples": pg["default_n_samples"],
        "max_n_samples": pg["max_n_samples"],
        "default_temperature": pg["default_temperature"],
        "default_prompt": pg["default_prompt"],
        "attacker_label": attacker["label"],
        "attacker_hf_id": attacker["hf_id"],
        "attacker_adapter": attacker.get("adapter"),
        **{
            f"hf_{k}": v
            for k, v in hf.items()
            if k in ("device", "dtype", "max_new_tokens", "batch_size")
        },
    }


def load_settings(app_yaml: Path = APP_YAML, env_file: Path | None = ENV_FILE) -> Settings:
    """Load settings; pass env_file=None to ignore .env (tests)."""
    return Settings(_env_file=env_file, **_yaml_defaults(app_yaml))
