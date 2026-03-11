from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"


def _load_yaml() -> dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_settings: dict[str, Any] | None = None


def get_settings() -> dict[str, Any]:
    global _settings
    if _settings is None:
        load_dotenv(_ROOT / ".env")
        _settings = _load_yaml()
    return _settings


def get_env(key: str, default: str = "") -> str:
    load_dotenv(_ROOT / ".env")
    return os.getenv(key, default)
