from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    secret_key: str


def load_config(path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config


def load_credentials(require: bool = True) -> AlpacaCredentials | None:
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not secret_key:
        if require:
            raise RuntimeError(
                "Missing Alpaca credentials. Copy .env.example to .env and add paper-trading keys."
            )
        return None
    return AlpacaCredentials(api_key=api_key, secret_key=secret_key)
