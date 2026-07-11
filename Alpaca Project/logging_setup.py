from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def configure_logging(config: dict[str, Any]) -> None:
    logging_config = config.get("logging", {})
    level_name = str(logging_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = Path(logging_config.get("file", "logs/trading_system.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
