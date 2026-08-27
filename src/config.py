from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_MODELS = {"logistic_regression", "random_forest"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML experiment configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    required_sections = {"data", "split", "preprocessing", "model", "evaluation", "output"}
    missing = required_sections - set(config or {})
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if config["model"]["name"] not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model {config['model']['name']!r}; choose from {sorted(SUPPORTED_MODELS)}"
        )
    return config


def with_overrides(
    config: dict[str, Any],
    *,
    db_path: str | None = None,
    model_name: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Return a copy of the config with explicit command-line overrides applied."""
    updated = deepcopy(config)
    if db_path:
        updated["data"]["db_path"] = db_path
    if model_name:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model {model_name!r}; choose from {sorted(SUPPORTED_MODELS)}")
        updated["model"]["name"] = model_name
        # Parameters in the default file belong to logistic regression.
        if model_name != config["model"]["name"]:
            updated["model"]["params"] = {}
    if output_dir:
        updated["output"]["dir"] = output_dir
    return updated

