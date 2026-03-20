"""Configuration loading utilities for PhaseForensics."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file and return a nested dict.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Nested dictionary of configuration values.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file cannot be parsed as valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML config at {path}: {exc}") from exc

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"Expected top-level YAML mapping, got {type(config).__name__}")

    return config


def get_default_config() -> Dict[str, Any]:
    """Load the default configuration from configs/default.yaml.

    Returns:
        Nested dictionary of default configuration values.
    """
    return load_config(_DEFAULT_CONFIG_PATH)
