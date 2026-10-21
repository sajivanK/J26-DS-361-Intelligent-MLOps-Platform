"""Configuration loader for Component 3 (Adaptive Synthetic Data Repair Engine)."""

import json
from pathlib import Path
from typing import Any

# Repository root: three levels above this file's parent directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "component3" / "configs" / "c3_config.json"
)
REQUIRED_SECTIONS = (
    "dataset",
    "model",
    "integration",
    "readiness",
)

_MISSING = object()


class ConfigurationError(Exception):
    """Raised when Component 3 configuration is invalid."""


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path to an absolute path."""
    if path_value is None or (isinstance(path_value, str) and not path_value.strip()):
        raise ConfigurationError("Path must be a non-empty string or Path.")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and perform basic validation of C3 configuration."""
    path = (
        resolve_project_path(config_path)
        if config_path is not None
        else DEFAULT_CONFIG_PATH
    )
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON configuration: {exc}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Unable to read configuration: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigurationError(
            "Configuration root must be a JSON object."
        )

    missing = [
        section for section in REQUIRED_SECTIONS
        if section not in config
    ]
    if missing:
        raise ConfigurationError(
            f"Missing required configuration sections: {', '.join(missing)}"
        )

    return config


def get_config_value(
    config: dict[str, Any],
    *keys: str,
    default: Any = _MISSING,
) -> Any:
    """Retrieve a nested configuration value safely."""
    if not keys:
        raise ConfigurationError("At least one key must be provided.")
    curr: Any = config
    for key in keys:
        if not isinstance(curr, dict) or key not in curr:
            if default is not _MISSING:
                return default
            raise ConfigurationError(f"Configuration key not found: {'.'.join(keys)}")
        curr = curr[key]
    return curr
