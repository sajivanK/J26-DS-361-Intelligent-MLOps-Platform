"""Trusted model artifact loading utilities for Component 3."""

import pickle
import sys
from pathlib import Path
from typing import Any

from component3.src.core.config_loader import (
    PROJECT_ROOT,
    ConfigurationError,
    get_config_value,
    resolve_project_path,
)


class ModelLoadingError(Exception):
    """Raised when a model artifact cannot be loaded."""


def load_trusted_pickle(
    path_value: str | Path,
    *,
    trusted: bool = False,
) -> Any:
    """Load a pickle artifact only after explicit trust approval.

    WARNING: Pickle deserialization can execute arbitrary code.
    The caller must ensure the artifact comes from a trusted source.
    """
    if not trusted:
        raise ModelLoadingError(
            "Pickle loading requires explicit trusted=True approval."
        )

    path = resolve_project_path(str(path_value))
    if not path.is_file():
        raise ModelLoadingError(f"Model artifact file not found: {path}")

    # Ensure shared package directory is in sys.path for unpickling shared modules (e.g. uci_adult)
    shared_dir = str((PROJECT_ROOT / "shared").resolve())
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)

    try:
        try:
            import joblib

            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as exc:
        raise ModelLoadingError(
            f"Failed to load trusted model artifact from {path}: {exc}"
        ) from exc


def load_baseline_artifacts(
    config: dict[str, Any],
    *,
    trusted: bool = False,
) -> tuple[Any, Any]:
    """Load baseline model and preprocessor artifacts configured for C3."""
    if not trusted:
        raise ModelLoadingError(
            "Pickle loading requires explicit trusted=True approval."
        )

    model_path = get_config_value(config, "model", "model_path")
    preprocessor_path = get_config_value(config, "model", "preprocessor_path")

    model = load_trusted_pickle(model_path, trusted=True)
    preprocessor = load_trusted_pickle(preprocessor_path, trusted=True)

    return model, preprocessor
