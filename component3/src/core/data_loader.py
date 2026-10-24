"""Data and evidence loader for Component 3."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from component3.src.core.config_loader import (
    get_config_value,
    resolve_project_path,
)


class DataLoadingError(Exception):
    """Raised when data loading fails."""


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    resolved = resolve_project_path(path)
    if not resolved.is_file():
        raise DataLoadingError(f"JSON file not found: {resolved}")
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise DataLoadingError(f"Invalid JSON in {resolved}: {exc}") from exc
    except OSError as exc:
        raise DataLoadingError(f"Unable to read JSON file {resolved}: {exc}") from exc


def load_csv_dataset(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    resolved = resolve_project_path(path)
    if not resolved.is_file():
        raise DataLoadingError(f"CSV file not found: {resolved}")
    try:
        df = pd.read_csv(resolved, **kwargs)
    except Exception as exc:
        raise DataLoadingError(f"Error reading CSV {resolved}: {exc}") from exc
    if df.empty:
        raise DataLoadingError(f"Loaded DataFrame is empty: {resolved}")
    return df


# Alias for load_csv_dataset
load_csv = load_csv_dataset


def load_reference_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load the baseline reference dataset configured for C3."""
    path_val = get_config_value(config, "dataset", "reference_path")
    return load_csv_dataset(path_val)


def load_current_dataset(
    config: dict[str, Any],
    scenario_id: str,
) -> pd.DataFrame:
    """Load a drifted current dataset for a given scenario ID."""
    if scenario_id not in {f"S{i:02d}" for i in range(8)}:
        raise DataLoadingError(f"Unsupported scenario ID: {scenario_id}")
    directory = get_config_value(config, "dataset", "current_data_dir")
    path = resolve_project_path(directory) / f"{scenario_id}.csv"
    return load_csv_dataset(path)


def load_c1_evidence(
    config: dict[str, Any],
    scenario_id: str,
) -> dict[str, Any]:
    """Load simulated Component 1 evidence."""
    if scenario_id not in {f"S{i:02d}" for i in range(8)}:
        raise DataLoadingError(f"Unsupported scenario ID: {scenario_id}")
    directory = get_config_value(config, "integration", "c1_evidence_dir")
    path = resolve_project_path(directory) / f"c1_evidence_{scenario_id}.json"
    return load_json_file(path)


def load_c2_diagnosis(
    config: dict[str, Any],
    scenario_id: str,
) -> dict[str, Any]:
    """Load simulated Component 2 diagnosis."""
    if scenario_id not in {f"S{i:02d}" for i in range(8)}:
        raise DataLoadingError(f"Unsupported scenario ID: {scenario_id}")
    directory = get_config_value(config, "integration", "c2_diagnosis_dir")
    path = resolve_project_path(directory) / f"c2_diagnosis_{scenario_id}.json"
    return load_json_file(path)


def load_integration_contracts(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load official C1 and C2 contract definitions."""
    c1_path = get_config_value(config, "integration", "c1_contract_path")
    c2_path = get_config_value(config, "integration", "c2_contract_path")
    return (
        load_json_file(c1_path),
        load_json_file(c2_path),
    )
