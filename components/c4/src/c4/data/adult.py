"""Loading and validation for the externally stored UCI Adult dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_ROOT_ENV = "C4_DATA_ROOT"
TARGET_COLUMN = "target"

FEATURE_COLUMNS = (
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
)

EXPECTED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)


@dataclass(frozen=True)
class AdultDataset:
    """Validated preprocessed Adult reference and holdout datasets."""

    reference: pd.DataFrame
    test: pd.DataFrame


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the external dataset root without embedding a personal path."""

    raw_path = data_root if data_root is not None else os.getenv(DATA_ROOT_ENV)

    if not raw_path:
        raise RuntimeError(
            f"{DATA_ROOT_ENV} is not configured. "
            "Set it to the external J26-DS-361-Datasets directory."
        )

    root = Path(raw_path).expanduser()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    return root


def _validate_frame(frame: pd.DataFrame, name: str) -> None:
    """Validate a preprocessed Adult dataframe."""

    if frame.empty:
        raise ValueError(f"{name} dataset is empty.")

    actual_columns = tuple(frame.columns)

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"{name} columns do not match the expected Adult schema. "
            f"Expected {EXPECTED_COLUMNS}, received {actual_columns}."
        )

    if frame.isna().any().any():
        raise ValueError(f"{name} dataset contains missing values.")

    non_numeric = [
        column
        for column in EXPECTED_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]

    if non_numeric:
        raise ValueError(
            f"{name} contains non-numeric processed columns: {non_numeric}"
        )

    labels = set(frame[TARGET_COLUMN].unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{name} target must contain only binary labels 0 and 1; "
            f"received {sorted(labels)}."
        )

    if labels != {0, 1}:
        raise ValueError(f"{name} must contain both target classes 0 and 1.")


def load_adult_dataset(
    data_root: str | Path | None = None,
) -> AdultDataset:
    """Load the preprocessed Adult files directly from external storage."""

    root = resolve_data_root(data_root)
    adult_directory = root / "uci_adult_data"
    reference_path = adult_directory / "reference.csv"
    test_path = adult_directory / "test.csv"

    for path in (reference_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Adult dataset file missing: {path}")

    reference = pd.read_csv(reference_path)
    test = pd.read_csv(test_path)

    _validate_frame(reference, "reference")
    _validate_frame(test, "test")

    return AdultDataset(reference=reference, test=test)