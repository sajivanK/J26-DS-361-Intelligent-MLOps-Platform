"""Unit tests for the externally stored Adult dataset loader."""

from pathlib import Path

import pandas as pd
import pytest

from c4.data import (
    DATA_ROOT_ENV,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_adult_dataset,
    resolve_data_root,
)


def make_valid_frame() -> pd.DataFrame:
    data = {
        column: [float(index), float(index + 1)]
        for index, column in enumerate(FEATURE_COLUMNS)
    }
    data[TARGET_COLUMN] = [0, 1]
    return pd.DataFrame(data)


def write_adult_files(root: Path) -> None:
    data_directory = root / "uci_adult_data"
    data_directory.mkdir(parents=True)

    frame = make_valid_frame()
    frame.to_csv(data_directory / "reference.csv", index=False)
    frame.to_csv(data_directory / "test.csv", index=False)


def test_loads_valid_adult_dataset_from_explicit_root(tmp_path):
    write_adult_files(tmp_path)

    dataset = load_adult_dataset(tmp_path)

    assert dataset.reference.shape == (2, 13)
    assert dataset.test.shape == (2, 13)
    assert list(dataset.reference.columns) == [
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]


def test_resolves_data_root_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))

    assert resolve_data_root() == tmp_path


def test_missing_environment_variable_is_rejected(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)

    with pytest.raises(RuntimeError, match=DATA_ROOT_ENV):
        resolve_data_root()


def test_missing_required_csv_is_rejected(tmp_path):
    data_directory = tmp_path / "uci_adult_data"
    data_directory.mkdir()

    make_valid_frame().to_csv(
        data_directory / "reference.csv",
        index=False,
    )

    with pytest.raises(FileNotFoundError, match="test.csv"):
        load_adult_dataset(tmp_path)


def test_invalid_column_schema_is_rejected(tmp_path):
    write_adult_files(tmp_path)

    reference_path = (
        tmp_path / "uci_adult_data" / "reference.csv"
    )
    invalid_reference = pd.read_csv(reference_path).drop(
        columns=["age"]
    )
    invalid_reference.to_csv(reference_path, index=False)

    with pytest.raises(ValueError, match="columns"):
        load_adult_dataset(tmp_path)


def test_missing_values_are_rejected(tmp_path):
    write_adult_files(tmp_path)

    test_path = tmp_path / "uci_adult_data" / "test.csv"
    invalid_test = pd.read_csv(test_path)
    invalid_test.loc[0, "age"] = None
    invalid_test.to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="missing values"):
        load_adult_dataset(tmp_path)


def test_non_binary_target_is_rejected(tmp_path):
    write_adult_files(tmp_path)

    test_path = tmp_path / "uci_adult_data" / "test.csv"
    invalid_test = pd.read_csv(test_path)
    invalid_test.loc[0, TARGET_COLUMN] = 2
    invalid_test.to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="binary labels"):
        load_adult_dataset(tmp_path)