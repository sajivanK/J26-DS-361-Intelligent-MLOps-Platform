"""Tests for champion training and artifact versioning."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c4.data import FEATURE_COLUMNS, TARGET_COLUMN
from c4.models import (
    ChampionConfig,
    load_champion,
    train_and_save_champion,
    train_champion,
)


def make_frame(size: int, offset: int = 0) -> pd.DataFrame:
    data = {}

    for column_number, column in enumerate(FEATURE_COLUMNS):
        data[column] = [
            float(offset + row_number) + column_number / 100
            for row_number in range(size)
        ]

    data[TARGET_COLUMN] = [
        row_number % 2
        for row_number in range(size)
    ]

    return pd.DataFrame(data)


def write_adult_dataset(root: Path) -> None:
    data_directory = root / "uci_adult_data"
    data_directory.mkdir(parents=True)

    make_frame(40).to_csv(
        data_directory / "reference.csv",
        index=False,
    )
    make_frame(20, offset=1000).to_csv(
        data_directory / "test.csv",
        index=False,
    )


def small_config() -> ChampionConfig:
    return ChampionConfig(
        n_estimators=4,
        max_depth=2,
        n_jobs=1,
    )


def test_train_champion_produces_binary_probabilities():
    reference = make_frame(40)

    model = train_champion(reference, small_config())

    features = reference.loc[:, FEATURE_COLUMNS]
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (40, 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(40),
    )


def test_train_and_save_champion_records_metadata(tmp_path):
    write_adult_dataset(tmp_path)

    artifact = train_and_save_champion(
        data_root=tmp_path,
        artifact_version="test-v1",
        config=small_config(),
    )

    assert artifact.model_path.is_file()
    assert artifact.metadata_path.is_file()

    metadata = artifact.metadata

    assert metadata["artifact_version"] == "test-v1"
    assert metadata["role"] == "champion"
    assert metadata["training_partition"] == "reference_only"
    assert metadata["reference_rows"] == 40
    assert metadata["training_config"]["n_estimators"] == 4
    assert metadata["model_sha256"]
    assert metadata["reference_csv_sha256"]

    data_usage = metadata["data_usage"]

    assert data_usage["reference_used_for_training"] is True
    assert data_usage["repair_used_for_training"] is False
    assert data_usage["release_validation_used_for_training"] is False
    assert data_usage["sealed_future_used_for_training"] is False


def test_saved_champion_reloads_with_equal_predictions(tmp_path):
    write_adult_dataset(tmp_path)

    artifact = train_and_save_champion(
        data_root=tmp_path,
        artifact_version="reload-test",
        config=small_config(),
    )

    original_model, _ = load_champion(
        artifact.model_path.parent
    )

    reference = pd.read_csv(
        tmp_path / "uci_adult_data" / "reference.csv"
    )
    features = reference.loc[:, FEATURE_COLUMNS]

    original_probabilities = original_model.predict_proba(features)

    reloaded_model, metadata = load_champion(
        artifact.model_path.parent
    )
    reloaded_probabilities = reloaded_model.predict_proba(features)

    np.testing.assert_allclose(
        original_probabilities,
        reloaded_probabilities,
        rtol=0.0,
        atol=0.0,
    )
    assert metadata["artifact_version"] == "reload-test"


def test_existing_champion_version_is_not_overwritten(tmp_path):
    write_adult_dataset(tmp_path)

    train_and_save_champion(
        data_root=tmp_path,
        artifact_version="protected-v1",
        config=small_config(),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        train_and_save_champion(
            data_root=tmp_path,
            artifact_version="protected-v1",
            config=small_config(),
        )


def test_invalid_artifact_version_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Artifact version"):
        train_and_save_champion(
            data_root=tmp_path,
            artifact_version="../invalid",
            config=small_config(),
        )