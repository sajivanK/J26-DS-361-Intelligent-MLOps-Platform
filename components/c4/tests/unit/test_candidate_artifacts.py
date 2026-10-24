import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c4.contracts.models import C3RepairPackage
from c4.data import FEATURE_COLUMNS
from c4.models import (
    ChampionConfig,
    load_candidate,
    load_champion,
    train_and_save_candidate,
    train_and_save_champion,
    train_candidate,
)


@pytest.fixture
def setup_artifacts(tmp_path):
    data_directory = tmp_path / "uci_adult_data"
    data_directory.mkdir()

    reference = pd.DataFrame({
        feature: [0.0, 1.0, 2.0, 3.0]
        for feature in FEATURE_COLUMNS
    })
    reference["target"] = [0, 1, 0, 1]
    reference.to_csv(data_directory / "reference.csv", index=False)
    reference.to_csv(data_directory / "test.csv", index=False)
    reference.iloc[:2].to_csv(tmp_path / "repair.csv", index=False)

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures" / "valid_upstream_input.json"
    )
    fields = json.loads(
        fixture_path.read_text(encoding="utf-8")
    )["c3_repair_package"]
    fields.update(
        repair_dataset_uri="repair.csv",
        n_synthetic_rows=2,
        augmentation_ratio=0.5,
    )
    package = C3RepairPackage.model_validate(fields)

    config = ChampionConfig(
        n_estimators=3,
        max_depth=2,
        n_jobs=1,
    )
    champion = train_and_save_champion(
        data_root=tmp_path,
        config=config,
    )

    # Candidate saving must work without the test dataset.
    (data_directory / "test.csv").unlink()

    return tmp_path, reference, package, config, champion


def test_candidate_artifact_preserves_predictions_and_champion(
    setup_artifacts,
):
    root, reference, package, config, champion = setup_artifacts
    champion_bytes = champion.model_path.read_bytes()

    expected = train_candidate(
        reference, package, data_root=root, config=config
    )
    artifact = train_and_save_candidate(
        package, artifact_version="candidate-001", data_root=root
    )
    reloaded, metadata = load_candidate(artifact.model_path.parent)

    features = reference.loc[:, list(FEATURE_COLUMNS)]
    np.testing.assert_allclose(
        expected.model.predict_proba(features),
        reloaded.predict_proba(features),
    )

    assert metadata["training_rows"] == 6
    assert metadata["repair_csv_sha256"] == (
        expected.repair.dataset_sha256
    )
    assert metadata["champion_model_sha256"] == (
        champion.metadata["model_sha256"]
    )
    assert champion.model_path.read_bytes() == champion_bytes
    load_champion(champion.model_path.parent)


def test_existing_candidate_version_is_not_overwritten(setup_artifacts):
    root, _, package, _, _ = setup_artifacts
    train_and_save_candidate(
        package, artifact_version="candidate-001", data_root=root
    )

    with pytest.raises(FileExistsError):
        train_and_save_candidate(
            package, artifact_version="candidate-001", data_root=root
        )


def test_changed_reference_is_rejected(setup_artifacts):
    root, reference, package, _, _ = setup_artifacts
    reference.loc[0, "age"] = 99.0
    reference.to_csv(
        root / "uci_adult_data" / "reference.csv", index=False
    )

    with pytest.raises(ValueError, match="Reference CSV differs"):
        train_and_save_candidate(
            package, artifact_version="candidate-001", data_root=root
        )


def test_candidate_checksum_detects_tampering(setup_artifacts):
    root, _, package, _, _ = setup_artifacts
    artifact = train_and_save_candidate(
        package, artifact_version="candidate-001", data_root=root
    )
    artifact.model_path.write_bytes(b"invalid model")

    with pytest.raises(ValueError, match="checksum"):
        load_candidate(artifact.model_path.parent)


@pytest.mark.parametrize("version", ["", ".", "..", "../other"])
def test_invalid_candidate_version_is_rejected(
    setup_artifacts, version
):
    root, _, package, _, _ = setup_artifacts

    with pytest.raises(ValueError, match="Version"):
        train_and_save_candidate(
            package, artifact_version=version, data_root=root
        )