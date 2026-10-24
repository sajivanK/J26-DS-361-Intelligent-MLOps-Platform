import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c4.contracts.models import C2Diagnosis
from c4.data import FEATURE_COLUMNS
from c4.data.split_manifest import (
    load_adult_split_manifest,
    save_adult_split_manifest,
)
from c4.models import save_pre_release_evidence
import c4.models.pre_release as module


class ExampleModel:
    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.zeros(len(features), dtype=int)

    def predict_proba(self, features):
        return np.tile([0.8, 0.2], (len(features), 1))


@pytest.fixture
def arguments(tmp_path, monkeypatch):
    data_directory = tmp_path / "uci_adult_data"
    data_directory.mkdir()

    for filename, start, rows in [
        ("reference.csv", 0, 20),
        ("test.csv", 100, 40),
    ]:
        frame = pd.DataFrame(
            {
                feature: np.arange(start, start + rows, dtype=float)
                for feature in FEATURE_COLUMNS
            }
        )
        frame["target"] = np.arange(rows) % 2
        frame.to_csv(data_directory / filename, index=False)

    artifact = save_adult_split_manifest(data_root=tmp_path)
    manifest = load_adult_split_manifest(
        artifact.manifest_path.parent,
        data_root=tmp_path,
    )

    test = pd.read_csv(data_directory / "test.csv")
    release_ids = manifest["partitions"]["release_validation"][
        "source_row_ids"
    ]
    release = test.loc[release_ids].copy()

    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "valid_upstream_input.json"
    )
    fields = json.loads(
        fixture.read_text(encoding="utf-8")
    )["c2_diagnosis"]

    # Update the rule BEFORE constructing the validated diagnosis.
    fields["affected_slice_rule"] = (
        f"age <= {float(release['age'].median())}"
    )
    diagnosis = C2Diagnosis.model_validate(fields)

    champion_metadata = {
        "artifact_version": "v1",
        "model_sha256": "a" * 64,
    }
    candidate_metadata = {
        "artifact_version": "candidate-001",
        "model_sha256": "b" * 64,
        "champion_model_sha256": "a" * 64,
        "repair_csv_sha256": "c" * 64,
        "c3_package": {"repair_version": "repair-001"},
    }

    # Use test models to isolate evidence saving and provenance checks.
    # Actual model loading has separate artifact-integrity tests.
    monkeypatch.setattr(
        module,
        "load_champion",
        lambda _: (ExampleModel(), champion_metadata),
    )
    monkeypatch.setattr(
        module,
        "load_candidate",
        lambda _: (ExampleModel(), candidate_metadata),
    )

    return {
        "champion_directory": tmp_path / "champion",
        "candidate_directory": tmp_path / "candidate",
        "split_manifest_directory": artifact.manifest_path.parent,
        "clean_release_validation": release.copy(deep=True),
        "release_validation": release.copy(deep=True),
        "diagnosis": diagnosis,
        "rule_feature_space": "processed",
        "scenario_id": "scenario-001",
        "data_root": tmp_path,
    }


def test_saved_evidence_and_data_have_matching_checksums(arguments):
    artifact = save_pre_release_evidence(**arguments)
    payload = json.loads(
        artifact.evidence_path.read_text(encoding="utf-8")
    )
    directory = artifact.evidence_path.parent

    assert hashlib.sha256(
        artifact.evidence_path.read_bytes()
    ).hexdigest() == artifact.evidence_sha256

    assert artifact.checksum_path.read_text(
        encoding="utf-8"
    ).strip() == artifact.evidence_sha256

    for filename, field in [
        (
            "clean_release_validation.csv",
            "clean_release_validation_csv_sha256",
        ),
        (
            "release_validation.csv",
            "release_validation_csv_sha256",
        ),
        (
            "diagnosed_membership.csv",
            "diagnosed_membership_csv_sha256",
        ),
    ]:
        assert hashlib.sha256(
            (directory / filename).read_bytes()
        ).hexdigest() == payload[field]

    manifest_path = (
        arguments["split_manifest_directory"] / "manifest.json"
    )
    assert payload["split_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    saved_rows = pd.read_csv(directory / "release_validation.csv")
    assert saved_rows["source_row_id"].tolist() == (
        arguments["release_validation"].index.tolist()
    )

    assert payload["slice_basis"] == "c2_diagnosed_slice"
    assert payload["release_membership_verified"] is True
    assert payload["evaluation_partition"] == "release_validation"
    assert payload["diagnosis_rule_feature_space"] == "processed"
    assert payload["membership_basis"] == (
        "clean_release_before_perturbation"
    )
    assert "sealed_future" not in payload
    assert payload["clean_release_contents_verified"] is True


def test_existing_evidence_is_not_overwritten(arguments):
    artifact = save_pre_release_evidence(**arguments)
    original = artifact.evidence_path.read_bytes()

    with pytest.raises(FileExistsError):
        save_pre_release_evidence(**arguments)

    assert artifact.evidence_path.read_bytes() == original


def test_different_champion_is_rejected(arguments, monkeypatch):
    model, metadata = module.load_candidate(None)
    metadata["champion_model_sha256"] = "d" * 64

    monkeypatch.setattr(
        module,
        "load_candidate",
        lambda _: (model, metadata),
    )

    with pytest.raises(ValueError, match="different champion"):
        save_pre_release_evidence(**arguments)


def test_different_clean_row_order_is_rejected(arguments):
    arguments["clean_release_validation"] = (
        arguments["clean_release_validation"].iloc[::-1].copy()
    )

    with pytest.raises(ValueError, match="identical index order"):
        save_pre_release_evidence(**arguments)


def test_invalid_scenario_id_is_rejected(arguments):
    arguments["scenario_id"] = "../other"

    with pytest.raises(ValueError, match="Scenario ID"):
        save_pre_release_evidence(**arguments)


@pytest.mark.parametrize(
    "forbidden_role",
    ["repair", "sealed_future"],
)
def test_nonrelease_row_is_rejected_before_model_loading(
    arguments,
    monkeypatch,
    forbidden_role,
):
    manifest = load_adult_split_manifest(
        arguments["split_manifest_directory"],
        data_root=arguments["data_root"],
    )
    forbidden_id = manifest["partitions"][forbidden_role][
        "source_row_ids"
    ][0]

    release = arguments["release_validation"].copy()
    ids = release.index.tolist()
    ids[0] = forbidden_id
    release.index = ids
    arguments["release_validation"] = release

    def unexpected_model_load(_):
        pytest.fail("Model loaded before rejecting nonrelease rows.")

    monkeypatch.setattr(
        module,
        "load_champion",
        unexpected_model_load,
    )

    with pytest.raises(ValueError, match="release-validation"):
        save_pre_release_evidence(**arguments)


def test_changed_source_file_blocks_evidence_saving(arguments):
    path = arguments["data_root"] / "uci_adult_data" / "test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "age"] = 999.0
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Source CSV checksum"):
        save_pre_release_evidence(**arguments)


def test_membership_is_computed_before_feature_perturbation(arguments):
    clean = arguments["clean_release_validation"]
    threshold = float(clean["age"].median())
    expected = (clean["age"] <= threshold).tolist()

    arguments["release_validation"] = clean.copy(deep=True)
    arguments["release_validation"]["age"] += 1000.0

    artifact = save_pre_release_evidence(**arguments)
    directory = artifact.evidence_path.parent

    membership = pd.read_csv(directory / "diagnosed_membership.csv")
    assert membership["diagnosed_slice_member"].tolist() == expected

    payload = json.loads(
        artifact.evidence_path.read_text(encoding="utf-8")
    )
    assert payload["membership_basis"] == (
        "clean_release_before_perturbation"
    )
    assert payload["diagnosis_rule_feature_space"] == "processed"


def test_raw_space_diagnosis_cannot_be_saved(arguments):
    arguments["rule_feature_space"] = "raw"

    with pytest.raises(ValueError, match="processed-space"):
        save_pre_release_evidence(**arguments)


@pytest.mark.parametrize("column", ["age", "target"])
def test_modified_clean_release_contents_are_rejected(
    arguments, monkeypatch, column
):
    clean = arguments["clean_release_validation"].copy(deep=True)
    row_id = clean.index[0]

    if column == "target":
        clean.loc[row_id, column] = 1 - clean.loc[row_id, column]
    else:
        clean.loc[row_id, column] += 1.0

    arguments["clean_release_validation"] = clean

    def unexpected_model_load(_):
        pytest.fail("Model loaded before rejecting altered clean rows.")

    monkeypatch.setattr(module, "load_champion", unexpected_model_load)

    with pytest.raises(ValueError, match="Clean release contents differ"):
        save_pre_release_evidence(**arguments)