"""Generation tests use temporary data, not research experiment results."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from c4.data import FEATURE_COLUMNS
from c4.data.split_manifest import (
    load_adult_split_manifest,
    save_adult_split_manifest,
)
from c4.experiments.tvae_generation import (
    TVAEGenerationConfig,
    generate_tvae_repair,
)
from c4.scenarios.feature_metadata import load_adult_feature_metadata
import c4.experiments.tvae_generation as module


class FakeTVAE:
    """Fast test double; it does not generate genuine synthetic data."""

    def set_random_state(self, seed):
        self.seed = seed

    def fit(self, frame, discrete_columns):
        self.fit_frame = frame.copy(deep=True)
        self.discrete_columns = list(discrete_columns)

    def sample(self, rows):
        return self.fit_frame.sample(
            n=rows,
            replace=True,
            random_state=self.seed,
        ).reset_index(drop=True)


class InvalidTVAE(FakeTVAE):
    def sample(self, rows):
        frame = super().sample(rows)
        frame["target"] = 2
        return frame


def write_metadata(directory, metadata):
    contents = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (directory / "metadata.json").write_bytes(contents)
    (directory / "metadata.sha256").write_text(
        hashlib.sha256(contents).hexdigest() + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def generation_inputs(tmp_path):
    data_directory = tmp_path / "uci_adult_data"
    data_directory.mkdir()
    types = load_adult_feature_metadata().feature_types

    for filename, start, rows in [
        ("reference.csv", 0, 20),
        ("test.csv", 100, 40),
    ]:
        frame = pd.DataFrame({
            feature: (
                np.arange(rows) % 3
                if types[feature] == "categorical"
                else np.arange(start, start + rows, dtype=float)
            )
            for feature in FEATURE_COLUMNS
        })
        frame["target"] = np.arange(rows) % 2
        frame.to_csv(data_directory / filename, index=False)

    artifact = save_adult_split_manifest(data_root=tmp_path)
    manifest = load_adult_split_manifest(
        artifact.manifest_path.parent,
        data_root=tmp_path,
    )
    repair_ids = manifest["partitions"]["repair"]["source_row_ids"]
    test = pd.read_csv(data_directory / "test.csv")
    repair = test.loc[repair_ids]

    train_ids, validation_ids = train_test_split(
        repair.index.to_numpy(),
        test_size=0.25,
        stratify=repair["target"].to_numpy(),
        random_state=42,
    )
    train_ids = sorted(int(value) for value in train_ids)
    validation_ids = sorted(int(value) for value in validation_ids)

    preparation = tmp_path / "preparation"
    preparation.mkdir()

    files = {
        "generator_fit.csv": test.loc[train_ids].to_csv(
            index=True, index_label="source_row_id"
        ).encode("utf-8"),
        "internal_validation.csv": test.loc[validation_ids].to_csv(
            index=True, index_label="source_row_id"
        ).encode("utf-8"),
        "validation_diagnosed_membership.csv": pd.DataFrame({
            "source_row_id": validation_ids,
            "diagnosed_slice_member": True,
        }).to_csv(index=False).encode("utf-8"),
    }

    for filename, contents in files.items():
        (preparation / filename).write_bytes(contents)

    metadata = {
        "purpose": "development_repair_preparation",
        "provenance": {
            "source_role": "repair",
            "split_manifest_sha256": artifact.manifest_sha256,
            "generator_fit_row_ids": train_ids,
            "internal_validation_row_ids": validation_ids,
        },
        "files": {
            filename: hashlib.sha256(contents).hexdigest()
            for filename, contents in files.items()
        },
    }
    write_metadata(preparation, metadata)

    return {
        "preparation_directory": preparation,
        "manifest_directory": artifact.manifest_path.parent,
        "artifact_version": "test-tvae-v1",
        "data_root": tmp_path,
        "config": TVAEGenerationConfig(
            epochs=1,
            batch_size=8,
            n_rows=10,
            seed=42,
        ),
    }


def test_generation_records_provenance_without_approval(
    generation_inputs, monkeypatch
):
    monkeypatch.setattr(module, "_make_tvae", lambda _: FakeTVAE())
    artifact = generate_tvae_repair(**generation_inputs)
    metadata = json.loads(
        artifact.metadata_path.read_text(encoding="utf-8")
    )
    saved_generator = joblib.load(artifact.directory / "generator.pkl")

    assert metadata["validation_status"] == "UNVALIDATED"
    assert metadata["repair_package_approved"] is False
    assert metadata["quality_metrics"] is None
    assert metadata["privacy_status"] == "NOT_EVALUATED"
    assert metadata["synthetic_rows"] == 10

    assert saved_generator.fit_frame.index.tolist() == (
        metadata["generator_fit_row_ids"]
    )
    assert "source_row_id" not in saved_generator.fit_frame.columns
    assert "target" in saved_generator.discrete_columns
    assert set(metadata["generator_fit_row_ids"]).isdisjoint(
        metadata["internal_validation_row_ids"]
    )

    assert metadata["synthetic_csv_sha256"] == hashlib.sha256(
        artifact.synthetic_path.read_bytes()
    ).hexdigest()
    assert metadata["generator_sha256"] == hashlib.sha256(
        (artifact.directory / "generator.pkl").read_bytes()
    ).hexdigest()


def test_existing_generation_version_is_not_overwritten(
    generation_inputs, monkeypatch
):
    monkeypatch.setattr(module, "_make_tvae", lambda _: FakeTVAE())
    artifact = generate_tvae_repair(**generation_inputs)
    original = artifact.synthetic_path.read_bytes()

    with pytest.raises(FileExistsError):
        generate_tvae_repair(**generation_inputs)

    assert artifact.synthetic_path.read_bytes() == original


def test_modified_fit_file_is_rejected_before_generation(
    generation_inputs, monkeypatch
):
    path = generation_inputs["preparation_directory"] / "generator_fit.csv"
    path.write_bytes(b"modified fitting data")

    def forbidden_generator(_):
        pytest.fail("Generator constructed before rejecting modified data.")

    monkeypatch.setattr(module, "_make_tvae", forbidden_generator)

    with pytest.raises(ValueError, match="Preparation file checksum"):
        generate_tvae_repair(**generation_inputs)


def test_modified_preparation_metadata_is_rejected(generation_inputs):
    path = generation_inputs["preparation_directory"] / "metadata.json"
    path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="metadata checksum"):
        generate_tvae_repair(**generation_inputs)


def test_nonrepair_preparation_is_rejected(generation_inputs):
    directory = generation_inputs["preparation_directory"]
    metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    metadata["provenance"]["source_role"] = "sealed_future"
    write_metadata(directory, metadata)

    with pytest.raises(ValueError, match="repair-only"):
        generate_tvae_repair(**generation_inputs)


def test_invalid_generated_labels_do_not_create_validated_package(
    generation_inputs, monkeypatch
):
    monkeypatch.setattr(module, "_make_tvae", lambda _: InvalidTVAE())

    with pytest.raises(ValueError, match="unknown discrete values"):
        generate_tvae_repair(**generation_inputs)

    directory = (
        generation_inputs["data_root"]
        / "c4_artifacts" / "offline_repair" / "adult"
        / generation_inputs["artifact_version"]
    )
    assert not (directory / "metadata.json").exists()
    assert not (directory / "synthetic_unvalidated.csv").exists()


def test_real_tvae_smoke_run_and_saved_sampling_state(generation_inputs):
    # Uses actual TVAE for one epoch on temporary data.
    artifact = generate_tvae_repair(**generation_inputs)

    synthetic = pd.read_csv(artifact.synthetic_path)
    assert len(synthetic) == 10
    assert synthetic["target"].isin([0, 1]).all()
    assert np.isfinite(synthetic.to_numpy(dtype=float)).all()

    generator = joblib.load(artifact.directory / "generator.pkl")
    replayed = generator.sample(10)

    pd.testing.assert_frame_equal(
        synthetic,
        replayed.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"n_rows": 0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_generation_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        TVAEGenerationConfig(**settings)