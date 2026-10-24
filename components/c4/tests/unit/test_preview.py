import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from c4.data import FEATURE_COLUMNS
from c4.data.split_manifest import (
    load_adult_split_manifest,
    save_adult_split_manifest,
)
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.preview import save_role_preview
from c4.scenarios.slices import SliceRule


@pytest.fixture
def preview_inputs(tmp_path):
    directory = tmp_path / "uci_adult_data"
    directory.mkdir()

    for filename, start, rows in [
        ("reference.csv", 0, 20),
        ("test.csv", 100, 40),
    ]:
        frame = pd.DataFrame({
            feature: np.arange(start, start + rows, dtype=float)
            for feature in FEATURE_COLUMNS
        })
        frame["target"] = np.arange(rows) % 2
        frame.to_csv(directory / filename, index=False)

    artifact = save_adult_split_manifest(data_root=tmp_path)

    return {
        "manifest_directory": artifact.manifest_path.parent,
        "config": ScenarioConfig(
            mechanism="marginal_multi",
            features=["education_num", "hours_per_week"],
            magnitude=0.5,
            seed=42,
        ),
        "true_slice": SliceRule(
            name="test_slice",
            feature="age",
            operator="le",
            value=120.0,
        ),
        "preview_id": "preview-001",
        "data_root": tmp_path,
    }


@pytest.mark.parametrize("role", ["repair", "release_validation"])
def test_preview_uses_only_verified_role_rows(preview_inputs, role):
    artifact = save_role_preview(**preview_inputs, role=role)
    metadata = json.loads(
        artifact.metadata_path.read_text(encoding="utf-8")
    )
    manifest = load_adult_split_manifest(
        preview_inputs["manifest_directory"],
        data_root=preview_inputs["data_root"],
    )

    expected_ids = manifest["partitions"][role]["source_row_ids"]
    clean = pd.read_csv(artifact.directory / "clean.csv", index_col=0)
    transformed = pd.read_csv(
        artifact.directory / "transformed.csv", index_col=0
    )

    assert clean.index.tolist() == expected_ids
    assert transformed.index.tolist() == expected_ids
    pd.testing.assert_series_equal(
        clean["target"], transformed["target"]
    )

    assert metadata["purpose"] == "development_preview"
    assert metadata["benchmark_execution"] is False

    for filename, expected_hash in metadata["files"].items():
        assert hashlib.sha256(
            (artifact.directory / filename).read_bytes()
        ).hexdigest() == expected_hash

    assert hashlib.sha256(
        artifact.metadata_path.read_bytes()
    ).hexdigest() == artifact.metadata_sha256


def test_preview_cannot_be_overwritten(preview_inputs):
    save_role_preview(**preview_inputs, role="repair")

    with pytest.raises(FileExistsError):
        save_role_preview(**preview_inputs, role="repair")


def test_future_preview_is_blocked(preview_inputs):
    with pytest.raises(ValueError, match="roles"):
        save_role_preview(**preview_inputs, role="sealed_future")


def test_changed_source_blocks_preview(preview_inputs):
    root = preview_inputs["data_root"]
    path = root / "uci_adult_data" / "test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "age"] = 999.0
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Source CSV checksum"):
        save_role_preview(**preview_inputs, role="repair")


def test_invalid_preview_id_is_rejected(preview_inputs):
    preview_inputs["preview_id"] = "../other"

    with pytest.raises(ValueError, match="Preview ID"):
        save_role_preview(**preview_inputs, role="repair")