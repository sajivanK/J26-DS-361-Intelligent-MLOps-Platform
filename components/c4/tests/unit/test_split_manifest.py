import hashlib

import numpy as np
import pandas as pd
import pytest

from c4.data import FEATURE_COLUMNS
from c4.data.split_manifest import (
    assert_release_membership,
    load_adult_split_manifest,
    save_adult_split_manifest,
)


@pytest.fixture
def data_root(tmp_path):
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

    return tmp_path


def test_manifest_records_disjoint_partitions_and_checksum(data_root):
    artifact = save_adult_split_manifest(data_root=data_root)
    manifest = load_adult_split_manifest(
        artifact.manifest_path.parent,
        data_root=data_root,
    )

    assert hashlib.sha256(
        artifact.manifest_path.read_bytes()
    ).hexdigest() == artifact.manifest_sha256

    roles = manifest["partitions"]
    assert roles["repair"]["rows"] == 16
    assert roles["release_validation"]["rows"] == 12
    assert roles["sealed_future"]["rows"] == 12

    sets = [
        set(roles[role]["source_row_ids"])
        for role in ("repair", "release_validation", "sealed_future")
    ]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert set.union(*sets) == set(range(40))


def test_manifest_version_is_not_overwritten(data_root):
    save_adult_split_manifest(data_root=data_root)

    with pytest.raises(FileExistsError):
        save_adult_split_manifest(data_root=data_root)


def test_manifest_tampering_is_detected(data_root):
    artifact = save_adult_split_manifest(data_root=data_root)
    artifact.manifest_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="checksum"):
        load_adult_split_manifest(
            artifact.manifest_path.parent,
            data_root=data_root,
        )


def test_changed_source_is_detected(data_root):
    artifact = save_adult_split_manifest(data_root=data_root)
    path = data_root / "uci_adult_data" / "test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "age"] = 999.0
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Source CSV checksum"):
        load_adult_split_manifest(
            artifact.manifest_path.parent,
            data_root=data_root,
        )


def test_release_membership_rejects_repair_rows(data_root):
    artifact = save_adult_split_manifest(data_root=data_root)
    manifest = load_adult_split_manifest(
        artifact.manifest_path.parent,
        data_root=data_root,
    )
    test = pd.read_csv(data_root / "uci_adult_data" / "test.csv")

    release_ids = manifest["partitions"][
        "release_validation"
    ]["source_row_ids"]
    release = test.loc[release_ids].copy()
    assert_release_membership(release, manifest)

    repair_id = manifest["partitions"]["repair"]["source_row_ids"][0]
    wrong_ids = release.index.tolist()
    wrong_ids[0] = repair_id
    release.index = wrong_ids

    with pytest.raises(ValueError, match="release-validation"):
        assert_release_membership(release, manifest)