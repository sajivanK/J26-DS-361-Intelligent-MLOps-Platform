import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from c4.contracts.models import C3RepairPackage
from c4.data import FEATURE_COLUMNS
from c4.repair import load_repair_package


def make_package(tmp_path, *, problem=None):
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "valid_upstream_input.json"
    )
    fields = json.loads(
        fixture.read_text(encoding="utf-8")
    )["c3_repair_package"]

    frame = pd.DataFrame({
        feature: [0.0, 1.0]
        for feature in FEATURE_COLUMNS
    })
    frame["target"] = [0, 1]

    if problem == "schema":
        frame = frame.drop(columns=[FEATURE_COLUMNS[0]])
    elif problem == "target":
        frame["target"] = frame["target"].astype(float)
        frame.loc[0, "target"] = 0.5
    elif problem == "infinite":
        frame.loc[0, FEATURE_COLUMNS[0]] = float("inf")

    path = tmp_path / "repair.csv"
    frame.to_csv(path, index=False)

    fields["repair_dataset_uri"] = "repair.csv"
    fields["n_synthetic_rows"] = 2

    if problem == "row_count":
        fields["n_synthetic_rows"] = 3
    elif problem == "rejected":
        fields["repair_package_status"] = "REJECTED"
    elif problem == "privacy":
        fields["privacy_status"] = "FAIL"

    return C3RepairPackage.model_validate(fields), path


def test_valid_repair_records_exact_file_checksum(tmp_path):
    package, path = make_package(tmp_path)

    loaded = load_repair_package(
        package,
        data_root=tmp_path,
    )

    assert loaded.dataset_path == path.resolve()
    assert loaded.dataset_sha256 == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert loaded.frame.columns.tolist() == [
        *FEATURE_COLUMNS, "target"
    ]
    assert loaded.frame["target"].tolist() == [0, 1]


@pytest.mark.parametrize(
    "problem",
    ["schema", "target", "infinite", "row_count"],
)
def test_invalid_repair_data_is_rejected(tmp_path, problem):
    package, _ = make_package(tmp_path, problem=problem)

    with pytest.raises(ValueError):
        load_repair_package(package, data_root=tmp_path)


@pytest.mark.parametrize("problem", ["rejected", "privacy"])
def test_failed_gate_is_checked_before_file_access(
    tmp_path, problem
):
    package, path = make_package(tmp_path, problem=problem)
    path.unlink()

    with pytest.raises(ValueError):
        load_repair_package(package, data_root=tmp_path)