import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c4.contracts.models import C3RepairPackage
from c4.data import FEATURE_COLUMNS
from c4.models import ChampionConfig, train_candidate


def make_inputs(tmp_path):
    reference = pd.DataFrame({
        feature: [0.0, 1.0, 2.0, 3.0]
        for feature in FEATURE_COLUMNS
    })
    reference["target"] = [0, 1, 0, 1]

    repair = reference.iloc[:2].copy()
    repair.to_csv(tmp_path / "repair.csv", index=False)

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "valid_upstream_input.json"
    )
    fields = json.loads(
        fixture_path.read_text(encoding="utf-8")
    )["c3_repair_package"]

    fields["repair_dataset_uri"] = "repair.csv"
    fields["n_synthetic_rows"] = 2
    fields["augmentation_ratio"] = 0.5

    package = C3RepairPackage.model_validate(fields)
    return reference, package


def small_config():
    return ChampionConfig(
        n_estimators=3,
        max_depth=2,
        n_jobs=1,
    )


def test_candidate_uses_reference_and_repair_without_mutating_reference(
    tmp_path,
):
    reference, package = make_inputs(tmp_path)
    original = reference.copy(deep=True)

    result = train_candidate(
        reference,
        package,
        data_root=tmp_path,
        config=small_config(),
    )

    assert result.reference_rows == 4
    assert result.repair_rows == 2
    assert result.training_rows == 6
    assert result.actual_augmentation_ratio == 0.5
    assert result.model.get_booster().feature_names == list(
        FEATURE_COLUMNS
    )

    probabilities = result.model.predict_proba(
        reference.loc[:, list(FEATURE_COLUMNS)]
    )
    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    pd.testing.assert_frame_equal(reference, original)


@pytest.mark.parametrize(
    "field,value",
    [
        ("repair_package_status", "REJECTED"),
        ("privacy_status", "FAIL"),
    ],
)
def test_candidate_rejects_failed_repair_package(
    tmp_path, field, value
):
    reference, package = make_inputs(tmp_path)
    fields = package.model_dump()
    fields[field] = value
    package = C3RepairPackage.model_validate(fields)

    with pytest.raises(ValueError):
        train_candidate(
            reference,
            package,
            data_root=tmp_path,
            config=small_config(),
        )


def test_candidate_rejects_nonfinite_reference(tmp_path):
    reference, package = make_inputs(tmp_path)
    reference.loc[0, FEATURE_COLUMNS[0]] = float("inf")

    with pytest.raises(ValueError, match="infinite"):
        train_candidate(
            reference,
            package,
            data_root=tmp_path,
            config=small_config(),
        )