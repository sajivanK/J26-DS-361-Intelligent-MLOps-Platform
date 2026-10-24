import numpy as np
import pandas as pd
import pytest

from c4.data import FEATURE_COLUMNS
from c4.scenarios.adapter import transform_role
from c4.scenarios.config import (
    CATEGORICAL_FEATURES,
    ScenarioConfig,
)
from c4.scenarios.slices import SliceRule


def source():
    frame = pd.DataFrame({
        feature: (
            np.arange(20) % 3
            if feature in CATEGORICAL_FEATURES
            else np.arange(20, dtype=float)
        )
        for feature in FEATURE_COLUMNS
    }, index=np.arange(100, 120))
    frame["target"] = np.arange(20) % 2
    return frame


def slice_rule():
    return SliceRule(
        name="example_true_slice",
        feature="age",
        operator="le",
        value=9.0,
    )


def configuration(mechanism):
    settings = {
        "marginal_multi": {
            "features": ["education_num", "hours_per_week"],
        },
        "interaction": {
            "interaction_pairs": [
                ["marital_status", "relationship"],
            ],
        },
        "combined": {
            "features": ["education_num", "hours_per_week"],
            "interaction_pairs": [
                ["marital_status", "relationship"],
            ],
        },
    }
    return ScenarioConfig(
        mechanism=mechanism,
        magnitude=0.5,
        seed=42,
        **settings[mechanism],
    )


@pytest.mark.parametrize(
    "mechanism",
    ["marginal_multi", "interaction", "combined"],
)
def test_identity_targets_and_nonaffected_rows_are_preserved(mechanism):
    clean = source()
    original = clean.copy(deep=True)

    result = transform_role(
        clean,
        config=configuration(mechanism),
        true_slice=slice_rule(),
        role="release_validation",
    )

    assert result.frame.index.equals(clean.index)
    pd.testing.assert_series_equal(
        result.frame["target"], clean["target"]
    )
    pd.testing.assert_frame_equal(
        result.frame.loc[~result.true_membership],
        clean.loc[~result.true_membership],
    )
    pd.testing.assert_frame_equal(clean, original)
    assert result.provenance["affected_rows"] == 10


def test_same_configuration_is_reproducible():
    settings = {
        "config": configuration("combined"),
        "true_slice": slice_rule(),
        "role": "repair",
    }

    first = transform_role(source(), **settings)
    second = transform_role(source(), **settings)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.provenance == second.provenance


def test_interaction_preserves_feature_value_multiset():
    clean = source()
    result = transform_role(
        clean,
        config=configuration("interaction"),
        true_slice=slice_rule(),
        role="repair",
    )

    mask = result.true_membership

    for feature in FEATURE_COLUMNS:
        assert sorted(result.frame.loc[mask, feature].tolist()) == (
            sorted(clean.loc[mask, feature].tolist())
        )


def test_no_drift_control_is_exact_identity():
    clean = source()
    result = transform_role(
        clean,
        config=ScenarioConfig(
            mechanism="none",
            magnitude=0.0,
            seed=42,
        ),
        true_slice=slice_rule(),
        role="release_validation",
    )

    pd.testing.assert_frame_equal(result.frame, clean)
    assert result.provenance["changed_feature_rows"] == 0


def test_insufficient_affected_rows_are_rejected():
    with pytest.raises(ValueError, match="at least two"):
        transform_role(
            source(),
            config=configuration("marginal_multi"),
            true_slice=SliceRule(
                name="tiny",
                feature="age",
                operator="le",
                value=0.0,
            ),
            role="repair",
        )


def test_sealed_future_role_is_not_accepted():
    with pytest.raises(ValueError, match="roles"):
        transform_role(
            source(),
            config=configuration("combined"),
            true_slice=slice_rule(),
            role="sealed_future",
        )