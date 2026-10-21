"""Permanent tests for Component 3 affected slice extraction."""

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from component3.src.contracts.safe_slice_evaluator import (
    evaluate_slice_rule,
    extract_affected_slice,
)
from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import (
    load_c2_diagnosis,
    load_current_dataset,
)


@pytest.fixture(scope="module")
def slice_resources():
    """Load configuration once for extraction tests."""
    return {"config": load_config()}


@pytest.mark.parametrize("scenario_id", [f"S{i:02d}" for i in range(8)])
def test_affected_slice_extraction_scenarios(slice_resources, scenario_id):
    """Verify slice extraction across all controlled drift scenarios."""
    config = slice_resources["config"]
    current_df = load_current_dataset(config, scenario_id)
    diagnosis = load_c2_diagnosis(config, scenario_id)
    original_df = current_df.copy(deep=True)

    rule = diagnosis["affected_slice_rule"]
    mask = evaluate_slice_rule(current_df, rule)
    affected_df = extract_affected_slice(current_df, rule)

    assert len(affected_df) == int(mask.sum()), scenario_id
    assert affected_df.index.equals(current_df.index[mask]), scenario_id
    assert_frame_equal(current_df, original_df)

    reported_support = diagnosis["slice_support"]
    actual_support = len(affected_df) / len(current_df)
    assert abs(reported_support - actual_support) <= 1e-5, f"{scenario_id}: support mismatch"


def test_s00_empty_slice_extraction(slice_resources):
    """S00 baseline scenario must extract an empty affected slice."""
    config = slice_resources["config"]
    current_df = load_current_dataset(config, "S00")
    diagnosis = load_c2_diagnosis(config, "S00")

    affected_df = extract_affected_slice(
        current_df,
        diagnosis["affected_slice_rule"],
    )
    assert len(affected_df) == 0
    assert affected_df.empty


def test_s02_full_batch_extraction(slice_resources):
    """S02 full batch drift must extract all rows and preserve content."""
    config = slice_resources["config"]
    current_df = load_current_dataset(config, "S02")
    diagnosis = load_c2_diagnosis(config, "S02")

    affected_df = extract_affected_slice(
        current_df,
        diagnosis["affected_slice_rule"],
    )
    assert len(affected_df) == len(current_df)
    assert_frame_equal(affected_df, current_df)


def test_extracted_slice_is_independent_copy():
    """Changing an extracted slice must not change the source."""
    df = pd.DataFrame({
        "age": [0.20, 0.75, 0.85, 0.40, 0.95],
        "education_num": [0.10, -0.10, 0.20, -0.20, -0.30],
        "target": [0, 1, 0, 1, 1],
    })
    original = df.copy(deep=True)
    affected = extract_affected_slice(
        df,
        "age > 0.69 and education_num < -0.03",
    )
    assert affected.index.tolist() == [1, 4]
    affected.loc[1, "age"] = 999
    assert_frame_equal(df, original)


def test_extraction_preserves_column_order():
    """Extracted data must retain the original schema and order."""
    df = pd.DataFrame({
        "feature_b": [1, 2, 3],
        "target": [0, 1, 0],
        "feature_a": [10, 20, 30],
    })
    affected = extract_affected_slice(
        df,
        "feature_a > 15",
    )
    assert affected.columns.tolist() == df.columns.tolist()
    assert affected.index.tolist() == [1, 2]
