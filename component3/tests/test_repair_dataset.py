"""Unit tests for repair experiment dataset creation and validation."""

import pandas as pd
import pytest

from component3.src.experiments.build_repair_dataset import (
    clean_and_validate_dataset,
)


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame([
        {
            "experiment_id": "exp_01",
            "dataset_id": "uci_adult",
            "drift_scenario": "S04",
            "generator": "CTGAN",
            "target_coverage": 1.0,
            "fidelity_score": 0.75,
            "privacy_status": "PASS",
            "delta_affected_slice_f1": -0.01,
            "label_source": "teacher_pseudo_label",
            "total_seconds": 5.0,
            "prescriptive_viability_label": 1,
        },
        {
            "experiment_id": "exp_02",
            "dataset_id": "uci_adult",
            "drift_scenario": "S01",
            "generator": "TVAE",
            "target_coverage": 0.70,
            "fidelity_score": 0.65,
            "privacy_status": "PASS",
            "delta_affected_slice_f1": -0.05,
            "label_source": "teacher_pseudo_label",
            "total_seconds": 4.0,
            "prescriptive_viability_label": 0,
        },
    ])


def test_clean_and_validate_valid_dataset(sample_raw_df):
    clean_df, audit = clean_and_validate_dataset(sample_raw_df, output_dir=None)
    assert len(clean_df) == 2
    assert audit["raw_record_count"] == 2
    assert audit["clean_record_count"] == 2
    assert audit["duplicate_experiment_ids"] == 0
    assert audit["validation_passed"] is True


def test_clean_and_validate_handles_duplicates(sample_raw_df):
    duplicated_df = pd.concat([sample_raw_df, sample_raw_df.iloc[0:1]], ignore_index=True)
    clean_df, audit = clean_and_validate_dataset(duplicated_df, output_dir=None)
    assert len(clean_df) == 2
    assert audit["duplicate_experiment_ids"] == 1
    assert any(ex["reason"] == "duplicate_experiment_id" for ex in audit["excluded_records"])


def test_clean_and_validate_filters_invalid_metrics(sample_raw_df):
    corrupt_row = sample_raw_df.iloc[0:1].copy()
    corrupt_row["experiment_id"] = "corrupt_exp"
    corrupt_row["target_coverage"] = 1.5  # Impossible coverage > 1.0

    invalid_df = pd.concat([sample_raw_df, corrupt_row], ignore_index=True)
    clean_df, audit = clean_and_validate_dataset(invalid_df, output_dir=None)

    assert len(clean_df) == 2
    assert "corrupt_exp" not in clean_df["experiment_id"].values
    assert any(ex["reason"] == "invalid_coverage_range" for ex in audit["excluded_records"])
