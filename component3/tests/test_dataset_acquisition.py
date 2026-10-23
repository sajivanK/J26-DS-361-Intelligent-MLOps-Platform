"""Tests for Dataset Acquisition, Verification, and Integrity Checks in Component 3."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import hashlib

from component3.src.datasets.dataset_registry import DatasetMetadata, DatasetStatus, get_dataset_registry
from component3.src.datasets.dataset_verifier import DatasetVerifier, DatasetVerificationReport
from component3.src.datasets.dataset_acquisition import DatasetAcquisitionManager


def test_dataset_verifier_clean_dataframe():
    """Verify that a valid tabular dataset passes all integrity checks."""
    df = pd.DataFrame({
        "age": [25, 30, 45, 50, 60],
        "workclass": ["Private", "Self-emp", "Private", "Gov", "Private"],
        "target": [0, 0, 1, 1, 0],
    })
    meta = DatasetMetadata(
        dataset_id="test_data",
        display_name="Test Tabular Data",
        version="1.0.0",
        source="unit_test",
        license_info="CC0",
        status=DatasetStatus.VERIFIED,
        target_column="target",
        target_mapping={0: 0, 1: 1},
        numerical_features=["age"],
        categorical_features=["workclass"],
        class_distribution={0: 0.6, 1: 0.4},
        leakage_prone_columns=["id"],
    )
    verifier = DatasetVerifier()
    res = verifier.verify(metadata=meta, df=df)

    assert res.is_valid is True
    assert res.duplicate_row_count == 0
    assert len(res.leakage_columns_detected) == 0
    assert res.target_distribution[0] == 0.6
    assert res.target_distribution[1] == 0.4


def test_dataset_verifier_detects_leakage_and_duplicates():
    """Verify that duplicate rows and leakage columns are detected."""
    df = pd.DataFrame({
        "customerID": ["C1", "C2", "C1"],  # Leakage column + duplicate
        "age": [25, 30, 25],
        "target": [0, 1, 0],
    })
    meta = DatasetMetadata(
        dataset_id="test_leakage",
        display_name="Leakage Test",
        version="1.0.0",
        source="unit_test",
        license_info="CC0",
        status=DatasetStatus.VERIFIED,
        target_column="target",
        target_mapping={0: 0, 1: 1},
        numerical_features=["age"],
        categorical_features=[],
        leakage_prone_columns=["customerID"],
    )
    verifier = DatasetVerifier()
    res = verifier.verify(metadata=meta, df=df)

    assert "customerID" in res.leakage_columns_detected
    assert res.duplicate_row_count == 1


def test_dataset_verifier_sha256_checksum():
    """Verify file integrity verification via SHA256 checksum."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write("colA,colB\n1,2\n3,4\n")
        tmp_path = Path(tmp.name)

    try:
        content = tmp_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()

        verifier = DatasetVerifier()
        assert verifier.verify_file_integrity(filepath=tmp_path, expected_hash=actual_hash) is True
        assert verifier.verify_file_integrity(filepath=tmp_path, expected_hash="0000000000000000000000000000000000000000000000000000000000000000") is False
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_dataset_acquisition_pending_status():
    """Verify that datasets requiring manual authentication (Home Credit, IEEE-CIS) return clean pending status."""
    mgr = DatasetAcquisitionManager()
    status = mgr.get_dataset_status("home_credit")
    assert status["status"] == "pending"
    assert "Kaggle API" in status["message"]

    status_ieee = mgr.get_dataset_status("ieee_cis")
    assert status_ieee["status"] == "pending"
    assert "Kaggle API" in status_ieee["message"]


def test_dataset_acquisition_available_datasets():
    """Verify registry lists all 5 candidate datasets and reports correct accessibility."""
    reg = get_dataset_registry()
    datasets = reg.list_datasets()
    assert "uci_adult" in datasets
    assert "telco_churn" in datasets
    assert "bank_marketing" in datasets
    assert "home_credit" in datasets
    assert "ieee_cis" in datasets
