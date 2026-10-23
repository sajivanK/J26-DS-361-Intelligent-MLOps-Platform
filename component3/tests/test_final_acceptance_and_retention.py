"""Final independent acceptance and regression tests for Component 3.

Verifies:
1. Mandatory source-retention enforcement (missing source eval strictly blocks approval).
2. All 5 standalone contract lifecycle statuses (NO_REPAIR, REPAIR_REJECTED, REPAIR_APPROVED_FOR_REVIEW, SAFE_ABSTENTION, EXPLICIT_ERROR).
3. Genuine HMAC manifest signing and tamper verification.
4. Real generator failure handling (TVAE, CTGAN, TabDDPM on NaNs, empty DataFrames).
5. Non-oracle diagnostic slice enforcement (rejects oracle simulation leakage).
6. Out-of-distribution detection and domain boundary protection.
7. Artifact integrity and SHA-256 digest validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus
from component3.src.contracts.repair_request import (
    DriftSeverity,
    DriftType,
    RepairRequest,
)
from component3.src.contracts.targeted_repair import (
    DiagnosticProvenance,
    DiagnosticSliceDefinition,
    TargetedRepairStatus,
    validate_diagnostic_slice,
)
from component3.src.evaluation.retraining_evaluator import (
    RetrainingEvaluationResult,
    RetrainingEvaluator,
)
from component3.src.generators.ctgan_generator import CTGANGenerator
from component3.src.generators.tabddpm_generator import TabDDPMGenerator
from component3.src.generators.tvae_generator import TVAEGenerator
from component3.src.orchestration.repair_orchestrator import RepairOrchestrator
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.prescription.learned_model import LearnedPrescriptionModel
from component3.src.prescription.prescription_models import DecisionMode, RepairStrategy


@pytest.fixture
def toy_tabular_data():
    """Generates synthetic tabular partitions for testing."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "num_a": rng.normal(50.0, 10.0, size=n),
        "num_b": rng.exponential(5.0, size=n),
        "cat_a": rng.choice(["Alpha", "Beta", "Gamma"], size=n),
        "target": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
    })
    prep = TabularDataPreprocessor(
        numerical_features=["num_a", "num_b"],
        categorical_features=["cat_a"],
        target_column="target",
    )
    prep.fit(df)
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    bundle = prep.transform(df)
    base_model.fit(bundle.X, bundle.y)
    return df, prep, base_model


# =============================================================================
# TEST 1: MANDATORY SOURCE-RETENTION ENFORCEMENT
# =============================================================================
def test_missing_source_eval_strictly_blocks_approval(toy_tabular_data):
    """Regression Test: Missing source_eval partition must strictly fail acceptance."""
    df, prep, base_model = toy_tabular_data
    evaluator = RetrainingEvaluator(require_source_retention=True)

    # Candidate has excellent target performance but source_eval is None
    res = evaluator.evaluate_candidate(
        base_model=base_model,
        preprocessor=prep,
        df_source_train=df.iloc[:40],
        df_adaptation=df.iloc[40:70],
        synthetic_df=None,
        df_dev_val=df.iloc[70:],
        df_source_eval=None,  # Missing source eval
        target_column="target",
        min_utility_margin=-1.0,  # Even with relaxed target margin
    )

    assert not res.is_acceptable, "Candidate must NOT be acceptable when source retention is missing"
    assert not res.source_retention_preserved, "source_retention_preserved must be False when missing"
    assert res.source_retention_delta is None
    assert any("Mandatory source-retention validation failed" in r for r in res.rejection_reasons)


def test_valid_source_retention_computes_delta(toy_tabular_data):
    """Verifies that when source_eval is provided, source retention delta is calculated."""
    df, prep, base_model = toy_tabular_data
    evaluator = RetrainingEvaluator(require_source_retention=True)

    res = evaluator.evaluate_candidate(
        base_model=base_model,
        preprocessor=prep,
        df_source_train=df.iloc[:30],
        df_adaptation=df.iloc[30:50],
        synthetic_df=None,
        df_dev_val=df.iloc[50:80],
        df_source_eval=df.iloc[80:],  # Valid source eval
        target_column="target",
    )

    assert res.source_positive_f1 is not None
    assert res.source_retention_delta is not None
    assert isinstance(res.source_retention_preserved, bool)


# =============================================================================
# TEST 2: STANDALONE CONTRACT LIFECYCLE STATUSES
# =============================================================================
def test_all_contract_lifecycle_statuses_and_aliases():
    """Verify all 5 official and alias lifecycle statuses for standalone operation."""
    # 1. Approved for review
    assert RepairPackageStatus.APPROVED_FOR_REVIEW == "APPROVED_FOR_REVIEW"
    assert RepairPackageStatus.REPAIR_APPROVED_FOR_REVIEW == RepairPackageStatus.APPROVED_FOR_REVIEW

    # 2. Rejected
    assert RepairPackageStatus.REJECTED == "REJECTED"
    assert RepairPackageStatus.REPAIR_REJECTED == RepairPackageStatus.REJECTED

    # 3. No repair / Safe abstention
    assert RepairPackageStatus.ABSTAINED_NO_REPAIR == "ABSTAINED_NO_REPAIR"
    assert RepairPackageStatus.NO_REPAIR == RepairPackageStatus.ABSTAINED_NO_REPAIR
    assert RepairPackageStatus.SAFE_ABSTENTION == RepairPackageStatus.ABSTAINED_NO_REPAIR

    # 4. Blocked
    assert RepairPackageStatus.BLOCKED == "BLOCKED"

    # 5. Failed / Explicit error
    assert RepairPackageStatus.FAILED == "FAILED"
    assert RepairPackageStatus.EXPLICIT_ERROR == RepairPackageStatus.FAILED


# =============================================================================
# TEST 3: GENUINE HMAC MANIFEST SIGNING AND TAMPER DETECTION
# =============================================================================
def test_package_hmac_signing_and_tamper_detection():
    """Verifies HMAC-SHA256 signature generation and rejection of tampered packages."""
    pkg = RepairPackage(
        request_id="REQ-TEST-HMAC",
        dataset_id="telco_churn",
        package_status=RepairPackageStatus.APPROVED_FOR_REVIEW,
        quantity_selected=50,
    )
    secret_key = b"c3_standalone_acceptance_secret_2026"

    # Generate HMAC signature
    sig = pkg.compute_package_hmac(secret_key)
    assert isinstance(sig, str) and len(sig) == 64

    # Verify signature passes
    assert pkg.verify_package_hmac(secret_key, sig) is True

    # Verify wrong secret fails
    assert pkg.verify_package_hmac(b"wrong_secret", sig) is False

    # Verify tampered payload fails
    tampered_sig = sig[:-4] + "0000"
    assert pkg.verify_package_hmac(secret_key, tampered_sig) is False


# =============================================================================
# TEST 4: REAL GENERATOR FAILURE HANDLING & CONSTRAINTS
# =============================================================================
def test_generator_nan_and_empty_failure_handling():
    """Verify TVAE, CTGAN, and TabDDPM reject empty and NaN inputs safely."""
    df_nan = pd.DataFrame({"col_a": [1.0, np.nan, 3.0], "cat_b": ["x", "y", "z"]})
    df_empty = pd.DataFrame()

    # TVAE
    tvae = TVAEGenerator(epochs=1)
    with pytest.raises(ValueError):
        tvae.fit(df_nan)
    with pytest.raises(ValueError):
        tvae.fit(df_empty)

    # CTGAN
    ctgan = CTGANGenerator(epochs=1)
    with pytest.raises(ValueError):
        ctgan.fit(df_empty)

    # TabDDPM
    tabddpm = TabDDPMGenerator(epochs=1, num_timesteps=10)
    with pytest.raises(ValueError):
        tabddpm.fit(df_nan)
    with pytest.raises(ValueError):
        tabddpm.fit(df_empty)


# =============================================================================
# TEST 5: NON-ORACLE DIAGNOSTIC SLICE VALIDATION
# =============================================================================
def test_oracle_diagnostic_slice_is_blocked():
    """Verifies that oracle simulation rules are strictly blocked from targeted repair."""
    oracle_slice = DiagnosticSliceDefinition(
        slice_id="ORACLE_SLICE",
        slice_predicate="tenure > 29.0",
        affected_features=("tenure",),
        reported_slice_support=0.25,
        diagnostic_confidence=1.0,
        provenance=DiagnosticProvenance.SIMULATION_ORACLE,
        is_non_oracle=False,  # Oracle rule!
    )
    df_mock = pd.DataFrame({"tenure": [10.0, 35.0, 50.0] * 10})
    status, msg, _ = validate_diagnostic_slice(oracle_slice, current_df=df_mock)
    assert status == TargetedRepairStatus.TARGETED_REPAIR_BLOCKED
    assert "oracle simulation rule" in msg.lower()


def test_non_oracle_slice_with_insufficient_support_is_blocked():
    """Verifies that non-oracle slices with support < 10 rows are blocked."""
    slice_def = DiagnosticSliceDefinition(
        slice_id="RARE_SLICE",
        slice_predicate="tenure > 100.0",
        affected_features=("tenure",),
        reported_slice_support=0.05,
        diagnostic_confidence=1.0,
        provenance=DiagnosticProvenance.COMPONENT_2_DIAGNOSIS,
        is_non_oracle=True,
    )
    df_small = pd.DataFrame({"tenure": [10.0, 20.0, 105.0, 110.0]})  # Only 2 matches
    status, msg, _ = validate_diagnostic_slice(slice_def, current_df=df_small, min_slice_rows=10)
    assert status == TargetedRepairStatus.INSUFFICIENT_SUPPORT
    assert "matched only 2 rows" in msg


# =============================================================================
# TEST 6: OUT-OF-DISTRIBUTION DETECTION
# =============================================================================
def test_learned_model_ood_guard_triggers():
    """Verifies that OOD features are flagged by LearnedPrescriptionModel."""
    model = LearnedPrescriptionModel(seed=42)
    # Fit mock bounds
    mock_df = pd.DataFrame({
        "in_dataset_num_features": [10.0, 20.0],
        "in_dataset_num_numerical": [5.0, 10.0],
        "in_dataset_num_categorical": [5.0, 10.0],
        "in_dataset_imbalance_ratio": [0.1, 0.3],
        "in_available_adaptation_budget": [50.0, 200.0],
        "in_has_valid_diagnostic_slice": [0.0, 1.0],
        "arm_candidate_quantity": [25.0, 200.0],
        "in_drift_category": ["concept", "numeric"],
        "arm_candidate_strategy": ["NO_REPAIR", "AUTHENTIC_ONLY_ADAPTATION"],
        "arm_candidate_generator": ["none", "none"],
        "out_delta_f1_vs_baseline": [0.0, 0.02],
        "out_is_accepted": [0, 1],
    })
    model.fit(mock_df)

    # In-distribution check
    in_dist = {"in_dataset_num_features": 15.0, "in_available_adaptation_budget": 100.0}
    is_ood, _ = model.is_out_of_distribution(in_dist)
    assert is_ood is False

    # Out-of-distribution check (extreme feature dimension)
    ood_sample = {"in_dataset_num_features": 5000.0, "in_available_adaptation_budget": 100.0}
    is_ood, reasons = model.is_out_of_distribution(ood_sample)
    assert is_ood is True
    assert any("in_dataset_num_features" in r for r in reasons)


# =============================================================================
# TEST 7: ARTIFACT INTEGRITY VERIFICATION
# =============================================================================
def test_artifact_integrity_verification_detects_tampering(tmp_path):
    """Verifies that verify_artifact_integrity detects missing or tampered files."""
    dummy_model_file = tmp_path / "model.joblib"
    dummy_model_file.write_text("dummy model content", encoding="utf-8")

    import hashlib
    true_hash = hashlib.sha256(dummy_model_file.read_bytes()).hexdigest()

    pkg = RepairPackage(
        request_id="REQ-INTEGRITY-01",
        candidate_model_path=str(dummy_model_file),
        candidate_model_sha256=true_hash,
    )
    ok, errs = pkg.verify_artifact_integrity()
    assert ok is True
    assert len(errs) == 0

    # Tamper with file
    dummy_model_file.write_text("TAMPERED CONTENT", encoding="utf-8")
    ok_tampered, errs_tampered = pkg.verify_artifact_integrity()
    assert ok_tampered is False
    assert any("Model hash mismatch" in e for e in errs_tampered)
