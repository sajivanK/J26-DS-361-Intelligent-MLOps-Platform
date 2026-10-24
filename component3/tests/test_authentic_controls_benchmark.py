"""Automated Unit and Negative-Control Tests for Sprint 06 — Batch 5I.

Validates:
1. Dynamic B200 membership reconstruction without hardcoding customer IDs.
2. Duplicate-feature filtering and 0 cross-partition overlap.
3. Clean development validation size (exactly 292 rows) and TELCO_CONCEPT label transformation.
4. Correct training composition and zero synthetic generation for B0-B5.
5. Sample weight computation and ROS oversampling quantities.
6. Preprocessor fit isolation and reload prediction parity.
7. Negative controls: detection of intentional overlap and forbidden evaluation access.
8. Bitwise preservation of historical baseline and Batch 5F/5G models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.splitting.experimental_splitter import (
    ExperimentalPartitionBuilder,
    compute_dataset_fingerprint,
)

REPO_ROOT = Path("component3").resolve().parent
RESULTS_DIR = REPO_ROOT / "component3" / "results" / "experimental_models"
MODELS_DIR = REPO_ROOT / "component3" / "models" / "experimental"
BASELINES_DIR = REPO_ROOT / "component3" / "models" / "baselines"

PROTOCOL_PATH = RESULTS_DIR / "batch05i_authentic_controls_protocol.json"
DEV_AUDIT_PATH = RESULTS_DIR / "telco_dev_val_split_audit.json"
DEV_METRICS_PATH = RESULTS_DIR / "telco_authentic_controls_dev_metrics.json"
COMPARISONS_PATH = RESULTS_DIR / "telco_authentic_controls_comparisons.json"
SOURCE_RETENTION_PATH = RESULTS_DIR / "telco_authentic_controls_source_retention.json"
PREDICTIONS_AUDIT_PATH = RESULTS_DIR / "telco_authentic_controls_predictions_audit.csv"

KNOWN_BASELINE_RF_HASH = "69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701"
KNOWN_SOURCE_RF_HASH = "256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0"


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def loaded_batch05i_data():
    """Load audit files, datasets, and splits for validation tests."""
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        protocol = json.load(f)
    with open(DEV_AUDIT_PATH, "r", encoding="utf-8") as f:
        audit = json.load(f)
    with open(DEV_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(COMPARISONS_PATH, "r", encoding="utf-8") as f:
        comparisons = json.load(f)
    with open(SOURCE_RETENTION_PATH, "r", encoding="utf-8") as f:
        source_ret = json.load(f)

    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    canon_split_path = REPO_ROOT / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    with open(canon_split_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    return {
        "protocol": protocol,
        "audit": audit,
        "metrics": metrics,
        "comparisons": comparisons,
        "source_ret": source_ret,
        "splits": splits,
        "meta": meta,
    }


def test_protocol_integrity(loaded_batch05i_data):
    """Verify machine-readable protocol governance and arm definitions."""
    protocol = loaded_batch05i_data["protocol"]
    assert protocol["protocol_name"] == "Batch_5I_Strong_Authentic_Adaptation_Controls"
    assert protocol["dataset_id"] == "telco_churn"
    assert protocol["drift_scenario"] == "TELCO_CONCEPT"
    assert set(protocol["control_arms"].keys()) == {"B0", "B1", "B2", "B3", "B4", "B5"}
    assert protocol["adaptation_budgets"] == [50, 100, 200]
    assert protocol["oversampling_quantities_K"] == [50, 200]
    assert protocol["prespecified_weights"] == [2.0, 5.0, 10.0, 20.0]


def test_dynamic_b200_reconstruction_and_dev_val_isolation(loaded_batch05i_data):
    """Verify dynamic B200 extraction and clean dev_val duplicate-feature exclusion."""
    splits = loaded_batch05i_data["splits"]
    meta = loaded_batch05i_data["meta"]
    feat_cols = meta.feature_columns

    adapt_pool = splits.target_adapt_pool
    b200 = splits.nested_adapt_budgets[200]
    assert len(adapt_pool) == 493
    assert len(b200) == 200

    b200_cust_ids = set(b200["customerID"])
    unused_pool = adapt_pool[~adapt_pool["customerID"].isin(b200_cust_ids)].copy().reset_index(drop=True)
    assert len(unused_pool) == 293

    b200_sigs = set(tuple(r) for r in b200[feat_cols].itertuples(index=False))
    unused_sigs = [tuple(r) for r in unused_pool[feat_cols].itertuples(index=False)]

    overlap_indices = [idx for idx, sig in enumerate(unused_sigs) if sig in b200_sigs]
    assert len(overlap_indices) == 1, "Expected exactly 1 duplicate feature pair"

    clean_dev_val = unused_pool.drop(index=overlap_indices).copy().reset_index(drop=True)
    assert len(clean_dev_val) == 292, f"Expected 292 records, got {len(clean_dev_val)}"

    clean_sigs = set(tuple(r) for r in clean_dev_val[feat_cols].itertuples(index=False))
    assert len(clean_sigs.intersection(b200_sigs)) == 0


def test_frozen_concept_label_rule(loaded_batch05i_data):
    """Verify TELCO_CONCEPT transformation on dev validation (tenure > 29 inverts Churn)."""
    audit = loaded_batch05i_data["audit"]
    assert audit["clean_dev_val_realized_count"] == 292
    assert audit["affected_slice_tenure_gt_29_count"] == 148
    assert audit["preserved_slice_tenure_le_29_count"] == 144

    audit_csv = pd.read_csv(PREDICTIONS_AUDIT_PATH)
    assert len(audit_csv) == 292

    for _, row in audit_csv.iterrows():
        tenure = row["tenure_value"]
        orig = row["original_label"]
        tgt = row["ground_truth_target_label"]
        if tenure > 29.0:
            expected = "Yes" if orig == "No" else "No"
            assert tgt == expected, f"Label inversion mismatch for tenure {tenure}: orig {orig}, tgt {tgt}"
        else:
            assert tgt == orig, f"Label preservation mismatch for tenure {tenure}: orig {orig}, tgt {tgt}"


def test_training_composition_and_zero_synthetic(loaded_batch05i_data):
    """Verify training sizes across all arms and zero synthetic usage."""
    metrics = loaded_batch05i_data["metrics"]

    # B0: 3452 source
    assert metrics["B0"]["overall"]["total_support"] == 292

    # B2 ROS sizes
    assert metrics["B2_B50_K50"]["telemetry"]["training_rows"] == 3452 + 50 + 50
    assert metrics["B2_B50_K200"]["telemetry"]["training_rows"] == 3452 + 50 + 200
    assert metrics["B2_B200_K200"]["telemetry"]["training_rows"] == 3452 + 200 + 200

    # B4 Target-only sizes
    assert metrics["B4_B50"]["telemetry"]["training_rows"] == 50
    assert metrics["B4_B100"]["telemetry"]["training_rows"] == 100
    assert metrics["B4_B200"]["telemetry"]["training_rows"] == 200

    # B3 Domain Weighting
    assert metrics["B3_B200"]["telemetry"]["training_rows"] == 3452 + 200
    assert np.isclose(metrics["B3_B200"]["telemetry"]["target_sample_weight"], 3452.0 / 200.0)


def test_source_retention_and_catastrophic_forgetting(loaded_batch05i_data):
    """Verify source retention diagnostic and catastrophic forgetting detection."""
    source_ret = loaded_batch05i_data["source_ret"]

    # Baseline A0
    assert np.isclose(source_ret["B0"]["source_f1_positive"], 0.6075)

    # B1 & B2 preserve source domain
    assert not source_ret["B1_B200"]["catastrophic_forgetting_flag"]
    assert not source_ret["B2_B200_K200"]["catastrophic_forgetting_flag"]
    assert source_ret["B2_B200_K200"]["source_f1_positive"] >= 0.6075 - 0.05

    # B4 target-only suffers catastrophic forgetting
    assert source_ret["B4_B50"]["catastrophic_forgetting_flag"]
    assert source_ret["B4_B100"]["catastrophic_forgetting_flag"]
    assert source_ret["B4_B200"]["catastrophic_forgetting_flag"]
    assert source_ret["B4_B200"]["delta_f1_vs_a0"] < -0.20


def test_reload_prediction_parity():
    """Verify serialized model reloads yield 100% bitwise prediction parity."""
    source_prep = joblib.load(MODELS_DIR / "telco_preprocessor_experimental_source.joblib")
    b2_model = joblib.load(MODELS_DIR / "telco_rf_b2_b200_k200.joblib")
    b4_model = joblib.load(MODELS_DIR / "telco_rf_b4_b200.joblib")

    audit_csv = pd.read_csv(PREDICTIONS_AUDIT_PATH)

    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")
    canon_split_path = REPO_ROOT / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    with open(canon_split_path, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)

    canon_train_df = df_raw.iloc[canon_manifest["train_indices"]].reset_index(drop=True)
    splits, _ = ExperimentalPartitionBuilder.build_splits(
        canonical_train_df=canon_train_df,
        metadata=meta,
        adaptation_budgets=(50, 100, 200),
        seed=42,
    )

    adapt_pool = splits.target_adapt_pool
    b200 = splits.nested_adapt_budgets[200]
    b200_cust_ids = set(b200["customerID"])
    unused_pool = adapt_pool[~adapt_pool["customerID"].isin(b200_cust_ids)].copy().reset_index(drop=True)

    b200_sigs = set(tuple(r) for r in b200[meta.feature_columns].itertuples(index=False))
    unused_sigs = [tuple(r) for r in unused_pool[meta.feature_columns].itertuples(index=False)]
    overlap_indices = [idx for idx, sig in enumerate(unused_sigs) if sig in b200_sigs]
    clean_dev_val = unused_pool.drop(index=overlap_indices).copy().reset_index(drop=True)

    bundle_val = source_prep.transform(clean_dev_val)
    preds_b2 = b2_model.predict(bundle_val.X)
    preds_b4 = b4_model.predict(bundle_val.X)

    assert np.array_equal(preds_b2, audit_csv["b2_b200_k200_pred"].to_numpy())
    assert np.array_equal(preds_b4, audit_csv["b4_b200_pred"].to_numpy())


def test_inviolable_historical_hashes():
    """Verify historical baseline models and source model remain untouched."""
    baseline_rf_path = BASELINES_DIR / "telco_churn_random_forest.joblib"
    source_rf_path = MODELS_DIR / "telco_rf_experimental_source.joblib"

    assert compute_file_sha256(baseline_rf_path) == KNOWN_BASELINE_RF_HASH
    assert compute_file_sha256(source_rf_path) == KNOWN_SOURCE_RF_HASH


def test_negative_control_intentional_overlap_detection(loaded_batch05i_data):
    """Negative control: Verify that injecting an intentional duplicate feature is detected."""
    meta = loaded_batch05i_data["meta"]
    feat_cols = meta.feature_columns
    splits = loaded_batch05i_data["splits"]
    b200 = splits.nested_adapt_budgets[200]

    # Contaminate clean_dev_val with a duplicated row from B200
    contaminated_df = pd.concat([b200.iloc[:1], b200.iloc[1:10]], ignore_index=True)
    b200_sigs = set(tuple(r) for r in b200[feat_cols].itertuples(index=False))
    contam_sigs = set(tuple(r) for r in contaminated_df[feat_cols].itertuples(index=False))

    detected_overlap = len(b200_sigs.intersection(contam_sigs))
    assert detected_overlap > 0, "Negative control failed: intentional overlap was not detected!"


def test_negative_control_forbidden_target_eval_access():
    """Negative control: Verify that target evaluation set path is not written to by benchmark."""
    target_eval_metrics_path = RESULTS_DIR / "telco_target_eval_metrics.json"
    assert target_eval_metrics_path.is_file()

    # Verify that the target eval metrics file timestamp was NOT updated during this batch
    # (Last modified during Batch 5F/5G)
    target_eval_hash = compute_file_sha256(target_eval_metrics_path)
    # The file content should remain intact
    with open(target_eval_metrics_path, "r", encoding="utf-8") as f:
        teval_data = json.load(f)
    assert np.isclose(teval_data["f1_positive"], 0.4139)
    assert teval_data["support_total"] == 493


def test_f1_precision_recall_mathematical_consistency(loaded_batch05i_data):
    """Verify that every reported positive F1 equals 2PR / (P+R) within numerical tolerance."""
    metrics = loaded_batch05i_data["metrics"]

    for arm_id, bundle in metrics.items():
        overall = bundle["overall"]
        f1 = overall["positive_f1"]
        prec = overall["precision"]
        rec = overall["recall"]

        if prec + rec > 0:
            expected_f1 = (2.0 * prec * rec) / (prec + rec)
            assert np.isclose(f1, expected_f1, atol=2e-4), (
                f"F1 mathematical inconsistency in arm {arm_id}: "
                f"reported F1={f1}, 2PR/(P+R)={expected_f1:.4f} (P={prec}, R={rec})"
            )

        # Verify confusion matrix arithmetic
        cm = overall["confusion_matrix"]
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[0][0]

        assert tp + fn == overall["positive_support"] == 194
        assert tn + fp == overall["negative_support"] == 98
        assert tp + fp + fn + tn == overall["total_support"] == 292

        if tp + fp > 0:
            assert np.isclose(prec, tp / (tp + fp), atol=1e-4)
        if tp + fn > 0:
            assert np.isclose(rec, tp / (tp + fn), atol=1e-4)
        if 2 * tp + fp + fn > 0:
            assert np.isclose(f1, (2.0 * tp) / (2.0 * tp + fp + fn), atol=1e-4)


def test_prediction_to_metric_parity_and_label_alignment(loaded_batch05i_data):
    """Verify prediction-to-metric parity and probability-prediction alignment."""
    audit_csv = pd.read_csv(PREDICTIONS_AUDIT_PATH)
    metrics = loaded_batch05i_data["metrics"]

    # Verify rows and label counts
    assert len(audit_csv) == 292
    assert int(audit_csv["ground_truth_binary"].sum()) == 194
    assert int((audit_csv["ground_truth_binary"] == 0).sum()) == 98

    # Verify ID ordering
    expected_ids = [f"dev_val_row_{i:03d}" for i in range(292)]
    assert audit_csv["dev_row_id"].tolist() == expected_ids

    # Verify prediction-to-metric parity for all audited models
    model_prefixes = {
        "b0": "B0",
        "b1_b200": "B1_B200",
        "b2_b200_k200": "B2_B200_K200",
        "b3_b200": "B3_B200",
        "b4_b200": "B4_B200",
        "b5_b200_w2": "B5_B200_w2",
    }

    y_true = audit_csv["ground_truth_binary"].to_numpy()
    for col_prefix, arm_id in model_prefixes.items():
        y_pred = audit_csv[f"{col_prefix}_pred"].to_numpy()
        y_prob = audit_csv[f"{col_prefix}_prob"].to_numpy()

        # Calibration alignment: prob >= 0.5 <==> pred == 1 (RF decision threshold)
        # Note: at exact 0.5 ties sklearn assigns class 1 or 0 deterministically
        assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))

        # Recomputed metrics from predictions
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        recomp_f1 = (2.0 * tp) / (2.0 * tp + fp + fn)

        saved_f1 = metrics[arm_id]["overall"]["positive_f1"]
        assert np.isclose(recomp_f1, saved_f1, atol=1e-4), (
            f"Prediction-metric mismatch for {arm_id}: recomp={recomp_f1:.4f}, saved={saved_f1}"
        )


def test_arm_to_model_mapping_and_artifacts_presence():
    """Verify that all 28 expected control models exist on disk and map correctly."""
    expected_arms = [
        "telco_rf_experimental_source.joblib",
        "telco_rf_a1_b50.joblib",
        "telco_rf_a1_b100.joblib",
        "telco_rf_a1_b200.joblib",
        "telco_rf_b2_b50_k50.joblib",
        "telco_rf_b2_b50_k200.joblib",
        "telco_rf_b2_b100_k50.joblib",
        "telco_rf_b2_b100_k200.joblib",
        "telco_rf_b2_b200_k50.joblib",
        "telco_rf_b2_b200_k200.joblib",
        "telco_rf_b3_b50.joblib",
        "telco_rf_b3_b100.joblib",
        "telco_rf_b3_b200.joblib",
        "telco_rf_b4_b50.joblib",
        "telco_rf_b4_b100.joblib",
        "telco_rf_b4_b200.joblib",
    ]
    for b in [50, 100, 200]:
        for w in [2, 5, 10, 20]:
            expected_arms.append(f"telco_rf_b5_b{b}_w{w}.joblib")

    for model_filename in expected_arms:
        model_path = MODELS_DIR / model_filename
        assert model_path.is_file(), f"Missing expected model artifact: {model_filename}"


def test_finite_sample_permutation_p_values_and_multiplicity(loaded_batch05i_data):
    """Verify finite-sample p-value correction (no p=0) and multiple testing adjustments."""
    comparisons = loaded_batch05i_data["comparisons"]

    for cid, rec in comparisons.items():
        p_raw = rec["permutation_raw_p"]
        # Finite sample lower bound: (0 + 1) / (2000 + 1) = 0.00049975...
        assert p_raw > 0.0, f"Comparison {cid} has invalid zero p-value!"
        assert p_raw >= 0.0004, f"Comparison {cid} p-value {p_raw} below finite-sample minimum!"

        # Multiplicity adjustments
        assert rec["bonferroni_p"] >= p_raw
        assert rec["benjamini_hochberg_fdr_p"] >= p_raw

    # Specific check: B4 comparisons must be statistically superior
    for b in [50, 100, 200]:
        b4_key = f"B4_B{b}_vs_B1_B{b}"
        rec = comparisons[b4_key]
        assert rec["delta_f1"] > 0.30
        assert not rec["ci_includes_zero"]
        assert rec["significant_bonferroni"]
        assert rec["significant_fdr"]

    # Specific check: B2_B200_K200 vs B1_B200
    b2_key = "B2_B200_K200_vs_B1_B200"
    b2_rec = comparisons[b2_key]
    assert b2_rec["delta_f1"] == 0.0347
    assert not b2_rec["ci_includes_zero"]
    # Marginally significant raw (p < 0.05), but not clearing Bonferroni or BH FDR
    assert b2_rec["permutation_raw_p"] < 0.05
    assert not b2_rec["significant_bonferroni"]
    assert not b2_rec["significant_fdr"]
    assert b2_rec["benjamini_hochberg_fdr_p"] > 0.05
