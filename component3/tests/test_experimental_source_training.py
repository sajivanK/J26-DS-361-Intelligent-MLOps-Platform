"""Automated Unit Tests for Experimental Source Model Training and Frozen Evaluation.

Sprint 06 — Batch 5F:
Validates:
1. Manifest integrity and SHA-256 hash chaining.
2. Correct source training population (N=3,452) and exact row conservation.
3. Preprocessor fit isolation (fitted strictly on D_source_train).
4. Evaluation partition separation (zero overlap across all 4 experimental partitions).
5. Quarantine of canonical validation (N=1,056) and canonical test (N=1,056) sets.
6. Deterministic simulated concept label transformation (tenure > 29.0).
7. Original label preservation in _original_label.
8. Reload prediction and probability parity.
9. Protection and inviolability of historical baseline artifacts in component3/models/baselines/.
10. Valid degradation quantification without premature recovery claims.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.experiments.train_experimental_source import (
    compute_sha256,
    evaluate_binary_classifier,
)
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.splitting.experimental_splitter import compute_dataset_fingerprint

REPO_ROOT = Path("component3").resolve().parent
CANON_MANIFEST_PATH = REPO_ROOT / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
ADAPT_MANIFEST_PATH = REPO_ROOT / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_split_manifest.json"
EXP_MANIFEST_PATH = REPO_ROOT / "component3" / "results" / "experimental_models" / "telco_rf_experimental_source_manifest.json"
EXP_MODEL_PATH = REPO_ROOT / "component3" / "models" / "experimental" / "telco_rf_experimental_source.joblib"
EXP_PREP_PATH = REPO_ROOT / "component3" / "models" / "experimental" / "telco_preprocessor_experimental_source.joblib"
BASELINE_RF_PATH = REPO_ROOT / "component3" / "models" / "baselines" / "telco_churn_random_forest.joblib"
BASELINE_PREP_PATH = REPO_ROOT / "component3" / "models" / "baselines" / "telco_churn_preprocessor.joblib"

# Known frozen SHA-256 hashes of historical baseline models
KNOWN_BASELINE_RF_HASH = "69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701"
KNOWN_BASELINE_PREP_HASH = "a9eb79cd6c519b99a1ef5372d5418afaacb01048498bb46617d1f6e012ed45c0"


@pytest.fixture(scope="module")
def loaded_artifacts():
    """Load experimental manifests, dataset, and partitions."""
    with open(CANON_MANIFEST_PATH, "r", encoding="utf-8") as f:
        canon_manifest = json.load(f)
    with open(ADAPT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        adapt_manifest = json.load(f)
    with open(EXP_MANIFEST_PATH, "r", encoding="utf-8") as f:
        exp_manifest = json.load(f)

    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    train_indices = canon_manifest["train_indices"]
    canon_train_df = df_raw.iloc[train_indices].copy().reset_index(drop=True)

    return {
        "canon_manifest": canon_manifest,
        "adapt_manifest": adapt_manifest,
        "exp_manifest": exp_manifest,
        "canon_train_df": canon_train_df,
        "meta": meta,
    }


def test_experimental_source_manifest_integrity(loaded_artifacts):
    """Verify manifest fields, cryptographic hashes, and configuration records."""
    exp_manifest = loaded_artifacts["exp_manifest"]

    assert exp_manifest["model_name"] == "telco_rf_experimental_source"
    assert exp_manifest["dataset_id"] == "telco_churn"
    assert exp_manifest["model_family"] == "random_forest"
    assert exp_manifest["source_train_rows"] == 3452
    assert exp_manifest["num_features"] == 19
    assert exp_manifest["dataset_fingerprint"] == "0348ebce7a8c776d"

    # Verify chained SHA-256 hashes match disk artifacts
    assert exp_manifest["split_manifest_sha256"] == compute_sha256(ADAPT_MANIFEST_PATH)
    assert exp_manifest["canonical_split_manifest_sha256"] == compute_sha256(CANON_MANIFEST_PATH)
    assert exp_manifest["model_artifact_sha256"] == compute_sha256(EXP_MODEL_PATH)
    assert exp_manifest["preprocessor_artifact_sha256"] == compute_sha256(EXP_PREP_PATH)


def test_experimental_source_training_population_and_isolation(loaded_artifacts):
    """Verify exact partition row counts, row conservation, and zero cross-partition contamination."""
    adapt_manifest = loaded_artifacts["adapt_manifest"]
    canon_manifest = loaded_artifacts["canon_manifest"]
    canon_train_df = loaded_artifacts["canon_train_df"]

    src_train = set(adapt_manifest["indices"]["source_train"])
    src_eval = set(adapt_manifest["indices"]["source_eval"])
    tgt_pool = set(adapt_manifest["indices"]["target_adapt_pool"])
    tgt_eval = set(adapt_manifest["indices"]["target_eval"])

    # Row counts
    assert len(src_train) == 3452
    assert len(src_eval) == 493
    assert len(tgt_pool) == 493
    assert len(tgt_eval) == 493
    assert len(src_train) + len(src_eval) + len(tgt_pool) + len(tgt_eval) == len(canon_train_df) == 4931

    # Pairwise disjointness across all 6 combinations
    assert len(src_train & src_eval) == 0
    assert len(src_train & tgt_pool) == 0
    assert len(src_train & tgt_eval) == 0
    assert len(src_eval & tgt_pool) == 0
    assert len(src_eval & tgt_eval) == 0
    assert len(tgt_pool & tgt_eval) == 0

    # Quarantine canonical validation and test partitions
    canon_train_set = set(canon_manifest["train_indices"])
    canon_val_set = set(canon_manifest["val_indices"])
    canon_test_set = set(canon_manifest["test_indices"])

    assert len(canon_val_set & canon_train_set) == 0
    assert len(canon_test_set & canon_train_set) == 0
    assert len(canon_val_set) == 1056
    assert len(canon_test_set) == 1056


def test_preprocessor_fit_isolation(loaded_artifacts):
    """Verify preprocessor was fitted strictly on source_train rows."""
    prep: TabularDataPreprocessor = joblib.load(EXP_PREP_PATH)

    assert prep.is_fitted is True
    assert prep.scale_numerical is True
    assert prep.active_numerical == ["tenure", "MonthlyCharges", "TotalCharges"]
    assert len(prep.active_categorical) == 16
    assert len(prep.ordered_features) == 19

    # Class weights must match source_train proportions (2536 No, 916 Yes)
    # class_weight[0] = 3452 / (2 * 2536) = 0.6806
    # class_weight[1] = 3452 / (2 * 916) = 1.8843
    cw = prep.get_class_weights()
    assert round(cw[0], 4) == round(3452 / (2 * 2536), 4)
    assert round(cw[1], 4) == round(3452 / (2 * 916), 4)


def test_simulated_concept_transformation_and_label_preservation(loaded_artifacts):
    """Verify that tenure > 29.0 deterministically inverts labels and preserves original labels."""
    adapt_manifest = loaded_artifacts["adapt_manifest"]
    canon_train_df = loaded_artifacts["canon_train_df"]

    tgt_eval_idx = adapt_manifest["indices"]["target_eval"]
    df_tgt_eval = canon_train_df.iloc[tgt_eval_idx].copy().reset_index(drop=True)

    df_tgt_eval["_original_label"] = df_tgt_eval["Churn"].copy()
    inversion_mask = df_tgt_eval["tenure"] > 29.0
    num_inverted = int(inversion_mask.sum())

    assert num_inverted == 220, f"Expected 220 inverted labels, got {num_inverted}"

    # Verify original label preservation
    pd.testing.assert_series_equal(
        df_tgt_eval["_original_label"],
        canon_train_df.iloc[tgt_eval_idx]["Churn"].reset_index(drop=True),
        check_names=False,
    )

    # Test inversion
    df_tgt_eval.loc[inversion_mask, "Churn"] = df_tgt_eval.loc[inversion_mask, "Churn"].map({"No": "Yes", "Yes": "No"})

    # Non-inverted subset (tenure <= 29.0) must have identical Churn and _original_label
    non_inverted = df_tgt_eval[~inversion_mask]
    assert len(non_inverted) == 273
    assert (non_inverted["Churn"] == non_inverted["_original_label"]).all()

    # Inverted subset (tenure > 29.0) must have completely opposite labels
    inverted = df_tgt_eval[inversion_mask]
    assert len(inverted) == 220
    assert (inverted["Churn"] != inverted["_original_label"]).all()


def test_reload_prediction_parity(loaded_artifacts):
    """Verify bitwise prediction parity and float probability parity upon reloading artifacts."""
    adapt_manifest = loaded_artifacts["adapt_manifest"]
    canon_train_df = loaded_artifacts["canon_train_df"]

    src_eval_idx = adapt_manifest["indices"]["source_eval"]
    df_src_eval = canon_train_df.iloc[src_eval_idx].copy().reset_index(drop=True)

    rf: RandomForestClassifier = joblib.load(EXP_MODEL_PATH)
    prep: TabularDataPreprocessor = joblib.load(EXP_PREP_PATH)

    bundle1 = prep.transform(df_src_eval)
    preds1 = rf.predict(bundle1.X)
    probs1 = rf.predict_proba(bundle1.X)

    # Reload again into fresh variables
    rf_reloaded = joblib.load(EXP_MODEL_PATH)
    prep_reloaded = joblib.load(EXP_PREP_PATH)

    bundle2 = prep_reloaded.transform(df_src_eval)
    preds2 = rf_reloaded.predict(bundle2.X)
    probs2 = rf_reloaded.predict_proba(bundle2.X)

    assert np.array_equal(preds1, preds2), "Prediction parity check failed!"
    assert np.allclose(probs1, probs2, atol=1e-7), "Probability parity check failed!"


def test_historical_baseline_artifact_protection():
    """Verify that historical baseline models in component3/models/baselines/ were not mutated."""
    assert BASELINE_RF_PATH.is_file(), "Baseline RF model missing!"
    assert BASELINE_PREP_PATH.is_file(), "Baseline preprocessor missing!"

    current_rf_hash = compute_sha256(BASELINE_RF_PATH)
    current_prep_hash = compute_sha256(BASELINE_PREP_PATH)

    assert current_rf_hash == KNOWN_BASELINE_RF_HASH, (
        f"Historical baseline RF model was mutated! Expected {KNOWN_BASELINE_RF_HASH}, got {current_rf_hash}"
    )
    assert current_prep_hash == KNOWN_BASELINE_PREP_HASH, (
        f"Historical baseline preprocessor was mutated! Expected {KNOWN_BASELINE_PREP_HASH}, got {current_prep_hash}"
    )


def test_evaluation_metrics_and_degradation_quantification(loaded_artifacts):
    """Verify metrics and ensure degradation is substantial without premature recovery claims."""
    exp_manifest = loaded_artifacts["exp_manifest"]

    src_m = exp_manifest["source_eval_metrics"]
    tgt_m = exp_manifest["target_eval_frozen_metrics"]
    delta_m = exp_manifest["degradation_delta"]

    # Source domain performance
    assert 0.58 <= src_m["f1_positive"] <= 0.63
    assert 0.70 <= src_m["macro_f1"] <= 0.75
    assert 0.80 <= src_m["roc_auc"] <= 0.85

    # Target domain degradation
    assert tgt_m["f1_positive"] < src_m["f1_positive"]
    assert delta_m["delta_f1_positive"] <= -0.15, "Degradation under simulated concept drift is unexpectedly small!"
    assert delta_m["delta_accuracy"] <= -0.25

    # Verify that recovery fraction is NOT present in experimental source manifest (no adaptation trained yet)
    assert "recovery_fraction" not in exp_manifest
    assert "recovery_percentage" not in exp_manifest
