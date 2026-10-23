"""Dedicated Experimental Source Model Trainer and Frozen Baseline Evaluator.

Sprint 06 — Batch 5F:
Trains one separately named experimental Random Forest source model on the verified
D_source_train partition (N=3,452), providing the frozen no-adaptation control (A0)
for subsequent authentic-data adaptation and synthetic-data repair experiments.

Strict Scientific and Isolation Guarantees:
1. Preprocessor is fitted EXCLUSIVELY on D_source_train.
2. Random Forest is fitted EXCLUSIVELY on D_source_train.
3. Zero data leakage: programmatic verification that D_source_eval, D_target_adapt_pool,
   D_target_eval, canonical validation, and canonical test are quarantined.
4. Input provenance: SHA-256 fingerprint and manifest hash verification prior to training.
5. Preserves all historical baseline models in component3/models/baselines/ without mutation.
6. Evaluates frozen model on D_source_eval (observational) and D_target_eval (simulated concept truth).
7. Validates reload parity and artifact SHA-256 integrity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from component3.src.datasets.dataset_registry import get_dataset_registry
from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor
from component3.src.splitting.experimental_splitter import compute_dataset_fingerprint


@dataclass
class EvaluationMetrics:
    """Standardized performance metrics for binary classification evaluation."""
    f1_positive: float
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    precision_positive: float
    recall_positive: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]]
    support_total: int
    class_counts: dict[str, int]
    class_proportions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentalSourceModelManifest:
    """Provenance and audit record for the experimental Random Forest source model."""
    model_name: str
    dataset_id: str
    model_family: str
    hyperparameters: dict[str, Any]
    training_seed: int
    source_train_rows: int
    training_label_distribution: dict[str, int]
    training_label_proportions: dict[str, float]
    feature_names: list[str]
    num_features: int
    dataset_fingerprint: str
    split_manifest_sha256: str
    canonical_split_manifest_sha256: str
    preprocessor_fitted_population: int
    training_duration_seconds: float
    peak_memory_mb: float
    dependency_versions: dict[str, str]
    model_artifact_path: str
    preprocessor_artifact_path: str
    model_artifact_sha256: str
    preprocessor_artifact_sha256: str
    reload_prediction_parity: bool
    reload_probability_parity: bool
    historical_baseline_integrity_verified: bool
    source_eval_metrics: dict[str, Any]
    target_eval_frozen_metrics: dict[str, Any]
    degradation_delta: dict[str, float]
    target_labels_inverted: int
    target_eval_total_rows: int
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path | str) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate_binary_classifier(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    label_names: tuple[str, str] = ("No", "Yes"),
) -> EvaluationMetrics:
    """Evaluate a binary classifier and calculate comprehensive performance metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    f1_pos = float(f1_score(y, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y, y_pred))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    prec_pos = float(precision_score(y, y_pred, zero_division=0))
    rec_pos = float(recall_score(y, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y, y_prob)) if y_prob is not None else 0.5
    pr_auc = float(average_precision_score(y, y_prob)) if y_prob is not None else 0.5
    cm = confusion_matrix(y, y_pred).tolist()

    unique, counts = np.unique(y, return_counts=True)
    count_dict = {label_names[int(u)]: int(c) for u, c in zip(unique, counts)}
    total = len(y)
    prop_dict = {label_names[int(u)]: round(float(c / total), 4) for u, c in zip(unique, counts)}

    return EvaluationMetrics(
        f1_positive=round(f1_pos, 4),
        macro_f1=round(macro_f1, 4),
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        precision_positive=round(prec_pos, 4),
        recall_positive=round(rec_pos, 4),
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        confusion_matrix=cm,
        support_total=int(total),
        class_counts=count_dict,
        class_proportions=prop_dict,
    )


def train_experimental_source_model(
    repo_root: Path | str = ".",
    seed: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
    n_jobs: int = 2,
) -> tuple[RandomForestClassifier, TabularDataPreprocessor, ExperimentalSourceModelManifest]:
    """Execute the end-to-end experimental source model training and frozen evaluation workflow."""
    root = Path(repo_root)

    # -------------------------------------------------------------------------
    # STEP 1: Verify and Freeze Input Provenance
    # -------------------------------------------------------------------------
    canon_split_path = root / "component3" / "results" / "split_manifests" / "telco_churn_split_manifest.json"
    adapt_split_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_split_manifest.json"
    leakage_audit_path = root / "component3" / "results" / "adaptation_splits" / "telco_churn_adaptation_group_overlap_audit.json"
    baseline_rf_path = root / "component3" / "models" / "baselines" / "telco_churn_random_forest.joblib"
    baseline_prep_path = root / "component3" / "models" / "baselines" / "telco_churn_preprocessor.joblib"

    if not canon_split_path.is_file():
        raise FileNotFoundError(f"Canonical split manifest missing: {canon_split_path}")
    if not adapt_split_path.is_file():
        raise FileNotFoundError(f"Experimental adaptation split manifest missing: {adapt_split_path}")
    if not leakage_audit_path.is_file():
        raise FileNotFoundError(f"Leakage audit artifact missing: {leakage_audit_path}")
    if not baseline_rf_path.is_file():
        raise FileNotFoundError(f"Baseline RF artifact missing: {baseline_rf_path}")
    if not baseline_prep_path.is_file():
        raise FileNotFoundError(f"Baseline preprocessor artifact missing: {baseline_prep_path}")

    canon_manifest_sha256 = compute_sha256(canon_split_path)
    adapt_manifest_sha256 = compute_sha256(adapt_split_path)
    baseline_rf_initial_sha256 = compute_sha256(baseline_rf_path)
    baseline_prep_initial_sha256 = compute_sha256(baseline_prep_path)

    with canon_split_path.open("r", encoding="utf-8") as f:
        canon_manifest = json.load(f)
    with adapt_split_path.open("r", encoding="utf-8") as f:
        adapt_manifest = json.load(f)
    with leakage_audit_path.open("r", encoding="utf-8") as f:
        leakage_audit = json.load(f)

    # Assert audit PASS and zero overlaps
    if leakage_audit.get("leakage_audit_status") != "PASS":
        raise ValueError(f"Leakage audit status is not PASS: {leakage_audit.get('leakage_audit_status')}")
    if leakage_audit.get("total_cross_partition_index_overlaps", -1) != 0:
        raise ValueError("Nonzero cross-partition index overlaps detected in audit!")
    if leakage_audit.get("total_cross_partition_group_overlaps", -1) != 0:
        raise ValueError("Nonzero cross-partition feature group overlaps detected in audit!")

    # Verify partition counts
    rc = adapt_manifest["realized_counts"]
    if rc["source_train"] != 3452 or rc["source_eval"] != 493 or rc["target_adapt_pool"] != 493 or rc["target_eval"] != 493:
        raise ValueError(f"Unexpected realized partition counts: {rc}")

    # Load dataset and extract canonical train partition
    reg = get_dataset_registry()
    df_raw = reg.load_dataset("telco_churn")
    meta = reg.get("telco_churn")

    train_indices = canon_manifest["train_indices"]
    val_indices = canon_manifest["val_indices"]
    test_indices = canon_manifest["test_indices"]

    canon_train_df = df_raw.iloc[train_indices].copy().reset_index(drop=True)
    if len(canon_train_df) != 4931:
        raise ValueError(f"Expected canonical train rows = 4,931, got {len(canon_train_df)}")

    # Verify dataset fingerprint
    current_fp = compute_dataset_fingerprint(canon_train_df)
    if current_fp != adapt_manifest["source_dataset_fingerprint"]:
        raise ValueError(
            f"Dataset fingerprint mismatch! Expected {adapt_manifest['source_dataset_fingerprint']}, got {current_fp}"
        )

    # Extract experimental partitions
    src_train_idx = set(adapt_manifest["indices"]["source_train"])
    src_eval_idx = set(adapt_manifest["indices"]["source_eval"])
    tgt_pool_idx = set(adapt_manifest["indices"]["target_adapt_pool"])
    tgt_eval_idx = set(adapt_manifest["indices"]["target_eval"])

    # Explicit isolation safeguards against accidental data leakage
    assert len(src_train_idx) == 3452, f"Expected 3452 source train rows, got {len(src_train_idx)}"
    assert len(src_eval_idx) == 493, f"Expected 493 source eval rows, got {len(src_eval_idx)}"
    assert len(tgt_pool_idx) == 493, f"Expected 493 target pool rows, got {len(tgt_pool_idx)}"
    assert len(tgt_eval_idx) == 493, f"Expected 493 target eval rows, got {len(tgt_eval_idx)}"

    # Ensure zero overlap
    assert len(src_train_idx & src_eval_idx) == 0, "Source train overlaps with source eval!"
    assert len(src_train_idx & tgt_pool_idx) == 0, "Source train overlaps with target pool!"
    assert len(src_train_idx & tgt_eval_idx) == 0, "Source train overlaps with target eval!"
    assert len(src_eval_idx & tgt_pool_idx) == 0, "Source eval overlaps with target pool!"
    assert len(src_eval_idx & tgt_eval_idx) == 0, "Source eval overlaps with target eval!"
    assert len(tgt_pool_idx & tgt_eval_idx) == 0, "Target pool overlaps with target eval!"

    # Ensure canonical validation and test sets were never touched
    train_idx_set = set(train_indices)
    val_idx_set = set(val_indices)
    test_idx_set = set(test_indices)
    assert len(val_idx_set & train_idx_set) == 0, "Canonical train and val intersect!"
    assert len(test_idx_set & train_idx_set) == 0, "Canonical train and test intersect!"

    # Construct DataFrames
    df_source_train = canon_train_df.iloc[sorted(src_train_idx)].copy().reset_index(drop=True)
    df_source_eval = canon_train_df.iloc[sorted(src_eval_idx)].copy().reset_index(drop=True)
    df_target_eval = canon_train_df.iloc[sorted(tgt_eval_idx)].copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # STEP 2: Dedicated Preprocessor Fitting (EXCLUSIVELY on D_source_train)
    # -------------------------------------------------------------------------
    preprocessor = TabularDataPreprocessor.from_dataset_metadata(meta, scale_numerical=True)
    preprocessor.fit(df_source_train)

    train_bundle = preprocessor.transform(df_source_train)
    X_train = train_bundle.X
    y_train = train_bundle.y

    if X_train.shape != (3452, len(preprocessor.ordered_features)):
        raise ValueError(f"Unexpected X_train shape: {X_train.shape}")
    if len(y_train) != 3452:
        raise ValueError(f"Unexpected y_train length: {len(y_train)}")

    train_counts = pd.Series(y_train).value_counts().to_dict()
    train_label_dist = {"No": int(train_counts.get(0, 0)), "Yes": int(train_counts.get(1, 0))}
    train_label_props = {
        "No": round(train_label_dist["No"] / 3452, 4),
        "Yes": round(train_label_dist["Yes"] / 3452, 4),
    }

    # -------------------------------------------------------------------------
    # STEP 3: Train Experimental Random Forest (EXCLUSIVELY on D_source_train)
    # -------------------------------------------------------------------------
    rf_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": n_jobs,
        "bootstrap": True,
        "criterion": "gini",
    }
    rf_model = RandomForestClassifier(**rf_params)

    # Track CPU runtime and memory
    mem_start = psutil.Process().memory_info().rss / (1024 * 1024) if HAS_PSUTIL else 0.0
    t0 = time.perf_counter()
    rf_model.fit(X_train, y_train)
    train_duration = time.perf_counter() - t0
    mem_end = psutil.Process().memory_info().rss / (1024 * 1024) if HAS_PSUTIL else 0.0
    peak_mem = max(mem_start, mem_end)

    # -------------------------------------------------------------------------
    # STEP 4: Frozen Source-Domain Evaluation on D_source_eval (Observational)
    # -------------------------------------------------------------------------
    source_eval_bundle = preprocessor.transform(df_source_eval)
    X_source_eval = source_eval_bundle.X
    y_source_eval = source_eval_bundle.y

    source_eval_metrics = evaluate_binary_classifier(
        model=rf_model,
        X=X_source_eval,
        y=y_source_eval,
        label_names=("No", "Yes"),
    )

    # -------------------------------------------------------------------------
    # STEP 5: Target-Domain Evaluation Protocol (Simulated Concept Truth)
    # -------------------------------------------------------------------------
    # Apply frozen TELCO_CONCEPT rule: tenure > 29.0 inverts Churn
    df_target_eval_sim = df_target_eval.copy()
    df_target_eval_sim["_original_label"] = df_target_eval_sim["Churn"].copy()
    inversion_mask = df_target_eval_sim["tenure"] > 29.0
    num_inverted = int(inversion_mask.sum())

    # Deterministic inversion
    label_flip_map = {"No": "Yes", "Yes": "No"}
    df_target_eval_sim.loc[inversion_mask, "Churn"] = df_target_eval_sim.loc[inversion_mask, "Churn"].map(label_flip_map)
    df_target_eval_sim["_concept_provenance"] = "SYNTHETIC_CONCEPT_TRUTH"

    target_eval_bundle = preprocessor.transform(df_target_eval_sim)
    X_target_eval = target_eval_bundle.X
    y_target_eval = target_eval_bundle.y

    target_eval_metrics = evaluate_binary_classifier(
        model=rf_model,
        X=X_target_eval,
        y=y_target_eval,
        label_names=("No", "Yes"),
    )

    # -------------------------------------------------------------------------
    # STEP 6: Quantify Degradation (Different Records & Simulated Drift Rule)
    # -------------------------------------------------------------------------
    delta_metrics = {
        "delta_f1_positive": round(target_eval_metrics.f1_positive - source_eval_metrics.f1_positive, 4),
        "delta_macro_f1": round(target_eval_metrics.macro_f1 - source_eval_metrics.macro_f1, 4),
        "delta_accuracy": round(target_eval_metrics.accuracy - source_eval_metrics.accuracy, 4),
        "delta_balanced_accuracy": round(target_eval_metrics.balanced_accuracy - source_eval_metrics.balanced_accuracy, 4),
        "delta_roc_auc": round(target_eval_metrics.roc_auc - source_eval_metrics.roc_auc, 4),
        "delta_pr_auc": round(target_eval_metrics.pr_auc - source_eval_metrics.pr_auc, 4),
    }

    # -------------------------------------------------------------------------
    # STEP 7: Serialization and Reload Parity Verification
    # -------------------------------------------------------------------------
    models_exp_dir = root / "component3" / "models" / "experimental"
    models_exp_dir.mkdir(parents=True, exist_ok=True)

    rf_save_path = models_exp_dir / "telco_rf_experimental_source.joblib"
    prep_save_path = models_exp_dir / "telco_preprocessor_experimental_source.joblib"

    joblib.dump(rf_model, rf_save_path)
    joblib.dump(preprocessor, prep_save_path)

    # Calculate artifact hashes
    rf_artifact_sha256 = compute_sha256(rf_save_path)
    prep_artifact_sha256 = compute_sha256(prep_save_path)

    # Reload from disk and verify exact prediction parity
    reloaded_rf: RandomForestClassifier = joblib.load(rf_save_path)
    reloaded_prep: TabularDataPreprocessor = joblib.load(prep_save_path)

    # Source eval parity
    reloaded_src_bundle = reloaded_prep.transform(df_source_eval)
    y_src_pred_orig = rf_model.predict(X_source_eval)
    y_src_pred_reload = reloaded_rf.predict(reloaded_src_bundle.X)
    y_src_prob_orig = rf_model.predict_proba(X_source_eval)
    y_src_prob_reload = reloaded_rf.predict_proba(reloaded_src_bundle.X)

    pred_parity_src = bool(np.array_equal(y_src_pred_orig, y_src_pred_reload))
    prob_parity_src = bool(np.allclose(y_src_prob_orig, y_src_prob_reload, atol=1e-7))

    # Target eval parity
    reloaded_tgt_bundle = reloaded_prep.transform(df_target_eval_sim)
    y_tgt_pred_orig = rf_model.predict(X_target_eval)
    y_tgt_pred_reload = reloaded_rf.predict(reloaded_tgt_bundle.X)
    y_tgt_prob_orig = rf_model.predict_proba(X_target_eval)
    y_tgt_prob_reload = reloaded_rf.predict_proba(reloaded_tgt_bundle.X)

    pred_parity_tgt = bool(np.array_equal(y_tgt_pred_orig, y_tgt_pred_reload))
    prob_parity_tgt = bool(np.allclose(y_tgt_prob_orig, y_tgt_prob_reload, atol=1e-7))

    overall_pred_parity = pred_parity_src and pred_parity_tgt
    overall_prob_parity = prob_parity_src and prob_parity_tgt

    if not overall_pred_parity or not overall_prob_parity:
        raise AssertionError("Reload parity check failed! Inconsistent predictions between memory and disk models.")

    # Verify historical baseline artifacts were not mutated
    baseline_rf_current_sha256 = compute_sha256(baseline_rf_path)
    baseline_prep_current_sha256 = compute_sha256(baseline_prep_path)

    baseline_integrity_verified = (
        baseline_rf_current_sha256 == baseline_rf_initial_sha256
        and baseline_prep_current_sha256 == baseline_prep_initial_sha256
    )
    if not baseline_integrity_verified:
        raise AssertionError("CRITICAL VIOLATION: Historical baseline artifacts were unexpectedly mutated!")

    # -------------------------------------------------------------------------
    # STEP 8: Construct and Save Experimental Source Model Manifest
    # -------------------------------------------------------------------------
    manifest = ExperimentalSourceModelManifest(
        model_name="telco_rf_experimental_source",
        dataset_id="telco_churn",
        model_family="random_forest",
        hyperparameters=rf_params,
        training_seed=seed,
        source_train_rows=len(df_source_train),
        training_label_distribution=train_label_dist,
        training_label_proportions=train_label_props,
        feature_names=preprocessor.ordered_features,
        num_features=len(preprocessor.ordered_features),
        dataset_fingerprint=current_fp,
        split_manifest_sha256=adapt_manifest_sha256,
        canonical_split_manifest_sha256=canon_manifest_sha256,
        preprocessor_fitted_population=len(df_source_train),
        training_duration_seconds=round(train_duration, 4),
        peak_memory_mb=round(peak_mem, 2),
        dependency_versions={
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        model_artifact_path=str(rf_save_path.relative_to(root)),
        preprocessor_artifact_path=str(prep_save_path.relative_to(root)),
        model_artifact_sha256=rf_artifact_sha256,
        preprocessor_artifact_sha256=prep_artifact_sha256,
        reload_prediction_parity=overall_pred_parity,
        reload_probability_parity=overall_prob_parity,
        historical_baseline_integrity_verified=baseline_integrity_verified,
        source_eval_metrics=source_eval_metrics.to_dict(),
        target_eval_frozen_metrics=target_eval_metrics.to_dict(),
        degradation_delta=delta_metrics,
        target_labels_inverted=num_inverted,
        target_eval_total_rows=len(df_target_eval),
    )

    results_dir = root / "component3" / "results" / "experimental_models"
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = results_dir / "telco_rf_experimental_source_manifest.json"
    manifest.save(manifest_path)

    # Save detailed evaluation summaries
    source_eval_summary_path = results_dir / "telco_source_eval_metrics.json"
    with source_eval_summary_path.open("w", encoding="utf-8") as f:
        json.dump(source_eval_metrics.to_dict(), f, indent=2)

    target_eval_summary_path = results_dir / "telco_target_eval_metrics.json"
    with target_eval_summary_path.open("w", encoding="utf-8") as f:
        json.dump(target_eval_metrics.to_dict(), f, indent=2)

    degradation_summary_path = results_dir / "telco_degradation_comparison.json"
    with degradation_summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_eval_metrics": source_eval_metrics.to_dict(),
                "target_eval_frozen_metrics": target_eval_metrics.to_dict(),
                "degradation_deltas": delta_metrics,
                "target_labels_inverted": num_inverted,
                "target_eval_total_rows": len(df_target_eval),
                "inversion_rule": "tenure > 29.0",
                "simulated_drift_scenario": "TELCO_CONCEPT",
                "label_provenance": "SYNTHETIC_CONCEPT_TRUTH",
                "scientific_note": (
                    "Source and target evaluations use different row instances and differing label generating rules. "
                    "Performance differences reflect both sample variance and synthetic concept drift; "
                    "no adaptation model has been trained and no recovery percentage can be claimed."
                ),
            },
            f,
            indent=2,
        )

    return rf_model, preprocessor, manifest


if __name__ == "__main__":
    print("=" * 70)
    print("Training Experimental Random Forest Source Model (Batch 5F)")
    print("=" * 70)

    model, prep, manifest = train_experimental_source_model(
        repo_root=".",
        seed=42,
        n_estimators=100,
        max_depth=10,
        n_jobs=2,
    )

    print(f"Model saved: {manifest.model_artifact_path}")
    print(f"Preprocessor saved: {manifest.preprocessor_artifact_path}")
    print(f"Training rows: {manifest.source_train_rows}")
    print(f"Training duration: {manifest.training_duration_seconds:.2f}s | Peak RAM: {manifest.peak_memory_mb:.1f} MB")
    print(f"Source Eval F1: {manifest.source_eval_metrics['f1_positive']:.4f}")
    print(f"Target Eval Frozen F1: {manifest.target_eval_frozen_metrics['f1_positive']:.4f}")
    print(f"Degradation Delta F1: {manifest.degradation_delta['delta_f1_positive']:.4f}")
    print(f"Labels Inverted: {manifest.target_labels_inverted} / {manifest.target_eval_total_rows}")
    print(f"Reload Parity: {manifest.reload_prediction_parity}")
    print(f"Historical Baseline Integrity: {manifest.historical_baseline_integrity_verified}")
    print("=" * 70)
