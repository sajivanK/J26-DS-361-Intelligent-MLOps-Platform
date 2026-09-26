"""
UCI Adult Census Income - Base Model Training
J26-DS-361 | Shared Artifact | Maintained by Sajivan K (IT23172296)

Model: XGBoost Classifier
Purpose: Production model to be monitored by the drift detection engine

This script:
1. Loads preprocessed reference data
2. Trains XGBoost classifier
3. Evaluates on test set
4. Computes SHAP feature importance
5. Defines high/low importance feature groups
6. Saves model + feature importance JSON
"""

import pandas as pd
import numpy as np
import json
import joblib
import os

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)
import shap
import matplotlib.pyplot as plt


def load_preprocessed_data(data_dir):
    """Load preprocessed reference and test splits."""
    ref_df = pd.read_csv(f"{data_dir}/data/reference.csv")
    test_df = pd.read_csv(f"{data_dir}/data/test.csv")

    X_ref = ref_df.drop(columns=['target'])
    y_ref = ref_df['target'].values

    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target'].values

    print(f"Reference: {X_ref.shape[0]:,} rows, {X_ref.shape[1]} features")
    print(f"Test:      {X_test.shape[0]:,} rows, {X_test.shape[1]} features")
    print(f"Features:  {list(X_ref.columns)}")

    return X_ref, y_ref, X_test, y_test


def train_xgboost(X_ref, y_ref):
    """
    Train XGBoost classifier on reference data.
    Hyperparameters chosen for good generalization on UCI Adult.
    Expected accuracy: ~86-87%
    """
    print("\nTraining XGBoost classifier...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_ref, y_ref,
        eval_set=[(X_ref, y_ref)],
        verbose=False
    )

    print("Training complete.")
    return model


def evaluate_model(model, X_ref, y_ref, X_test, y_test):
    """Full evaluation on reference and test sets."""
    print("\n=== MODEL EVALUATION ===")

    # Reference performance
    y_ref_pred = model.predict(X_ref)
    y_ref_proba = model.predict_proba(X_ref)[:, 1]
    ref_acc = accuracy_score(y_ref, y_ref_pred)
    ref_auc = roc_auc_score(y_ref, y_ref_proba)

    # Test performance
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\nReference set:")
    print(f"  Accuracy: {ref_acc:.4f} ({ref_acc*100:.2f}%)")
    print(f"  AUC:      {ref_auc:.4f}")

    print(f"\nTest set (BASELINE - record this):")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  AUC:      {test_auc:.4f}")

    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred,
                                target_names=['<=50K', '>50K']))

    # Check for overfitting
    acc_gap = ref_acc - test_acc
    if acc_gap > 0.05:
        print(f"WARNING: Possible overfitting. Train-test gap: {acc_gap:.4f}")
    else:
        print(f"Train-test gap: {acc_gap:.4f} (acceptable)")

    return {
        "reference_accuracy": round(ref_acc, 4),
        "reference_auc": round(ref_auc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_auc": round(test_auc, 4)
    }


def compute_shap_importance(model, X_ref, save_dir):
    """
    Compute SHAP values on reference data.
    Use mean absolute SHAP as feature importance.

    Returns feature importance dict sorted by importance (descending).
    """
    print("\n=== SHAP FEATURE IMPORTANCE ===")
    print("Computing SHAP values (this may take 1-2 minutes)...")

    # Use a sample for speed if reference is large
    sample_size = min(2000, len(X_ref))
    X_sample = X_ref.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = list(X_ref.columns)

    importance_dict = dict(zip(feature_names, mean_abs_shap.tolist()))
    importance_sorted = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    print("\nFeature importance (mean |SHAP|):")
    for i, (feat, imp) in enumerate(importance_sorted.items()):
        print(f"  {i+1:2d}. {feat:<20} {imp:.4f}")

    # Plot and save
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar",
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSHAP plot saved: {save_dir}/shap_importance.png")

    return importance_sorted


def define_feature_groups(importance_sorted, n_high=4, n_low=4):
    """
    Define high and low importance feature groups based on SHAP ranking.

    High importance: top N features (drift here = malignant, large accuracy drop)
    Low importance:  bottom N features (drift here = benign, small accuracy drop)

    Args:
        importance_sorted: dict of {feature: shap_importance} sorted descending
        n_high: number of high importance features to include
        n_low:  number of low importance features to include
    """
    features_ranked = list(importance_sorted.keys())

    high_importance = features_ranked[:n_high]
    low_importance = features_ranked[-n_low:]

    print(f"\nHigh importance features (top {n_high}) - MALIGNANT drift group:")
    for feat in high_importance:
        print(f"  {feat}: {importance_sorted[feat]:.4f}")

    print(f"\nLow importance features (bottom {n_low}) - BENIGN drift group:")
    for feat in low_importance:
        print(f"  {feat}: {importance_sorted[feat]:.4f}")

    return high_importance, low_importance


def save_feature_importance(importance_sorted, high_importance,
                            low_importance, metrics, save_dir):
    """Save feature importance and model metadata as JSON."""

    output = {
        "dataset": "UCI Adult Census Income",
        "model": "XGBoost",
        "baseline_metrics": metrics,
        "shap_importance_ranked": importance_sorted,
        "feature_groups": {
            "high_importance": {
                "features": high_importance,
                "description": "Top SHAP features. Drift here causes large accuracy drop (malignant).",
                "n_features": len(high_importance)
            },
            "low_importance": {
                "features": low_importance,
                "description": "Bottom SHAP features. Drift here causes small accuracy drop (benign).",
                "n_features": len(low_importance)
            }
        },
        "notes": [
            "High importance drift = malignant (large accuracy drop expected)",
            "Low importance drift = benign (small accuracy drop expected)",
            "This distinction is the core demonstration of Novelty 1",
            "SHAP computed on sample of 2000 rows from reference split"
        ]
    }

    path = f"{save_dir}/feature_importance_adult.json"
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nFeature importance saved: {path}")
    return output


def run_base_model_pipeline(data_dir, save_dir=None):
    """
    Full base model training pipeline.

    Args:
        data_dir: directory containing data/reference.csv and data/test.csv
        save_dir: directory to save model (defaults to data_dir)
    """
    if save_dir is None:
        save_dir = data_dir

    os.makedirs(f"{save_dir}/models", exist_ok=True)

    print("=" * 60)
    print("UCI ADULT BASE MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading preprocessed data...")
    X_ref, y_ref, X_test, y_test = load_preprocessed_data(data_dir)

    # Step 2: Train model
    print("\n[2/5] Training XGBoost model...")
    model = train_xgboost(X_ref, y_ref)

    # Step 3: Evaluate
    print("\n[3/5] Evaluating model...")
    metrics = evaluate_model(model, X_ref, y_ref, X_test, y_test)

    # Step 4: SHAP importance
    print("\n[4/5] Computing SHAP feature importance...")
    importance_sorted = compute_shap_importance(model, X_ref, save_dir)

    # Step 5: Define feature groups and save
    print("\n[5/5] Defining feature groups and saving...")
    high_importance, low_importance = define_feature_groups(
        importance_sorted, n_high=4, n_low=4
    )
    save_feature_importance(
        importance_sorted, high_importance,
        low_importance, metrics, save_dir
    )

    # Save model
    model_path = f"{save_dir}/models/base_model_adult.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    print("\n" + "=" * 60)
    print("BASE MODEL TRAINING COMPLETE")
    print(f"Baseline accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"Baseline AUC:      {metrics['test_auc']:.4f}")
    print("=" * 60)

    return model, metrics, importance_sorted, high_importance, low_importance


if __name__ == "__main__":
    model, metrics, importance, high_feats, low_feats = run_base_model_pipeline(
        data_dir="shared/uci_adult"
    )
