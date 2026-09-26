"""
IEEE-CIS Fraud Detection - Base Model Training
J26-DS-361 | Shared Artifact | Maintained by Sajivan K (IT23172296)

Model: LightGBM Classifier
Why LightGBM:
  - Much faster than XGBoost on 200K rows × 100+ features
  - Native handling of missing values
  - class_weight='balanced' handles 3.5% fraud rate
  - Good calibrated probabilities for confidence statistics
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
import matplotlib.pyplot as plt


def load_preprocessed_data(data_dir):
    """Load preprocessed reference and test splits."""
    print("Loading preprocessed data...")

    ref_df = pd.read_csv(f"{data_dir}/data/reference.csv")
    test_df = pd.read_csv(f"{data_dir}/data/test.csv")

    X_ref = ref_df.drop(columns=['target'])
    y_ref = ref_df['target'].values

    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target'].values

    fraud_rate = y_ref.mean()
    print(f"Reference: {X_ref.shape[0]:,} rows, {X_ref.shape[1]} features")
    print(f"Test:      {X_test.shape[0]:,} rows, {X_test.shape[1]} features")
    print(f"Fraud rate (reference): {fraud_rate*100:.2f}%")

    return X_ref, y_ref, X_test, y_test


def train_lightgbm(X_ref, y_ref):
    """
    Train LightGBM classifier on reference data.

    Key decisions:
    - class_weight='balanced': handles severe class imbalance (3.5% fraud)
    - n_estimators=500: enough trees without overfitting
    - Evaluate by AUC not accuracy (imbalanced problem)
    """
    print("\nTraining LightGBM classifier...")
    print(f"  Training samples: {len(X_ref):,}")
    print(f"  Features: {X_ref.shape[1]}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # Use early stopping with validation set
    val_size = min(10000, int(0.1 * len(X_ref)))
    val_idx = np.random.choice(len(X_ref), val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(len(X_ref)), val_idx)

    X_train = X_ref.iloc[train_idx]
    y_train = y_ref[train_idx]
    X_val = X_ref.iloc[val_idx]
    y_val = y_ref[val_idx]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )

    print(f"  Best iteration: {model.best_iteration_}")
    print("Training complete.")
    return model


def evaluate_model(model, X_ref, y_ref, X_test, y_test):
    """
    Full evaluation. For fraud detection, AUC and PR-AUC
    are more meaningful than raw accuracy.
    """
    print("\n=== MODEL EVALUATION ===")

    # Reference
    y_ref_proba = model.predict_proba(X_ref)[:, 1]
    y_ref_pred = (y_ref_proba >= 0.5).astype(int)
    ref_auc = roc_auc_score(y_ref, y_ref_proba)
    ref_pr_auc = average_precision_score(y_ref, y_ref_proba)
    ref_acc = accuracy_score(y_ref, y_ref_pred)

    # Test
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nReference set:")
    print(f"  Accuracy: {ref_acc:.4f} ({ref_acc*100:.2f}%)")
    print(f"  AUC:      {ref_auc:.4f}")
    print(f"  PR-AUC:   {ref_pr_auc:.4f}")

    print(f"\nTest set (BASELINE - record this):")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  AUC:      {test_auc:.4f}  ← primary metric")
    print(f"  PR-AUC:   {test_pr_auc:.4f}")

    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Legitimate', 'Fraud']))

    print(f"\nConfusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    return {
        "reference_accuracy": round(ref_acc, 4),
        "reference_auc": round(ref_auc, 4),
        "reference_pr_auc": round(ref_pr_auc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_auc": round(test_auc, 4),
        "test_pr_auc": round(test_pr_auc, 4),
        "primary_metric": "AUC (fraud detection is imbalanced)"
    }


def compute_feature_importance(model, X_ref, save_dir, top_n=50):
    """
    Use LightGBM's built-in feature importance (gain-based).
    Gain importance is more reliable than split count for this task.

    Returns top_n features sorted by importance.
    """
    print(f"\n=== FEATURE IMPORTANCE (LightGBM Gain) ===")

    feature_names = list(X_ref.columns)
    importance_values = model.feature_importances_

    importance_dict = dict(zip(feature_names, importance_values.tolist()))
    importance_sorted = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # Show top features
    print(f"\nTop {min(20, top_n)} features by importance:")
    for i, (feat, imp) in enumerate(list(importance_sorted.items())[:20]):
        print(f"  {i+1:2d}. {feat:<25} {imp:>10.1f}")

    # Plot top features
    top_features = list(importance_sorted.keys())[:top_n]
    top_values = [importance_sorted[f] for f in top_features]

    plt.figure(figsize=(10, 12))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.xlabel('Feature Importance (Gain)')
    plt.title(f'LightGBM Feature Importance - Top {top_n} Features')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nImportance plot saved: {save_dir}/feature_importance.png")

    return importance_sorted


def define_feature_groups(importance_sorted, n_high=10, n_low=10):
    """
    IEEE-CIS has many more features than UCI Adult.
    Use top 10 and bottom 10 for injection experiments.
    """
    features_ranked = list(importance_sorted.keys())

    # Filter out near-zero importance features from low group
    # (they add no signal)
    non_zero_features = [f for f in features_ranked
                         if importance_sorted[f] > 0]

    high_importance = non_zero_features[:n_high]
    low_importance = non_zero_features[-n_low:]

    print(f"\nHigh importance features (top {n_high}) - MALIGNANT drift group:")
    for feat in high_importance:
        print(f"  {feat}: {importance_sorted[feat]:.1f}")

    print(f"\nLow importance features (bottom {n_low}) - BENIGN drift group:")
    for feat in low_importance:
        print(f"  {feat}: {importance_sorted[feat]:.1f}")

    return high_importance, low_importance


def save_feature_importance(importance_sorted, high_importance,
                            low_importance, metrics, save_dir):
    """Save feature importance and model metadata."""

    output = {
        "dataset": "IEEE-CIS Fraud Detection",
        "model": "LightGBM",
        "baseline_metrics": metrics,
        "importance_method": "LightGBM gain-based importance",
        "importance_ranked": {
            k: round(v, 2) for k, v in importance_sorted.items()
        },
        "feature_groups": {
            "high_importance": {
                "features": high_importance,
                "description": "Top LightGBM gain features. Drift here causes large AUC/accuracy drop (malignant).",
                "n_features": len(high_importance)
            },
            "low_importance": {
                "features": low_importance,
                "description": "Bottom non-zero gain features. Drift here causes small AUC/accuracy drop (benign).",
                "n_features": len(low_importance)
            }
        },
        "notes": [
            "Primary evaluation metric is AUC due to class imbalance (3.5% fraud rate)",
            "class_weight=balanced used during training to handle imbalance",
            "High importance features = malignant drift group",
            "Low importance features = benign drift group",
            "This distinction demonstrates Novelty 1 (harm-aware prediction)"
        ]
    }

    path = f"{save_dir}/feature_importance_ieeecis.json"
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nFeature importance saved: {path}")
    return output


def run_base_model_pipeline(data_dir, save_dir=None):
    """
    Full base model training pipeline for IEEE-CIS.

    Args:
        data_dir: directory containing data/reference.csv and data/test.csv
        save_dir: directory to save model (defaults to data_dir)
    """
    if save_dir is None:
        save_dir = data_dir

    os.makedirs(f"{save_dir}/models", exist_ok=True)

    print("=" * 60)
    print("IEEE-CIS BASE MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/5] Loading preprocessed data...")
    X_ref, y_ref, X_test, y_test = load_preprocessed_data(data_dir)

    # Step 2: Train
    print("\n[2/5] Training LightGBM model...")
    model = train_lightgbm(X_ref, y_ref)

    # Step 3: Evaluate
    print("\n[3/5] Evaluating model...")
    metrics = evaluate_model(model, X_ref, y_ref, X_test, y_test)

    # Step 4: Feature importance
    print("\n[4/5] Computing feature importance...")
    importance_sorted = compute_feature_importance(model, X_ref, save_dir)

    # Step 5: Feature groups and save
    print("\n[5/5] Defining feature groups and saving...")
    high_importance, low_importance = define_feature_groups(
        importance_sorted, n_high=10, n_low=10
    )
    save_feature_importance(
        importance_sorted, high_importance,
        low_importance, metrics, save_dir
    )

    # Save model
    model_path = f"{save_dir}/models/base_model_ieeecis.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    print("\n" + "=" * 60)
    print("BASE MODEL TRAINING COMPLETE")
    print(f"Baseline accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"Baseline AUC:      {metrics['test_auc']:.4f}  (primary metric)")
    print("=" * 60)

    return model, metrics, importance_sorted, high_importance, low_importance


if __name__ == "__main__":
    model, metrics, importance, high_feats, low_feats = run_base_model_pipeline(
        data_dir="shared/ieee_cis"
    )
