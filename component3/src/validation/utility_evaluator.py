"""Offline predictive utility evaluator for synthetic tabular data repair."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from component3.src.contracts.safe_slice_evaluator import extract_affected_slice
from component3.src.validation.validation_models import (
    ValidationStageResult,
    ValidationStatus,
)


class OfflineUtilityEvaluator:
    """Evaluates downstream classification performance under synthetic augmentation."""

    def __init__(
        self,
        min_slice_f1_improvement: float = -0.02,
        max_non_affected_degradation: float = 0.05,
        target_column: str = "target",
    ) -> None:
        """Initialize OfflineUtilityEvaluator.

        Parameters
        ----------
        min_slice_f1_improvement : float
            Minimum allowable F1 delta on affected slice (default: -0.02, allowing non-degrading repairs).
        max_non_affected_degradation : float
            Maximum allowable performance drop on the non-affected population.
        target_column : str
            Name of the classification target column.
        """
        self.min_slice_f1_improvement = float(min_slice_f1_improvement)
        self.max_non_affected_degradation = float(max_non_affected_degradation)
        self.target_column = target_column

    def evaluate(
        self,
        synthetic_data: pd.DataFrame,
        reference_train_data: pd.DataFrame,
        evaluation_test_data: pd.DataFrame,
        slice_rule: str | None = None,
        baseline_model: Any | None = None,
        label_authorized: bool = False,
    ) -> ValidationStageResult:
        """Execute comparative offline utility evaluation."""
        return evaluate_offline_utility(
            synthetic_data=synthetic_data,
            reference_train_data=reference_train_data,
            evaluation_test_data=evaluation_test_data,
            slice_rule=slice_rule,
            target_column=self.target_column,
            baseline_model=baseline_model,
            label_authorized=label_authorized,
            min_slice_f1_improvement=self.min_slice_f1_improvement,
            max_non_affected_degradation=self.max_non_affected_degradation,
        )


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute standard classification metrics safely with single-class edge cases."""
    if len(y_true) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "support": 0,
            "confusion_matrix": None,
        }

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y_true, y_pred).tolist() if len(np.unique(y_true)) > 1 else None

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "support": int(len(y_true)),
        "confusion_matrix": cm,
    }


def evaluate_offline_utility(
    synthetic_data: pd.DataFrame,
    reference_train_data: pd.DataFrame,
    evaluation_test_data: pd.DataFrame,
    slice_rule: str | None = None,
    target_column: str = "target",
    baseline_model: Any | None = None,
    label_authorized: bool = False,
    min_slice_f1_improvement: float = -0.02,
    max_non_affected_degradation: float = 0.05,
) -> ValidationStageResult:
    """Evaluate offline utility by comparing baseline vs. retrained vs. augmented models.

    Parameters
    ----------
    synthetic_data : pd.DataFrame
        Synthetic dataset.
    reference_train_data : pd.DataFrame
        Authorized reference training dataset.
    evaluation_test_data : pd.DataFrame
        Independent held-out evaluation dataset.
    slice_rule : str | None
        Target slice rule expression.
    target_column : str
        Target label column name.
    baseline_model : Any | None
        Pre-trained baseline model (optional).
    label_authorized : bool
        Governance flag authorizing supervised retraining.
    min_slice_f1_improvement : float
        Acceptance threshold for affected slice F1 change.
    max_non_affected_degradation : float
        Maximum allowable degradation on non-affected population.

    Returns
    -------
    ValidationStageResult
        The utility evaluation outcome containing metrics, comparative diffs, and status.
    """
    if not isinstance(synthetic_data, pd.DataFrame):
        raise TypeError("synthetic_data must be a pandas DataFrame")
    if not isinstance(reference_train_data, pd.DataFrame):
        raise TypeError("reference_train_data must be a pandas DataFrame")
    if not isinstance(evaluation_test_data, pd.DataFrame):
        raise TypeError("evaluation_test_data must be a pandas DataFrame")

    metrics: dict[str, Any] = {
        "label_authorized": bool(label_authorized),
        "target_column": target_column,
        "evaluation_rows": int(len(evaluation_test_data)),
        "baseline_metrics": None,
        "real_only_retrained_metrics": None,
        "augmented_retrained_metrics": None,
        "deltas": {},
        "utility_score": None,
    }
    reasons: list[str] = []

    # Safeguard 1: Governance gate on label authorization
    if not label_authorized:
        reasons.append(
            "Supervised retraining is NOT_EVALUATED: synthetic target labels are untrusted or unauthorized"
        )
        return ValidationStageResult(
            stage="utility",
            status=ValidationStatus.NOT_EVALUATED,
            metrics=metrics,
            reasons=tuple(reasons),
        )

    # Safeguard 2: Check target column presence and validity
    if target_column not in synthetic_data.columns:
        reasons.append(
            f"Target column '{target_column}' is missing from synthetic data; cannot perform supervised retraining"
        )
        return ValidationStageResult(
            stage="utility",
            status=ValidationStatus.NOT_EVALUATED,
            metrics=metrics,
            reasons=tuple(reasons),
        )
    if target_column not in reference_train_data.columns or target_column not in evaluation_test_data.columns:
        reasons.append(
            f"Target column '{target_column}' is missing from reference train or evaluation test data"
        )
        return ValidationStageResult(
            stage="utility",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=tuple(reasons),
        )

    # Check target column validity (missing values and class conformity)
    if synthetic_data[target_column].isna().any():
        reasons.append(f"Synthetic target column '{target_column}' contains missing values")
        return ValidationStageResult(
            stage="utility",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=tuple(reasons),
        )
    ref_targets = set(reference_train_data[target_column].dropna().unique())
    syn_targets = set(synthetic_data[target_column].dropna().unique())
    if not syn_targets.issubset(ref_targets):
        invalid_classes = sorted(list(syn_targets - ref_targets), key=str)
        reasons.append(f"Synthetic target column contains invalid classes not in reference: {invalid_classes}")
        return ValidationStageResult(
            stage="utility",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=tuple(reasons),
        )

    # Align feature column ordering with reference training data
    feature_cols = [c for c in reference_train_data.columns if c != target_column]
    if any(c not in synthetic_data.columns for c in feature_cols):
        missing = [c for c in feature_cols if c not in synthetic_data.columns]
        reasons.append(f"Synthetic data missing required reference features: {missing}")
        return ValidationStageResult(
            stage="utility",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=tuple(reasons),
        )

    # Reorder columns to guarantee exact match with reference training data
    syn_cols = feature_cols + ([target_column] if target_column in synthetic_data.columns else [])
    synthetic_data = synthetic_data[syn_cols].copy()
    reference_train_data = reference_train_data[feature_cols + [target_column]].copy()
    evaluation_test_data = evaluation_test_data[feature_cols + [target_column]].copy()

    # Convert object dtype columns to category for robust XGBoost support
    for c in feature_cols:
        if reference_train_data[c].dtype == "object":
            reference_train_data[c] = reference_train_data[c].astype("category")
            synthetic_data[c] = synthetic_data[c].astype("category")
            evaluation_test_data[c] = evaluation_test_data[c].astype("category")

    # Prepare evaluation test splits (Overall, Affected slice, Non-affected slice)
    test_df = evaluation_test_data.copy()
    X_test_all = test_df[feature_cols]
    y_test_all = test_df[target_column].to_numpy(dtype=int)

    if slice_rule:
        try:
            affected_test_df = extract_affected_slice(test_df, slice_rule)
            affected_idx = affected_test_df.index
            non_affected_test_df = test_df.drop(index=affected_idx)
        except Exception:
            affected_test_df = test_df
            non_affected_test_df = pd.DataFrame(columns=test_df.columns)
    else:
        affected_test_df = test_df
        non_affected_test_df = pd.DataFrame(columns=test_df.columns)

    X_test_aff = affected_test_df.drop(columns=[target_column]) if not affected_test_df.empty else pd.DataFrame()
    y_test_aff = affected_test_df[target_column].to_numpy(dtype=int) if not affected_test_df.empty else np.array([])

    X_test_non = non_affected_test_df.drop(columns=[target_column]) if not non_affected_test_df.empty else pd.DataFrame()
    y_test_non = non_affected_test_df[target_column].to_numpy(dtype=int) if not non_affected_test_df.empty else np.array([])

    # Helper to evaluate model across 3 evaluation populations
    def _evaluate_model(model: Any) -> dict[str, Any]:
        res: dict[str, Any] = {}
        # Overall
        preds_all = model.predict(X_test_all)
        probs_all = model.predict_proba(X_test_all)[:, 1] if hasattr(model, "predict_proba") else None
        res["overall"] = _compute_classification_metrics(y_test_all, preds_all, probs_all)

        # Affected slice
        if len(y_test_aff) > 0:
            preds_aff = model.predict(X_test_aff)
            probs_aff = model.predict_proba(X_test_aff)[:, 1] if hasattr(model, "predict_proba") else None
            res["affected_slice"] = _compute_classification_metrics(y_test_aff, preds_aff, probs_aff)
        else:
            res["affected_slice"] = _compute_classification_metrics(np.array([]), np.array([]))

        # Non-affected slice
        if len(y_test_non) > 0:
            preds_non = model.predict(X_test_non)
            probs_non = model.predict_proba(X_test_non)[:, 1] if hasattr(model, "predict_proba") else None
            res["non_affected"] = _compute_classification_metrics(y_test_non, preds_non, probs_non)
        else:
            res["non_affected"] = _compute_classification_metrics(np.array([]), np.array([]))

        return res

    # 1. Baseline model evaluation
    if baseline_model is not None:
        metrics["baseline_metrics"] = _evaluate_model(baseline_model)

    # 2. Retrain Real-Only model (Controlled control group)
    X_train_ref = reference_train_data.drop(columns=[target_column])
    y_train_ref = reference_train_data[target_column].to_numpy(dtype=int)

    real_only_model = XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        enable_categorical=True,
    )
    real_only_model.fit(X_train_ref, y_train_ref)
    metrics["real_only_retrained_metrics"] = _evaluate_model(real_only_model)

    # 3. Retrain Augmented model (Reference + Synthetic) with identical architecture
    augmented_train_df = pd.concat([reference_train_data, synthetic_data], ignore_index=True)
    X_train_aug = augmented_train_df.drop(columns=[target_column])
    y_train_aug = augmented_train_df[target_column].to_numpy(dtype=int)

    augmented_model = XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        enable_categorical=True,
    )
    augmented_model.fit(X_train_aug, y_train_aug)
    metrics["augmented_retrained_metrics"] = _evaluate_model(augmented_model)

    # 4. Compute Performance Deltas (Augmented vs Controlled Real-Only Baseline)
    real_ref = metrics["real_only_retrained_metrics"]
    aug_ref = metrics["augmented_retrained_metrics"]

    delta_overall_f1 = aug_ref["overall"]["f1"] - real_ref["overall"]["f1"]
    delta_overall_acc = aug_ref["overall"]["accuracy"] - real_ref["overall"]["accuracy"]

    delta_aff_f1 = 0.0
    if aug_ref["affected_slice"]["f1"] is not None and real_ref["affected_slice"]["f1"] is not None:
        delta_aff_f1 = aug_ref["affected_slice"]["f1"] - real_ref["affected_slice"]["f1"]

    delta_non_aff_f1 = 0.0
    if aug_ref["non_affected"]["f1"] is not None and real_ref["non_affected"]["f1"] is not None:
        delta_non_aff_f1 = aug_ref["non_affected"]["f1"] - real_ref["non_affected"]["f1"]

    metrics["deltas"] = {
        "delta_overall_f1": round(float(delta_overall_f1), 4),
        "delta_overall_accuracy": round(float(delta_overall_acc), 4),
        "delta_affected_slice_f1": round(float(delta_aff_f1), 4),
        "delta_non_affected_f1": round(float(delta_non_aff_f1), 4),
    }

    # Record deployment deltas vs pre-drift baseline model if provided
    if metrics["baseline_metrics"] is not None:
        base_ref = metrics["baseline_metrics"]
        base_slice_f1 = base_ref["affected_slice"]["f1"]
        aug_slice_f1 = aug_ref["affected_slice"]["f1"]
        base_non_f1 = base_ref["non_affected"]["f1"]
        aug_non_f1 = aug_ref["non_affected"]["f1"]

        metrics["deltas_vs_baseline"] = {
            "delta_overall_f1": round(float(aug_ref["overall"]["f1"] - base_ref["overall"]["f1"]), 4),
            "delta_overall_accuracy": round(float(aug_ref["overall"]["accuracy"] - base_ref["overall"]["accuracy"]), 4),
            "delta_affected_slice_f1": round(float(aug_slice_f1 - base_slice_f1), 4) if aug_slice_f1 is not None and base_slice_f1 is not None else 0.0,
            "delta_non_affected_f1": round(float(aug_non_f1 - base_non_f1), 4) if aug_non_f1 is not None and base_non_f1 is not None else 0.0,
        }

    # Bounded Utility Score: centered around controlled real-only accuracy with affected-slice improvement boost
    base_acc = real_ref["overall"]["accuracy"] if real_ref["overall"]["accuracy"] is not None else 0.5
    raw_utility = base_acc + (delta_aff_f1 * 0.5)
    utility_score = max(0.0, min(1.0, float(raw_utility)))
    metrics["utility_score"] = round(utility_score, 4)

    # Validation Checks
    if delta_aff_f1 < min_slice_f1_improvement:
        reasons.append(
            f"Affected slice F1 delta ({delta_aff_f1:+.4f}) is below minimum threshold ({min_slice_f1_improvement:+.4f})"
        )
    if delta_non_aff_f1 < -max_non_affected_degradation:
        reasons.append(
            f"Non-affected population degraded by {-delta_non_aff_f1:.4f}, exceeding limit ({max_non_affected_degradation:.4f})"
        )

    status = ValidationStatus.FAIL if reasons else ValidationStatus.PASS
    return ValidationStageResult(
        stage="utility",
        status=status,
        metrics=metrics,
        reasons=tuple(reasons),
    )
