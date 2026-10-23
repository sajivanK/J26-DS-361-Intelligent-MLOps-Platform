"""Isolated retraining and downstream utility evaluator for Component 3.

Evaluates candidate model retraining against authentic baselines, calculates
paired statistical confidence intervals, enforces source retention gates,
and produces candidate model artifacts without modifying registered production models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from component3.src.preprocessing.tabular_preprocessor import TabularDataPreprocessor


@dataclass(frozen=True)
class RetrainingEvaluationResult:
    """Consolidated downstream retraining evaluation metrics and decision."""
    status: str  # "EVALUATED", "NOT_EVALUABLE", "REJECTED"
    is_acceptable: bool
    rejection_reasons: tuple[str, ...]

    # Target adaptation metrics (D_dev_val)
    target_positive_f1: float | None
    target_macro_f1: float | None
    target_balanced_accuracy: float | None
    target_accuracy: float | None
    target_precision: float | None
    target_recall: float | None

    # Source retention metrics (D_source_eval)
    source_positive_f1: float | None
    source_retention_delta: float | None
    source_retention_preserved: bool

    # Comparisons vs Baseline
    delta_target_f1_vs_baseline: float | None
    bootstrap_ci_95: tuple[float, float] | None
    uncertainty_ci_width: float | None = None
    uncertainty_flag: bool = False
    delta_vs_authentic_control: float | None = None
    min_practical_utility_margin: float = 0.01

    training_rows: int = 0
    training_duration_seconds: float = 0.0
    candidate_model: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rejection_reasons"] = list(self.rejection_reasons)
        if self.bootstrap_ci_95:
            d["bootstrap_ci_95"] = list(self.bootstrap_ci_95)
        d.pop("candidate_model", None)
        return d



def paired_bootstrap_ci_f1(
    y_true: np.ndarray,
    y_pred_cand: np.ndarray,
    y_pred_ref: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Calculate paired bootstrap confidence intervals for positive F1 difference."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        f1_c = f1_score(y_true[idx], y_pred_cand[idx], pos_label=1, zero_division=0)
        f1_r = f1_score(y_true[idx], y_pred_ref[idx], pos_label=1, zero_division=0)
        diffs[b] = f1_c - f1_r

    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)


class RetrainingEvaluator:
    """Isolated evaluator for candidate model retraining and utility measurement."""

    def __init__(
        self,
        source_retention_tolerance: float = 0.05,
        target_f1_improvement_threshold: float = 0.01,
        min_practical_utility_margin: float = 0.01,
        max_uncertainty_ci_width: float = 0.50,
        seed: int = 42,
        require_source_retention: bool = True,
    ) -> None:
        self.source_retention_tolerance = source_retention_tolerance
        self.target_f1_improvement_threshold = target_f1_improvement_threshold
        self.min_practical_utility_margin = min_practical_utility_margin
        self.max_uncertainty_ci_width = max_uncertainty_ci_width
        self.seed = seed
        self.require_source_retention = require_source_retention

    def evaluate_candidate(
        self,
        base_model: Any,
        preprocessor: TabularDataPreprocessor,
        df_source_train: pd.DataFrame,
        df_adaptation: pd.DataFrame | None = None,
        synthetic_df: pd.DataFrame | None = None,
        df_dev_val: pd.DataFrame | None = None,
        df_source_eval: pd.DataFrame | None = None,
        sample_weight_policy: dict[str, float] | None = None,
        target_column: str = "Churn",
        reference_authentic_f1: float | None = None,
        min_utility_margin: float | None = None,
    ) -> RetrainingEvaluationResult:

        """Fit candidate model on authorized data and evaluate against baseline."""
        reasons: list[str] = []

        if df_dev_val is None or target_column not in df_dev_val.columns:
            return RetrainingEvaluationResult(
                status="NOT_EVALUABLE",
                is_acceptable=False,
                rejection_reasons=("Authorized development validation set or labels unavailable.",),
                target_positive_f1=None,
                target_macro_f1=None,
                target_balanced_accuracy=None,
                target_accuracy=None,
                target_precision=None,
                target_recall=None,
                source_positive_f1=None,
                source_retention_delta=None,
                source_retention_preserved=False,
                delta_target_f1_vs_baseline=None,
                bootstrap_ci_95=None,
                training_rows=0,
                training_duration_seconds=0.0,
                candidate_model=None,
            )

        # Preprocess development and source evaluation sets
        bundle_dev = preprocessor.transform(df_dev_val)
        X_dev, y_dev = bundle_dev.X, bundle_dev.y

        X_seval, y_seval = None, None
        base_source_f1 = None
        if df_source_eval is not None and target_column in df_source_eval.columns:
            bundle_seval = preprocessor.transform(df_source_eval)
            X_seval, y_seval = bundle_seval.X, bundle_seval.y
            y_base_seval = base_model.predict(X_seval)
            base_source_f1 = float(f1_score(y_seval, y_base_seval, pos_label=1, zero_division=0))

        # Baseline performance on dev_val
        y_base_dev = base_model.predict(X_dev)
        base_dev_f1 = float(f1_score(y_dev, y_base_dev, pos_label=1, zero_division=0))

        # Transform and combine training data
        weights_map = sample_weight_policy or {"source": 1.0, "target": 1.0, "synth": 1.0}
        w_src = weights_map.get("source", 1.0)
        w_tgt = weights_map.get("target", 1.0)
        w_syn = weights_map.get("synth", 1.0)

        bundle_src = preprocessor.transform(df_source_train)
        X_train_list = [bundle_src.X]
        y_train_list = [bundle_src.y]
        w_train_list = [np.full(len(bundle_src.y), w_src, dtype=np.float64)]

        if df_adaptation is not None and len(df_adaptation) > 0:
            bundle_adapt = preprocessor.transform(df_adaptation)
            X_train_list.append(bundle_adapt.X)
            y_train_list.append(bundle_adapt.y)
            w_train_list.append(np.full(len(bundle_adapt.y), w_tgt, dtype=np.float64))

        if synthetic_df is not None and len(synthetic_df) > 0:
            bundle_synth = preprocessor.transform(synthetic_df)
            X_train_list.append(bundle_synth.X)
            y_train_list.append(bundle_synth.y)
            w_train_list.append(np.full(len(bundle_synth.y), w_syn, dtype=np.float64))

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        w_train = np.concatenate(w_train_list)

        # Clone and train candidate model
        t0_train = time.perf_counter()
        try:
            cand_model = clone(base_model)
        except Exception:
            # Fallback if estimator cloning fails
            cand_model = base_model.__class__(**base_model.get_params())

        # Fit with sample weights if supported
        try:
            cand_model.fit(X_train, y_train, sample_weight=w_train)
        except TypeError:
            cand_model.fit(X_train, y_train)

        dur_train = round(time.perf_counter() - t0_train, 3)

        # Evaluate candidate on dev_val
        y_cand_dev = cand_model.predict(X_dev)
        cand_dev_f1 = float(f1_score(y_dev, y_cand_dev, pos_label=1, zero_division=0))
        cand_macro_f1 = float(f1_score(y_dev, y_cand_dev, average="macro", zero_division=0))
        cand_bal_acc = float(balanced_accuracy_score(y_dev, y_cand_dev))
        cand_acc = float(accuracy_score(y_dev, y_cand_dev))
        cand_prec = float(precision_score(y_dev, y_cand_dev, pos_label=1, zero_division=0))
        cand_rec = float(recall_score(y_dev, y_cand_dev, pos_label=1, zero_division=0))

        delta_target_f1 = round(cand_dev_f1 - base_dev_f1, 4)

        # Compute paired bootstrap CI vs baseline
        boot_ci = paired_bootstrap_ci_f1(y_dev, y_cand_dev, y_base_dev, n_bootstraps=1000, seed=self.seed)

        # Evaluate source retention
        cand_src_f1 = None
        src_ret_delta = None
        src_preserved = True
        if X_seval is not None and y_seval is not None and base_source_f1 is not None:
            y_cand_seval = cand_model.predict(X_seval)
            cand_src_f1 = float(f1_score(y_seval, y_cand_seval, pos_label=1, zero_division=0))
            src_ret_delta = round(cand_src_f1 - base_source_f1, 4)
            if src_ret_delta < -self.source_retention_tolerance:
                src_preserved = False
                reasons.append(
                    f"Source retention degraded by {src_ret_delta:.4f} (tolerance: -{self.source_retention_tolerance:.4f})."
                )
        elif self.require_source_retention:
            src_preserved = False
            reasons.append(
                "Mandatory source-retention validation failed: source evaluation partition (D_source_eval) is missing, empty, or un-evaluable."
            )

        # Check performance gain vs practical utility margin
        is_acceptable = True
        if not src_preserved:
            is_acceptable = False

        margin = min_utility_margin if min_utility_margin is not None else self.min_practical_utility_margin
        if delta_target_f1 < margin:
            is_acceptable = False
            reasons.append(
                f"Target adaptation performance gain ({delta_target_f1:+.4f}) is below minimum practical utility margin ({margin:+.4f}) vs baseline."
            )

        # Uncertainty checks
        ci_width = round(boot_ci[1] - boot_ci[0], 4)
        uncertainty_flag = False
        if boot_ci[1] <= 0.0:
            is_acceptable = False
            uncertainty_flag = True
            reasons.append("Bootstrap 95% CI upper bound is non-positive; candidate improvement is indistinguishable from zero or negative.")
        elif ci_width > self.max_uncertainty_ci_width:
            uncertainty_flag = True

        # Matched-budget authentic control comparison
        delta_vs_auth = None
        if reference_authentic_f1 is not None:
            delta_vs_auth = round(cand_dev_f1 - reference_authentic_f1, 4)
            if delta_vs_auth < -0.05:
                is_acceptable = False
                reasons.append(
                    f"Candidate performance ({cand_dev_f1:.4f}) significantly underperforms matched authentic control ({reference_authentic_f1:.4f}) by {delta_vs_auth:+.4f}."
                )

        return RetrainingEvaluationResult(
            status="EVALUATED",
            is_acceptable=is_acceptable,
            rejection_reasons=tuple(reasons),
            target_positive_f1=round(cand_dev_f1, 4),
            target_macro_f1=round(cand_macro_f1, 4),
            target_balanced_accuracy=round(cand_bal_acc, 4),
            target_accuracy=round(cand_acc, 4),
            target_precision=round(cand_prec, 4),
            target_recall=round(cand_rec, 4),
            source_positive_f1=round(cand_src_f1, 4) if cand_src_f1 is not None else None,
            source_retention_delta=src_ret_delta,
            source_retention_preserved=src_preserved,
            delta_target_f1_vs_baseline=delta_target_f1,
            bootstrap_ci_95=boot_ci,
            uncertainty_ci_width=ci_width,
            uncertainty_flag=uncertainty_flag,
            delta_vs_authentic_control=delta_vs_auth,
            min_practical_utility_margin=margin,
            training_rows=len(X_train),
            training_duration_seconds=dur_train,
            candidate_model=cand_model,
        )

