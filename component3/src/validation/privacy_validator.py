"""Privacy and memorization risk validator for synthetic tabular data."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from component3.src.generators.data_preparer import PreparedGeneratorData
from component3.src.validation.validation_models import (
    ValidationStageResult,
    ValidationStatus,
)


def _nearest_distances(samples, reference):
    """Exact DCR with bounded temporary distance matrices (at most ~16 MB)."""
    if len(samples) == 0 or len(reference) == 0:
        raise ValueError("DCR requires non-empty synthetic and reference data.")
    distances = np.full(len(samples), np.inf)
    for start in range(0, len(samples), 256):
        block = samples[start:start + 256]
        best = np.full(len(block), np.inf)
        for offset in range(0, len(reference), 8192):
            best = np.minimum(best, cdist(block, reference[offset:offset + 8192], metric="euclidean").min(axis=1))
        distances[start:start + len(block)] = best
    if not np.isfinite(distances).all():
        raise ValueError("DCR produced non-finite distances.")
    return distances


class PrivacyValidator:
    """Evaluates memorization, exact copies, and Distance to Closest Record (DCR)."""

    def __init__(
        self,
        max_exact_match_ratio: float = 0.0,
        min_dcr_p05: float = 0.02,
        epsilon_near_duplicate: float = 0.01,
        max_near_duplicate_ratio: float = 0.0,
    ) -> None:
        """Initialize PrivacyValidator with configurable provisional thresholds.

        Parameters
        ----------
        max_exact_match_ratio : float
            Maximum allowable proportion of synthetic records that exactly match reference records.
        min_dcr_p05 : float
            Minimum allowable 5th percentile of Distance to Closest Record (DCR).
        epsilon_near_duplicate : float
            Distance threshold below which a synthetic record is considered a near-duplicate.
        max_near_duplicate_ratio : float
            Maximum allowable proportion of synthetic records that are near-duplicates.
        """
        for name, val in [
            ("max_exact_match_ratio", max_exact_match_ratio),
            ("min_dcr_p05", min_dcr_p05),
            ("epsilon_near_duplicate", epsilon_near_duplicate),
            ("max_near_duplicate_ratio", max_near_duplicate_ratio),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be a number between 0.0 and 1.0")

        self.max_exact_match_ratio = float(max_exact_match_ratio)
        self.min_dcr_p05 = float(min_dcr_p05)
        self.epsilon_near_duplicate = float(epsilon_near_duplicate)
        self.max_near_duplicate_ratio = float(max_near_duplicate_ratio)

    def validate(
        self,
        synthetic_data: pd.DataFrame,
        reference_data: pd.DataFrame | PreparedGeneratorData,
        holdout_data: pd.DataFrame | None = None,
        continuous_columns: list[str] | tuple[str, ...] | None = None,
        discrete_columns: list[str] | tuple[str, ...] | None = None,
        include_record_distances: bool = False,
    ) -> ValidationStageResult:
        """Evaluate privacy risk against reference and optional holdout data."""
        return validate_privacy(
            synthetic_data=synthetic_data,
            reference_data=reference_data,
            holdout_data=holdout_data,
            continuous_columns=continuous_columns,
            discrete_columns=discrete_columns,
            max_exact_match_ratio=self.max_exact_match_ratio,
            min_dcr_p05=self.min_dcr_p05,
            epsilon_near_duplicate=self.epsilon_near_duplicate,
            max_near_duplicate_ratio=self.max_near_duplicate_ratio,
            include_record_distances=include_record_distances,
        )


def validate_privacy(
    synthetic_data: pd.DataFrame,
    reference_data: pd.DataFrame | PreparedGeneratorData,
    holdout_data: pd.DataFrame | None = None,
    continuous_columns: list[str] | tuple[str, ...] | None = None,
    discrete_columns: list[str] | tuple[str, ...] | None = None,
    max_exact_match_ratio: float = 0.0,
    min_dcr_p05: float = 0.02,
    epsilon_near_duplicate: float = 0.01,
    max_near_duplicate_ratio: float = 0.0,
    include_record_distances: bool = False,
) -> ValidationStageResult:
    """Evaluate Distance to Closest Record (DCR), exact copies, and memorization risk.

    Parameters
    ----------
    synthetic_data : pd.DataFrame
        Synthetic dataset to evaluate.
    reference_data : pd.DataFrame | PreparedGeneratorData
        Real reference training data.
    holdout_data : pd.DataFrame | None, optional
        Independent held-out test data for baseline distance comparison.
    continuous_columns : list[str] | tuple[str, ...] | None
        List of continuous feature names.
    discrete_columns : list[str] | tuple[str, ...] | None
        List of categorical feature names.
    max_exact_match_ratio : float
        Maximum allowable exact copy ratio (default: 0.0).
    min_dcr_p05 : float
        Minimum allowable 5th percentile of DCR (default: 0.02).
    epsilon_near_duplicate : float
        Distance threshold defining near duplicates (default: 0.01).
    max_near_duplicate_ratio : float
        Maximum allowable near duplicate ratio (default: 0.0).

    Returns
    -------
    ValidationStageResult
        The privacy validation outcome containing aggregate indicators and status.
    """
    if not isinstance(synthetic_data, pd.DataFrame):
        raise TypeError("synthetic_data must be a pandas DataFrame")

    if isinstance(reference_data, PreparedGeneratorData):
        ref_df = reference_data.training_data
        if continuous_columns is None:
            continuous_columns = reference_data.continuous_columns
        if discrete_columns is None:
            discrete_columns = reference_data.discrete_columns
    elif isinstance(reference_data, pd.DataFrame):
        ref_df = reference_data
    else:
        raise TypeError("reference_data must be a pandas DataFrame or PreparedGeneratorData")

    continuous = list(continuous_columns or [])
    categorical = list(discrete_columns or [])

    metrics: dict[str, Any] = {
        "synthetic_rows": int(len(synthetic_data)),
        "reference_rows": int(len(ref_df)),
        "exact_matches": {},
        "dcr_metrics": {},
        "holdout_comparison": None,
    }
    reasons: list[str] = []

    if synthetic_data.empty:
        return ValidationStageResult(
            stage="privacy",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=("Synthetic data is empty",),
        )
    if ref_df.empty:
        return ValidationStageResult(
            stage="privacy",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=("Reference data is empty",),
        )

    # 1. Exact Match Detection
    eval_cols = [c for c in continuous + categorical if c in synthetic_data.columns and c in ref_df.columns]
    if not eval_cols:
        return ValidationStageResult(
            stage="privacy",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=("No common columns between synthetic and reference data",),
        )

    syn_subset = synthetic_data[eval_cols].copy()
    ref_subset = ref_df[eval_cols].copy()

    # Exact matches via inner merge on all evaluated columns
    exact_matches = pd.merge(syn_subset, ref_subset.drop_duplicates(), on=eval_cols, how="inner")
    exact_match_count = int(len(exact_matches))
    exact_match_ratio = float(exact_match_count / len(syn_subset)) if len(syn_subset) > 0 else 0.0

    metrics["exact_matches"] = {
        "exact_match_count": exact_match_count,
        "exact_match_ratio": exact_match_ratio,
        "max_allowable_ratio": max_exact_match_ratio,
    }

    if exact_match_ratio > max_exact_match_ratio:
        reasons.append(
            f"Exact match ratio ({exact_match_ratio:.2%}) exceeds threshold ({max_exact_match_ratio:.2%})"
        )

    # 2. Normalized Mixed-Type Distance to Closest Record (DCR)
    # Numerical scaling: [0, 1] MinMax scaling relative to reference bounds
    # Categorical scaling: one-hot encoding normalized by total features
    ref_encoded_parts = []
    syn_encoded_parts = []

    for col in continuous:
        ref_vals = pd.to_numeric(ref_subset[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        syn_vals = pd.to_numeric(syn_subset[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        c_min = float(ref_vals.min())
        c_max = float(ref_vals.max())
        denom = (c_max - c_min) if (c_max - c_min) > 1e-8 else 1.0

        ref_norm = np.clip((ref_vals - c_min) / denom, 0.0, 1.0).reshape(-1, 1)
        syn_norm = np.clip((syn_vals - c_min) / denom, 0.0, 1.0).reshape(-1, 1)

        ref_encoded_parts.append(ref_norm)
        syn_encoded_parts.append(syn_norm)

    for col in categorical:
        # Categorical one-hot encoding using union of categories
        all_cats = sorted(list(set(ref_subset[col].astype(str)).union(set(syn_subset[col].astype(str)))))
        ref_cat_series = pd.Categorical(ref_subset[col].astype(str), categories=all_cats)
        syn_cat_series = pd.Categorical(syn_subset[col].astype(str), categories=all_cats)

        ref_dummies = pd.get_dummies(ref_cat_series, dtype=float).to_numpy() / np.sqrt(2.0)
        syn_dummies = pd.get_dummies(syn_cat_series, dtype=float).to_numpy() / np.sqrt(2.0)

        ref_encoded_parts.append(ref_dummies)
        syn_encoded_parts.append(syn_dummies)

    if ref_encoded_parts and syn_encoded_parts:
        X_ref = np.hstack(ref_encoded_parts)
        X_syn = np.hstack(syn_encoded_parts)

        # Scale by 1/sqrt(n_features) so distance is normalized in [0, 1]
        n_features = len(continuous) + len(categorical)
        scale_factor = 1.0 / np.sqrt(max(1, n_features))
        X_ref_scaled = X_ref * scale_factor
        X_syn_scaled = X_syn * scale_factor

        # Distance to closest record: min distance from each synthetic row to reference rows
        # Vectorized cdist (computationally lightweight for research batch sizes)
        min_distances = _nearest_distances(X_syn_scaled, X_ref_scaled)
        if include_record_distances:
            metrics["record_distances"] = min_distances.tolist()

        dcr_p05 = float(np.percentile(min_distances, 5))
        dcr_p50 = float(np.percentile(min_distances, 50))
        dcr_mean = float(np.mean(min_distances))
        dcr_min = float(np.min(min_distances))

        near_duplicates = int(np.sum(min_distances < epsilon_near_duplicate))
        near_duplicate_ratio = float(near_duplicates / len(min_distances))

        metrics["dcr_metrics"] = {
            "dcr_min": round(dcr_min, 4),
            "dcr_p05": round(dcr_p05, 4),
            "dcr_p50": round(dcr_p50, 4),
            "dcr_mean": round(dcr_mean, 4),
            "near_duplicate_count": near_duplicates,
            "near_duplicate_ratio": round(near_duplicate_ratio, 4),
            "min_dcr_p05_threshold": min_dcr_p05,
            "max_near_duplicate_ratio": max_near_duplicate_ratio,
        }

        if dcr_p05 < min_dcr_p05:
            reasons.append(
                f"5th percentile DCR ({dcr_p05:.4f}) is below threshold ({min_dcr_p05:.4f}), indicating possible memorization"
            )

        if near_duplicate_ratio > max_near_duplicate_ratio:
            reasons.append(
                f"Near-duplicate ratio ({near_duplicate_ratio:.2%}) exceeds threshold ({max_near_duplicate_ratio:.2%}), indicating memorization of individual records"
            )

    # 3. Optional Holdout Distance Comparison (Train vs Holdout DCR)
    if holdout_data is not None and isinstance(holdout_data, pd.DataFrame) and not holdout_data.empty:
        try:
            holdout_subset = holdout_data[eval_cols].copy()
            holdout_encoded_parts = []
            for col in continuous:
                h_vals = pd.to_numeric(holdout_subset[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                c_min = float(ref_subset[col].min())
                c_max = float(ref_subset[col].max())
                denom = (c_max - c_min) if (c_max - c_min) > 1e-8 else 1.0
                h_norm = np.clip((h_vals - c_min) / denom, 0.0, 1.0).reshape(-1, 1)
                holdout_encoded_parts.append(h_norm)
            for col in categorical:
                all_cats = sorted(list(set(ref_subset[col].astype(str)).union(set(syn_subset[col].astype(str)))))
                h_cat_series = pd.Categorical(holdout_subset[col].astype(str), categories=all_cats)
                h_dummies = pd.get_dummies(h_cat_series, dtype=float).to_numpy() / np.sqrt(2.0)
                holdout_encoded_parts.append(h_dummies)

            X_holdout = np.hstack(holdout_encoded_parts) * scale_factor
            min_dist_holdout = _nearest_distances(X_syn_scaled, X_holdout)

            # Nearest Neighbor Distance Ratio: mean(DCR_train) / mean(DCR_holdout)
            # A ratio significantly < 1 suggests synthetic data is memorizing training data relative to holdout
            dcr_holdout_mean = float(np.mean(min_dist_holdout))
            nndr = float(dcr_mean / dcr_holdout_mean) if dcr_holdout_mean > 1e-8 else 1.0

            metrics["holdout_comparison"] = {
                "dcr_holdout_mean": round(dcr_holdout_mean, 4),
                "nndr": round(nndr, 4),
            }
        except Exception:
            metrics["holdout_comparison"] = "EVALUATION_ERROR"
            reasons.append("Requested holdout privacy comparison could not be evaluated.")

    status = ValidationStatus.FAIL if reasons else ValidationStatus.PASS
    return ValidationStageResult(
        stage="privacy",
        status=status,
        metrics=metrics,
        reasons=tuple(reasons),
    )
