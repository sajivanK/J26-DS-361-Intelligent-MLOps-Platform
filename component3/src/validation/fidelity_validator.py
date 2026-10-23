"""Fidelity and diversity validator for synthetic tabular data."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from component3.src.generators.data_preparer import PreparedGeneratorData
from component3.src.validation.validation_models import (
    ValidationStageResult,
    ValidationStatus,
)


class FidelityValidator:
    """Evaluates statistical similarity and diversity of synthetic tabular data."""

    def __init__(
        self,
        max_mean_ks_statistic: float = 0.35,
        max_mean_js_divergence: float = 0.35,
        max_duplicate_ratio: float = 0.15,
    ) -> None:
        """Initialize FidelityValidator with configurable provisional thresholds.

        Parameters
        ----------
        max_mean_ks_statistic : float
            Maximum allowable mean Kolmogorov-Smirnov statistic across continuous features.
        max_mean_js_divergence : float
            Maximum allowable mean Jensen-Shannon divergence across categorical features.
        max_duplicate_ratio : float
            Maximum allowable proportion of duplicate rows in synthetic data.
        """
        for name, val in [
            ("max_mean_ks_statistic", max_mean_ks_statistic),
            ("max_mean_js_divergence", max_mean_js_divergence),
            ("max_duplicate_ratio", max_duplicate_ratio),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be a number between 0.0 and 1.0")

        self.max_mean_ks_statistic = float(max_mean_ks_statistic)
        self.max_mean_js_divergence = float(max_mean_js_divergence)
        self.max_duplicate_ratio = float(max_duplicate_ratio)

    def validate(
        self,
        synthetic_data: pd.DataFrame,
        reference_data: pd.DataFrame | PreparedGeneratorData,
        continuous_columns: list[str] | tuple[str, ...] | None = None,
        discrete_columns: list[str] | tuple[str, ...] | None = None,
    ) -> ValidationStageResult:
        """Evaluate fidelity and diversity against reference data."""
        return validate_fidelity(
            synthetic_data=synthetic_data,
            reference_data=reference_data,
            continuous_columns=continuous_columns,
            discrete_columns=discrete_columns,
            max_mean_ks_statistic=self.max_mean_ks_statistic,
            max_mean_js_divergence=self.max_mean_js_divergence,
            max_duplicate_ratio=self.max_duplicate_ratio,
        )


def validate_fidelity(
    synthetic_data: pd.DataFrame,
    reference_data: pd.DataFrame | PreparedGeneratorData,
    continuous_columns: list[str] | tuple[str, ...] | None = None,
    discrete_columns: list[str] | tuple[str, ...] | None = None,
    max_mean_ks_statistic: float = 0.35,
    max_mean_js_divergence: float = 0.35,
    max_duplicate_ratio: float = 0.15,
) -> ValidationStageResult:
    """Evaluate statistical fidelity, multivariate dependence, and diversity.

    Parameters
    ----------
    synthetic_data : pd.DataFrame
        Synthetic tabular dataset.
    reference_data : pd.DataFrame | PreparedGeneratorData
        Real reference dataset or PreparedGeneratorData object.
    continuous_columns : list[str] | tuple[str, ...] | None
        List of continuous feature names. Inferred from PreparedGeneratorData if provided.
    discrete_columns : list[str] | tuple[str, ...] | None
        List of categorical feature names. Inferred from PreparedGeneratorData if provided.
    max_mean_ks_statistic : float
        Provisional threshold for mean KS statistic.
    max_mean_js_divergence : float
        Provisional threshold for mean JS divergence.
    max_duplicate_ratio : float
        Provisional threshold for synthetic duplicate ratio.

    Returns
    -------
    ValidationStageResult
        The fidelity validation result containing metrics, reasons, and status.
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
        "continuous_metrics": {},
        "categorical_metrics": {},
        "correlation_difference": None,
        "diversity_metrics": {},
    }
    reasons: list[str] = []

    # Safeguard against empty inputs
    if synthetic_data.empty:
        return ValidationStageResult(
            stage="fidelity",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=("Synthetic data is empty",),
        )
    if ref_df.empty:
        return ValidationStageResult(
            stage="fidelity",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=("Reference data is empty",),
        )

    # Missing columns check
    missing_syn = [col for col in continuous + categorical if col not in synthetic_data.columns]
    missing_ref = [col for col in continuous + categorical if col not in ref_df.columns]
    if missing_syn or missing_ref:
        err_msg = f"Missing columns: synthetic={missing_syn}, reference={missing_ref}"
        return ValidationStageResult(
            stage="fidelity",
            status=ValidationStatus.FAIL,
            metrics=metrics,
            reasons=(err_msg,),
        )

    # 1. Continuous feature metrics
    ks_stats: list[float] = []
    wasserstein_dists: list[float] = []

    for col in continuous:
        ref_s = pd.to_numeric(ref_df[col], errors="coerce").dropna()
        syn_s = pd.to_numeric(synthetic_data[col], errors="coerce").dropna()

        if len(ref_s) == 0:
            reasons.append(f"Reference continuous feature '{col}' has no valid numeric values")
            continue
        if len(syn_s) == 0:
            reasons.append(f"Synthetic continuous feature '{col}' has no valid numeric values")
            continue

        ref_mean = float(ref_s.mean())
        syn_mean = float(syn_s.mean())
        ref_std = float(ref_s.std(ddof=1)) if len(ref_s) > 1 else 0.0
        syn_std = float(syn_s.std(ddof=1)) if len(syn_s) > 1 else 0.0

        # KS 2-sample test
        ks_res = ks_2samp(ref_s, syn_s)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)
        ks_stats.append(ks_stat)

        # Wasserstein distance (normalized by ref std to be scale-invariant if std > 0)
        w_dist = float(wasserstein_distance(ref_s, syn_s))
        norm_w_dist = float(w_dist / ref_std) if ref_std > 1e-8 else float(w_dist)
        wasserstein_dists.append(norm_w_dist)

        # Quantile differences (25%, 50%, 75%)
        q_diff = {
            "q25_diff": float(abs(syn_s.quantile(0.25) - ref_s.quantile(0.25))),
            "q50_diff": float(abs(syn_s.quantile(0.50) - ref_s.quantile(0.50))),
            "q75_diff": float(abs(syn_s.quantile(0.75) - ref_s.quantile(0.75))),
        }

        # Boundary concentration: proportion of syn values at the exact min or max of ref
        ref_min = float(ref_s.min())
        ref_max = float(ref_s.max())
        boundary_conc = float(((syn_s <= ref_min) | (syn_s >= ref_max)).mean())

        metrics["continuous_metrics"][col] = {
            "ref_mean": ref_mean,
            "syn_mean": syn_mean,
            "mean_abs_diff": float(abs(syn_mean - ref_mean)),
            "ref_std": ref_std,
            "syn_std": syn_std,
            "std_abs_diff": float(abs(syn_std - ref_std)),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "wasserstein_distance": w_dist,
            "normalized_wasserstein_distance": norm_w_dist,
            "quantile_diffs": q_diff,
            "boundary_concentration": boundary_conc,
        }

    mean_ks = float(np.mean(ks_stats)) if ks_stats else 0.0
    mean_w_dist = float(np.mean(wasserstein_dists)) if wasserstein_dists else 0.0
    metrics["continuous_metrics_summary"] = {
        "mean_ks_statistic": mean_ks,
        "mean_normalized_wasserstein": mean_w_dist,
    }

    if mean_ks > max_mean_ks_statistic:
        reasons.append(
            f"Mean KS statistic ({mean_ks:.3f}) exceeds threshold ({max_mean_ks_statistic:.3f})"
        )

    # 2. Categorical feature metrics
    js_divergences: list[float] = []
    tvd_list: list[float] = []

    for col in categorical:
        ref_vc = ref_df[col].astype(str).value_counts(normalize=True)
        syn_vc = synthetic_data[col].astype(str).value_counts(normalize=True)

        all_cats = sorted(list(set(ref_vc.index).union(set(syn_vc.index))))
        ref_probs = np.array([ref_vc.get(cat, 0.0) for cat in all_cats], dtype=float)
        syn_probs = np.array([syn_vc.get(cat, 0.0) for cat in all_cats], dtype=float)

        # Total Variation Distance: 0.5 * sum(|p - q|)
        tvd = float(0.5 * np.sum(np.abs(ref_probs - syn_probs)))
        tvd_list.append(tvd)

        # Jensen-Shannon divergence (base 2)
        js_dist = float(jensenshannon(ref_probs, syn_probs, base=2.0))
        # jensenshannon returns the distance (sqrt of divergence); square to get divergence
        js_div = float(js_dist ** 2) if not np.isnan(js_dist) else 1.0
        js_divergences.append(js_div)

        # Category coverage (fraction of reference categories present in synthetic)
        ref_cat_set = set(ref_vc.index)
        syn_cat_set = set(syn_vc.index)
        cat_coverage = float(len(syn_cat_set.intersection(ref_cat_set)) / len(ref_cat_set)) if ref_cat_set else 1.0
        unseen_cats = sorted(list(syn_cat_set - ref_cat_set))

        metrics["categorical_metrics"][col] = {
            "total_variation_distance": tvd,
            "js_divergence": js_div,
            "category_coverage": cat_coverage,
            "unseen_categories": unseen_cats,
        }

    mean_js = float(np.mean(js_divergences)) if js_divergences else 0.0
    mean_tvd = float(np.mean(tvd_list)) if tvd_list else 0.0
    metrics["categorical_metrics_summary"] = {
        "mean_js_divergence": mean_js,
        "mean_total_variation_distance": mean_tvd,
    }

    if mean_js > max_mean_js_divergence:
        reasons.append(
            f"Mean JS divergence ({mean_js:.3f}) exceeds threshold ({max_mean_js_divergence:.3f})"
        )

    # 3. Multivariate correlation difference (for continuous columns)
    if len(continuous) >= 2:
        ref_corr = ref_df[continuous].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0)
        syn_corr = synthetic_data[continuous].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0)
        corr_diff_matrix = (syn_corr - ref_corr).abs()
        triu_idx = np.triu_indices(len(continuous), k=1)
        corr_diff_arr = corr_diff_matrix.to_numpy()
        mean_corr_diff = float(corr_diff_arr[triu_idx].mean()) if len(triu_idx[0]) > 0 else 0.0
        max_corr_diff = float(corr_diff_arr[triu_idx].max()) if len(triu_idx[0]) > 0 else 0.0
        frobenius_norm = float(np.linalg.norm(syn_corr.to_numpy() - ref_corr.to_numpy(), ord="fro"))

        metrics["correlation_difference"] = {
            "mean_absolute_difference": mean_corr_diff,
            "max_absolute_difference": max_corr_diff,
            "frobenius_norm": frobenius_norm,
        }

    # 4. Diversity metrics
    eval_cols = [c for c in continuous + categorical if c in synthetic_data.columns]
    if eval_cols:
        syn_subset = synthetic_data[eval_cols]
        n_duplicates = int(syn_subset.duplicated().sum())
        dup_ratio = float(n_duplicates / len(syn_subset)) if len(syn_subset) > 0 else 0.0
        unique_ratio = float(len(syn_subset.drop_duplicates()) / len(syn_subset)) if len(syn_subset) > 0 else 0.0

        metrics["diversity_metrics"] = {
            "duplicate_count": n_duplicates,
            "duplicate_ratio": dup_ratio,
            "unique_record_ratio": unique_ratio,
        }

        if dup_ratio > max_duplicate_ratio:
            reasons.append(
                f"Synthetic duplicate ratio ({dup_ratio:.2%}) exceeds threshold ({max_duplicate_ratio:.2%})"
            )

    # 5. Composite Fidelity Score (0.0 to 1.0)
    # 1 - composite penalty based on mean KS, mean JS, and normalized mean correlation difference
    cont_penalty = mean_ks if continuous else 0.0
    cat_penalty = mean_js if categorical else 0.0
    # Mean correlation difference is in [0, 2]; normalize by 2.0 to bound in [0, 1]
    corr_penalty = (metrics["correlation_difference"]["mean_absolute_difference"] / 2.0) if metrics.get("correlation_difference") else 0.0
    
    components = []
    if continuous:
        components.append(cont_penalty)
    if categorical:
        components.append(cat_penalty)
    if metrics.get("correlation_difference"):
        components.append(corr_penalty)

    avg_penalty = float(np.mean(components)) if components else 0.0
    fidelity_score = float(max(0.0, min(1.0, 1.0 - avg_penalty)))
    metrics["fidelity_score"] = round(fidelity_score, 4)

    status = ValidationStatus.FAIL if reasons else ValidationStatus.PASS
    return ValidationStageResult(
        stage="fidelity",
        status=status,
        metrics=metrics,
        reasons=tuple(reasons),
    )
