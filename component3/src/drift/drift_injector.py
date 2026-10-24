"""Controlled Drift Injection and Severity Measurement Engine for Component 3.

Executes dataset-aware distribution shifts across 8 canonical drift categories,
quantifies empirical statistical divergence (KS, TVD, Wasserstein),
measures baseline model performance degradation, and generates C1/C2 diagnostic manifests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from component3.src.baselines.model_adapter import BaselineModelAdapter
from component3.src.contracts.safe_slice_evaluator import extract_affected_slice, evaluate_slice_rule
from component3.src.datasets.dataset_registry import DatasetMetadata, get_dataset_registry
from component3.src.drift.drift_config import DriftCategory, DriftScenarioConfig, DriftSeverityTier


@dataclass
class DriftInjectionResult:
    """Outcome bundle of a controlled drift injection experiment."""
    scenario_id: str
    dataset_id: str
    category: str
    injected_strength: float
    measured_drift_score: float
    feature_ks_statistics: dict[str, float]
    feature_tvd_statistics: dict[str, float]
    affected_slice_rows: int
    affected_slice_support: float
    pre_drift_overall_f1: float | None
    post_drift_overall_f1: float | None
    delta_overall_f1: float | None
    pre_drift_slice_f1: float | None
    post_drift_slice_f1: float | None
    delta_slice_f1: float | None
    drifted_df: pd.DataFrame
    manifest: dict[str, Any]
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    feature_wasserstein_statistics: dict[str, float] = field(default_factory=dict)
    flip_rate: float | None = None
    label_validity_status: str = "stress_test_uncertain"
    is_label_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert result summary to dictionary (excluding raw DataFrame)."""
        d = asdict(self)
        d.pop("drifted_df", None)
        return d

    def to_diagnostic_manifest(self) -> dict[str, Any]:
        """Return the C1/C2 diagnostic manifest."""
        return dict(self.manifest)


class DriftInjector:
    """Executes controlled drift injection and evaluates empirical degradation."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def inject(
        self,
        df: pd.DataFrame,
        config: DriftScenarioConfig,
        metadata: DatasetMetadata | None = None,
        model_adapter: BaselineModelAdapter | None = None,
    ) -> DriftInjectionResult:
        """Inject specified drift scenario into DataFrame and calculate degradation."""
        drifted = df.copy()
        cat = config.category
        strength = config.injected_strength
        target_features = list(config.target_features)
        target_col = metadata.target_column if metadata else "target"

        feature_ks: dict[str, float] = {}
        feature_tvd: dict[str, float] = {}
        feature_wasserstein: dict[str, float] = {}

        # Resolve target mapping for classification
        mapping = metadata.target_mapping if metadata else {}
        pos_keys = [k for k, v in mapping.items() if v == 1] or [1, "1", "Yes", "yes", ">50K"]
        neg_keys = [k for k, v in mapping.items() if v == 0] or [0, "0", "No", "no", "<=50K"]

        # ----------------------------------------------------------------------
        # 1. Execute Drift Operator Based on Category
        # ----------------------------------------------------------------------
        if cat == DriftCategory.NUMERICAL_DISTRIBUTION_SHIFT:
            for col in target_features:
                if col in drifted.columns:
                    orig_series = pd.to_numeric(drifted[col], errors="coerce").dropna()
                    frozen_stds = config.metadata.get("frozen_feature_std", {})
                    std = frozen_stds.get(col, float(orig_series.std()) if orig_series.std() > 0 else 1.0)
                    shift = strength * std * 1.5
                    noise = self.rng.normal(0, std * 0.1, len(drifted))
                    drifted[col] = pd.to_numeric(drifted[col], errors="coerce") + shift + noise

        elif cat == DriftCategory.CATEGORICAL_DISTRIBUTION_SHIFT:
            for col in target_features:
                if col in drifted.columns:
                    val_counts = drifted[col].value_counts()
                    dominant_val = config.metadata.get("frozen_dominant_val", {}).get(
                        col, val_counts.index[0] if len(val_counts) > 0 else "shifted"
                    )
                    mask = self.rng.random(len(drifted)) < (strength * 0.8)
                    drifted.loc[mask, col] = dominant_val

        elif cat == DriftCategory.MIXED_FEATURE_SHIFT:
            for col in target_features:
                if col in drifted.columns:
                    if pd.api.types.is_numeric_dtype(drifted[col]):
                        frozen_stds = config.metadata.get("frozen_feature_std", {})
                        std = frozen_stds.get(col, float(drifted[col].std()) if drifted[col].std() > 0 else 1.0)
                        drifted[col] = drifted[col] + (strength * std)
                    else:
                        dominant_val = drifted[col].mode().iloc[0] if not drifted[col].mode().empty else "shifted"
                        mask = self.rng.random(len(drifted)) < (strength * 0.6)
                        drifted.loc[mask, col] = dominant_val

        elif cat == DriftCategory.MISSINGNESS_SHIFT:
            for col in target_features:
                if col in drifted.columns:
                    mask = self.rng.random(len(drifted)) < (strength * 0.5)
                    drifted.loc[mask, col] = np.nan

        elif cat == DriftCategory.FEATURE_CORRELATION_SHIFT:
            if len(target_features) >= 2:
                col1, col2 = target_features[0], target_features[1]
                if col1 in drifted.columns and col2 in drifted.columns:
                    num1 = pd.to_numeric(drifted[col1], errors="coerce").fillna(0)
                    num2 = pd.to_numeric(drifted[col2], errors="coerce").fillna(0)
                    drifted[col2] = (1.0 - strength) * num2 + strength * num1

        elif cat == DriftCategory.CLASS_PRIOR_SHIFT:
            if target_col in drifted.columns:
                pos_mask = drifted[target_col].isin(pos_keys)
                neg_mask = drifted[target_col].isin(neg_keys)
                pos = drifted[pos_mask]
                neg = drifted[neg_mask]
                if len(pos) > 0 and len(neg) > 0:
                    target_pos_ratio = max(0.05, min(0.95, 0.5 + (strength - 0.5) * 0.8))
                    n_pos = int(len(drifted) * target_pos_ratio)
                    n_neg = len(drifted) - n_pos
                    sampled_pos = pos.sample(n=n_pos, replace=True, random_state=self.seed)
                    sampled_neg = neg.sample(n=n_neg, replace=True, random_state=self.seed)
                    drifted = pd.concat([sampled_pos, sampled_neg]).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        elif cat == DriftCategory.COVARIATE_SHIFT:
            for col in target_features:
                if col in drifted.columns:
                    frozen_stds = config.metadata.get("frozen_feature_std", {})
                    std = frozen_stds.get(col, float(pd.to_numeric(drifted[col], errors="coerce").std()) or 1.0)
                    if config.slice_rule:
                        try:
                            sub_mask = evaluate_slice_rule(drifted, config.slice_rule)
                        except Exception:
                            sub_mask = self.rng.random(len(drifted)) < 0.5
                    else:
                        sub_mask = self.rng.random(len(drifted)) < 0.5
                    drifted.loc[sub_mask, col] = pd.to_numeric(drifted.loc[sub_mask, col], errors="coerce") + (strength * std * 2.0)

        elif cat == DriftCategory.GENUINE_CONCEPT_DRIFT:
            if target_col in drifted.columns:
                # Prioritize frozen slice rule or frozen threshold over dynamic partition median
                if config.slice_rule:
                    try:
                        slice_mask = evaluate_slice_rule(drifted, config.slice_rule)
                    except Exception:
                        slice_mask = pd.Series(True, index=drifted.index)
                elif len(target_features) > 0 and "frozen_threshold" in config.metadata:
                    cond_col = target_features[0]
                    thresh = config.metadata["frozen_threshold"]
                    slice_mask = (pd.to_numeric(drifted[cond_col], errors="coerce") > thresh).fillna(False)
                elif len(target_features) > 0:
                    cond_col = target_features[0]
                    median_val = pd.to_numeric(drifted[cond_col], errors="coerce").median()
                    slice_mask = (pd.to_numeric(drifted[cond_col], errors="coerce") > median_val).fillna(False)
                else:
                    slice_mask = pd.Series(True, index=drifted.index)

                # Invert labels for subpopulation preserving data types
                if mapping:
                    if pd.api.types.is_numeric_dtype(drifted[target_col]):
                        inv_map = {1: 1, 0: 0}
                    else:
                        str_pos = [k for k, v in mapping.items() if v == 1 and isinstance(k, str)]
                        str_neg = [k for k, v in mapping.items() if v == 0 and isinstance(k, str)]
                        pos_repr = str_pos[0] if str_pos else (pos_keys[0] if isinstance(pos_keys[0], (str, int, np.integer)) else 1)
                        neg_repr = str_neg[0] if str_neg else (neg_keys[0] if isinstance(neg_keys[0], (str, int, np.integer)) else 0)
                        inv_map = {1: pos_repr, 0: neg_repr}
                    mapped_target = drifted.loc[slice_mask, target_col].map(mapping).fillna(0).astype(int)
                    drifted.loc[slice_mask, target_col] = (1 - mapped_target).map(inv_map)
                else:
                    drifted.loc[slice_mask, target_col] = 1 - pd.to_numeric(drifted.loc[slice_mask, target_col], errors="coerce").fillna(0).astype(int)

        # ----------------------------------------------------------------------
        # 2. Measure Empirical Statistical Drift
        # ----------------------------------------------------------------------
        drift_scores: list[float] = []

        for col in target_features:
            if col in df.columns and col in drifted.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(drifted[col]):
                    orig_clean = pd.to_numeric(df[col], errors="coerce").dropna()
                    drift_clean = pd.to_numeric(drifted[col], errors="coerce").dropna()
                    if len(orig_clean) > 0 and len(drift_clean) > 0:
                        ks_stat, _ = ks_2samp(orig_clean, drift_clean)
                        w_dist = wasserstein_distance(orig_clean, drift_clean)
                        feature_ks[col] = round(float(ks_stat), 4)
                        feature_wasserstein[col] = round(float(w_dist), 4)
                        drift_scores.append(float(ks_stat))
                else:
                    # Categorical TVD
                    p1 = df[col].astype(str).value_counts(normalize=True)
                    p2 = drifted[col].astype(str).value_counts(normalize=True)
                    all_cats = set(p1.index) | set(p2.index)
                    tvd = 0.5 * sum(abs(p1.get(c, 0.0) - p2.get(c, 0.0)) for c in all_cats)
                    feature_tvd[col] = round(float(tvd), 4)
                    drift_scores.append(float(tvd))

        mean_drift_score = round(float(np.mean(drift_scores)), 4) if drift_scores else 0.0

        # ----------------------------------------------------------------------
        # 3. Slice Support Measurement
        # ----------------------------------------------------------------------
        slice_mask_arr = np.zeros(len(drifted), dtype=bool)
        if config.slice_rule:
            try:
                slice_mask_arr = evaluate_slice_rule(drifted, config.slice_rule).to_numpy()
            except Exception:
                slice_mask_arr = np.zeros(len(drifted), dtype=bool)

        slice_count = int(np.sum(slice_mask_arr))
        slice_support = round(slice_count / max(len(drifted), 1), 4)

        # ----------------------------------------------------------------------
        # 4. Measure Model Predictions, Flip Rate & Degradation
        # ----------------------------------------------------------------------
        pre_f1: float | None = None
        post_f1: float | None = None
        delta_f1: float | None = None
        pre_slice_f1: float | None = None
        post_slice_f1: float | None = None
        delta_slice_f1: float | None = None
        flip_rate: float | None = None

        # Determine label validity status
        label_validity_str = getattr(config, "label_validity", None)
        if hasattr(label_validity_str, "value"):
            validity_status = label_validity_str.value
        elif isinstance(label_validity_str, str):
            validity_status = label_validity_str
        elif cat == DriftCategory.GENUINE_CONCEPT_DRIFT or config.is_concept_drift:
            validity_status = "synthetic_concept_truth"
        elif cat == DriftCategory.CLASS_PRIOR_SHIFT:
            validity_status = "verified_resampled_ground_truth"
        elif cat == DriftCategory.MISSINGNESS_SHIFT:
            validity_status = "verified_observational_missingness"
        elif cat == DriftCategory.COVARIATE_SHIFT:
            validity_status = "verified_observational_ground_truth"
        else:
            validity_status = "stress_test_uncertain"

        is_valid_label = validity_status != "stress_test_uncertain"

        if model_adapter is not None:
            # Prediction flip rate on aligned rows
            try:
                if len(df) == len(drifted):
                    orig_preds = model_adapter.predict(df)
                    drift_preds = model_adapter.predict(drifted)
                    flip_rate = round(float(np.mean(orig_preds != drift_preds)), 4)
            except Exception:
                pass

            if target_col in df.columns and target_col in drifted.columns:
                try:
                    ref_m = model_adapter.evaluate(df, df[target_col])
                    pre_f1 = ref_m.f1

                    drift_m = model_adapter.evaluate(drifted, drifted[target_col])
                    post_f1 = drift_m.f1
                    delta_f1 = round(post_f1 - pre_f1, 4)

                    # Slice-level metrics if slice has valid support
                    if 0 < slice_count < len(drifted):
                        ref_slice_m, _ = model_adapter.evaluate_slice(df, df[target_col], slice_mask_arr[:len(df)])
                        drift_slice_m, _ = model_adapter.evaluate_slice(drifted, drifted[target_col], slice_mask_arr)
                        pre_slice_f1 = ref_slice_m.f1
                        post_slice_f1 = drift_slice_m.f1
                        delta_slice_f1 = round(post_slice_f1 - pre_slice_f1, 4)
                except Exception:
                    pass

        # ----------------------------------------------------------------------
        # 5. Build Diagnostic Manifest (C1/C2 Compatibility)
        # ----------------------------------------------------------------------
        manifest = {
            "scenario_id": config.scenario_id,
            "dataset_id": config.dataset_id,
            "category": cat.value,
            "injected_strength": strength,
            "measured_drift_score": mean_drift_score,
            "target_features": target_features,
            "feature_ks": feature_ks,
            "feature_tvd": feature_tvd,
            "feature_wasserstein": feature_wasserstein,
            "slice_rule": config.slice_rule,
            "slice_rows": slice_count,
            "slice_support": slice_support,
            "flip_rate": flip_rate,
            "pre_drift_overall_f1": pre_f1,
            "post_drift_overall_f1": post_f1,
            "delta_overall_f1": delta_f1,
            "pre_drift_slice_f1": pre_slice_f1,
            "post_drift_slice_f1": post_slice_f1,
            "delta_slice_f1": delta_slice_f1,
            "is_concept_drift": config.is_concept_drift or (cat == DriftCategory.GENUINE_CONCEPT_DRIFT),
            "label_provenance": "ground_truth_retained" if (config.is_concept_drift or cat == DriftCategory.GENUINE_CONCEPT_DRIFT) else "simulation_injected",
            "label_validity_status": validity_status,
            "is_label_valid": is_valid_label,
        }

        return DriftInjectionResult(
            scenario_id=config.scenario_id,
            dataset_id=config.dataset_id,
            category=cat.value,
            injected_strength=strength,
            measured_drift_score=mean_drift_score,
            feature_ks_statistics=feature_ks,
            feature_tvd_statistics=feature_tvd,
            affected_slice_rows=slice_count,
            affected_slice_support=slice_support,
            pre_drift_overall_f1=pre_f1,
            post_drift_overall_f1=post_f1,
            delta_overall_f1=delta_f1,
            pre_drift_slice_f1=pre_slice_f1,
            post_drift_slice_f1=post_slice_f1,
            delta_slice_f1=delta_slice_f1,
            drifted_df=drifted,
            manifest=manifest,
            feature_wasserstein_statistics=feature_wasserstein,
            flip_rate=flip_rate,
            label_validity_status=validity_status,
            is_label_valid=is_valid_label,
        )
