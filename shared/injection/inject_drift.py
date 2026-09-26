"""
Main Injection Dispatcher
J26-DS-361 | Shared Artifact | Maintained by Sajivan K (IT23172296)

PRIMARY INTERFACE for all team members.

Usage:
    from injection.inject_drift import inject_drift

    result = inject_drift(
        reference_df=ref_df,
        model=base_model,
        drift_type="interaction",
        features=["education_num", "occupation"],
        magnitude=0.3,
        feature_types={"education_num": "continuous", "occupation": "categorical"},
        seed=42
    )

    drifted_df   = result['drifted_df']
    ground_truth = result['ground_truth']

Result dictionary always contains:
    drifted_df:       pandas DataFrame (the drifted production batch)
    ground_truth:     dict with all metadata and actual accuracy drop
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from injection.drift_types import (
    inject_marginal_single,
    inject_marginal_multi,
    inject_interaction,
    inject_interaction_multi_pair,
    inject_combined,
    inject_none,
    compute_per_feature_ks
)


# ── VALID DRIFT TYPES ─────────────────────────────────────────────────────────
VALID_DRIFT_TYPES = {
    "marginal_single",
    "marginal_multi",
    "interaction",
    "combined",
    "none"
}

# ── SEVERITY THRESHOLDS ───────────────────────────────────────────────────────
# Based on predicted accuracy drop percentage
SEVERITY_THRESHOLDS = {
    "none":     (0.0,  1.0),   # 0-1% drop
    "low":      (1.0,  3.0),   # 1-3% drop
    "moderate": (3.0,  7.0),   # 3-7% drop
    "high":     (7.0,  15.0),  # 7-15% drop
    "critical": (15.0, 100.0)  # >15% drop
}


def get_severity_label(accuracy_drop_pct):
    """Map accuracy drop percentage to severity label."""
    for label, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= accuracy_drop_pct < high:
            return label
    return "critical"


def compute_accuracy_drop(model, reference_df, drifted_df, target_col='target'):
    """
    Compute actual accuracy drop caused by the drift.

    Evaluates the model on reference data and drifted data.
    Returns the drop in accuracy (percentage points).

    Args:
        model:         trained sklearn/xgboost/lightgbm model
        reference_df:  reference DataFrame with 'target' column
        drifted_df:    drifted DataFrame with 'target' column
        target_col:    name of target column

    Returns:
        dict with baseline_accuracy, drifted_accuracy, accuracy_drop
    """
    # Separate features and labels
    X_ref = reference_df.drop(columns=[target_col])
    y_ref = reference_df[target_col].values

    X_drifted = drifted_df.drop(columns=[target_col])
    y_drifted = drifted_df[target_col].values

    # Compute accuracy on reference (baseline)
    y_ref_pred = model.predict(X_ref)
    baseline_accuracy = accuracy_score(y_ref, y_ref_pred)

    # Compute accuracy on drifted data
    y_drifted_pred = model.predict(X_drifted)
    drifted_accuracy = accuracy_score(y_drifted, y_drifted_pred)

    # Accuracy drop in percentage points
    accuracy_drop_pct = (baseline_accuracy - drifted_accuracy) * 100

    # Also compute AUC if model supports predict_proba
    baseline_auc = None
    drifted_auc = None
    try:
        y_ref_proba = model.predict_proba(X_ref)[:, 1]
        y_drifted_proba = model.predict_proba(X_drifted)[:, 1]
        baseline_auc = round(roc_auc_score(y_ref, y_ref_proba), 4)
        drifted_auc = round(roc_auc_score(y_drifted, y_drifted_proba), 4)
    except Exception:
        pass

    return {
        "baseline_accuracy": round(baseline_accuracy, 4),
        "drifted_accuracy": round(drifted_accuracy, 4),
        "actual_accuracy_drop": round(accuracy_drop_pct, 4),
        "baseline_auc": baseline_auc,
        "drifted_auc": drifted_auc,
        "is_malignant": accuracy_drop_pct > 2.0  # threshold for malignant
    }


def inject_drift(reference_df,
                 model,
                 drift_type,
                 features,
                 magnitude,
                 feature_types=None,
                 interaction_pairs=None,
                 batch_size=None,
                 target_col='target',
                 seed=42):
    """
    Main injection function. Called by all team members.

    Args:
        reference_df:       clean reference DataFrame (with target column)
        model:              trained base model (sklearn/xgboost/lightgbm)
        drift_type:         one of: marginal_single, marginal_multi,
                                    interaction, combined, none
        features:           list of feature names to drift
                            For interaction: pass [feature_a, feature_b]
                            For combined: pass marginal features here
        magnitude:          float 0.0 to 1.0 (how severe the drift is)
        feature_types:      dict {feature: 'continuous'|'categorical'}
                            If None, all features assumed continuous
        interaction_pairs:  list of (feature_a, feature_b) tuples
                            For 'interaction' drift: overrides 'features' param
                            For 'combined' drift: interaction pairs to break
        batch_size:         size of drifted batch (default: same as reference)
        target_col:         name of target column in reference_df
        seed:               random seed for reproducibility

    Returns:
        dict with keys:
            'drifted_df':    pandas DataFrame (features + target)
            'ground_truth':  dict with all metadata and actual accuracy drop

    Example:
        result = inject_drift(
            reference_df=ref_df,
            model=model,
            drift_type="interaction",
            features=["education_num", "occupation"],
            magnitude=0.3,
            feature_types={
                "education_num": "continuous",
                "occupation": "categorical"
            },
            seed=42
        )
    """
    # ── VALIDATION ─────────────────────────────────────────────────────────
    assert drift_type in VALID_DRIFT_TYPES, \
        f"drift_type must be one of {VALID_DRIFT_TYPES}, got '{drift_type}'"
    assert 0.0 <= magnitude <= 1.0, \
        f"magnitude must be between 0.0 and 1.0, got {magnitude}"
    assert target_col in reference_df.columns, \
        f"target_col '{target_col}' not found in reference_df"

    # Default feature types to continuous
    if feature_types is None:
        feature_types = {f: 'continuous' for f in features}

    # Default batch size to same as reference
    if batch_size is None:
        batch_size = len(reference_df)

    # Create seeded random generator
    rng = np.random.default_rng(seed)

    # ── SEPARATE FEATURES AND TARGET ───────────────────────────────────────
    feature_cols = [c for c in reference_df.columns if c != target_col]
    ref_features = reference_df[feature_cols]

    # ── INJECT DRIFT ───────────────────────────────────────────────────────
    if drift_type == "none":
        drifted_features = inject_none(ref_features, batch_size, rng)
        # For no-drift, sample matching targets
        sample_idx = ref_features.sample(
            n=len(drifted_features),
            replace=False,
            random_state=int(rng.integers(0, 99999))
        ).index
        drifted_target = reference_df.loc[sample_idx, target_col].values[:len(drifted_features)]

    elif drift_type == "marginal_single":
        assert len(features) >= 1, "marginal_single requires at least 1 feature"
        feature = features[0]
        ftype = feature_types.get(feature, 'continuous')
        drifted_features = inject_marginal_single(
            ref_features, feature, magnitude, ftype, rng
        )
        drifted_target = reference_df[target_col].values

    elif drift_type == "marginal_multi":
        assert len(features) >= 2, "marginal_multi requires at least 2 features"
        drifted_features = inject_marginal_multi(
            ref_features, features, magnitude, feature_types, rng
        )
        drifted_target = reference_df[target_col].values

    elif drift_type == "interaction":
        if interaction_pairs is not None:
            drifted_features = inject_interaction_multi_pair(
                ref_features, interaction_pairs, magnitude, rng
            )
        else:
            assert len(features) >= 2, \
                "interaction drift requires at least 2 features [feature_a, feature_b]"
            drifted_features = inject_interaction(
                ref_features, features[0], features[1], magnitude, rng
            )
        drifted_target = reference_df[target_col].values

    elif drift_type == "combined":
        assert interaction_pairs is not None, \
            "combined drift requires interaction_pairs parameter"
        assert len(features) >= 1, \
            "combined drift requires marginal features in 'features' parameter"
        drifted_features = inject_combined(
            ref_features, features, interaction_pairs,
            magnitude, feature_types, rng
        )
        drifted_target = reference_df[target_col].values

    # ── RECONSTRUCT DRIFTED DATAFRAME WITH TARGET ──────────────────────────
    drifted_df = drifted_features.copy()
    drifted_df[target_col] = drifted_target
    drifted_df = drifted_df.reset_index(drop=True)

    # ── COMPUTE GROUND TRUTH ───────────────────────────────────────────────
    accuracy_metrics = compute_accuracy_drop(
        model, reference_df, drifted_df, target_col
    )

    # Per-feature KS scores (for validation)
    continuous_feats = [f for f, t in feature_types.items()
                        if t == 'continuous' and f in ref_features.columns]
    per_feature_ks = compute_per_feature_ks(
        ref_features, drifted_features, continuous_feats
    )

    ground_truth = {
        # What was injected
        "drift_type": drift_type,
        "magnitude": magnitude,
        "injected_features": features,
        "interaction_pairs": interaction_pairs,
        "feature_types": feature_types,
        "seed": seed,
        "batch_size": len(drifted_df),

        # Accuracy metrics
        "baseline_accuracy": accuracy_metrics["baseline_accuracy"],
        "drifted_accuracy": accuracy_metrics["drifted_accuracy"],
        "actual_accuracy_drop": accuracy_metrics["actual_accuracy_drop"],
        "baseline_auc": accuracy_metrics["baseline_auc"],
        "drifted_auc": accuracy_metrics["drifted_auc"],
        "is_malignant": accuracy_metrics["is_malignant"],
        "severity": get_severity_label(accuracy_metrics["actual_accuracy_drop"]),

        # Per-feature KS (for interaction drift validation)
        "per_feature_ks": per_feature_ks
    }

    return {
        "drifted_df": drifted_df,
        "ground_truth": ground_truth
    }


def inject_drift_batch(reference_df, model, conditions, target_col='target'):
    """
    Run multiple injection conditions in sequence.
    Used by the benchmark loop.

    Args:
        reference_df:  clean reference DataFrame
        model:         trained base model
        conditions:    list of dicts, each with keys matching inject_drift params
        target_col:    name of target column

    Returns:
        list of result dicts (same format as inject_drift)
    """
    results = []
    for i, condition in enumerate(conditions):
        try:
            result = inject_drift(
                reference_df=reference_df,
                model=model,
                target_col=target_col,
                **condition
            )
            results.append({
                "condition_id": i,
                "condition": condition,
                "result": result,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "condition_id": i,
                "condition": condition,
                "result": None,
                "status": f"error: {str(e)}"
            })
            print(f"  Condition {i} failed: {e}")

    n_success = sum(1 for r in results if r['status'] == 'success')
    print(f"Batch complete: {n_success}/{len(conditions)} succeeded")
    return results
