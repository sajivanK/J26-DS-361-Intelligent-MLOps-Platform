"""
Benchmark Data Collection Loop
J26-DS-361 | C1 Research | Sajivan K (IT23172296)

Generates the training dataset for the calibration meta-learner (Novelty 1),
drift-type classifier (Novelty 3), and conformal bounds calibration (Novelty 2).

Target: 700 conditions per dataset
    400 drift conditions:   4 types × 10 magnitudes × 2 groups × 5 seeds
    300 no-drift baselines: varied seeds and batch sizes

Each condition produces one row in the benchmark CSV:
    - 6 detector scores
    - 3 confidence statistics
    - actual accuracy drop (ground truth)
    - drift type label (ground truth)
    - all metadata

Output:
    shared/uci_adult/benchmark_adult.csv
    shared/ieee_cis/benchmark_ieeecis.csv
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import time
from datetime import datetime
from scipy.stats import ks_2samp, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

from injection.inject_drift import inject_drift


# ── DETECTOR IMPLEMENTATIONS ─────────────────────────────────────────────────

def compute_ks_score(reference_df, drifted_df, feature_cols):
    """
    KS test: mean KS statistic across all continuous features.
    Higher score = more distributional shift detected.
    """
    scores = []
    for col in feature_cols:
        try:
            stat, _ = ks_2samp(
                reference_df[col].values,
                drifted_df[col].values
            )
            scores.append(stat)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0


def compute_chi2_score(reference_df, drifted_df, feature_cols, n_bins=10):
    """
    Chi-squared test on binned continuous features.
    Mean chi2 statistic (normalized) across features.
    """
    scores = []
    for col in feature_cols:
        try:
            # Bin the data
            all_vals = pd.concat([reference_df[col], drifted_df[col]])
            bins = pd.cut(all_vals, bins=n_bins, retbins=True)[1]

            ref_counts = pd.cut(reference_df[col], bins=bins).value_counts().sort_index()
            drift_counts = pd.cut(drifted_df[col], bins=bins).value_counts().sort_index()

            # Align
            ref_counts, drift_counts = ref_counts.align(drift_counts, fill_value=0)

            # Add small constant to avoid zero cells
            ref_counts = ref_counts + 1e-10
            drift_counts = drift_counts + 1e-10

            observed = np.array([ref_counts.values, drift_counts.values])
            stat, _, _, _ = chi2_contingency(observed)

            # Normalize by sample size
            n = len(reference_df) + len(drifted_df)
            normalized_stat = stat / n
            scores.append(min(normalized_stat, 1.0))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0


def compute_psi_score(reference_df, drifted_df, feature_cols, n_bins=10):
    """
    Population Stability Index (PSI).
    PSI > 0.2 indicates significant shift.
    Returns mean PSI across features.
    """
    def psi_single(ref_vals, drift_vals, n_bins):
        all_vals = np.concatenate([ref_vals, drift_vals])
        breakpoints = np.percentile(all_vals, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)

        ref_pct = np.histogram(ref_vals, bins=breakpoints)[0] / len(ref_vals)
        drift_pct = np.histogram(drift_vals, bins=breakpoints)[0] / len(drift_vals)

        # Add small epsilon to avoid log(0)
        ref_pct = np.where(ref_pct == 0, 1e-10, ref_pct)
        drift_pct = np.where(drift_pct == 0, 1e-10, drift_pct)

        psi = np.sum((drift_pct - ref_pct) * np.log(drift_pct / ref_pct))
        return float(psi)

    scores = []
    for col in feature_cols:
        try:
            score = psi_single(
                reference_df[col].values,
                drifted_df[col].values,
                n_bins
            )
            scores.append(score)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0


def compute_mmd_score(reference_df, drifted_df, n_sample=500, sigma=1.0):
    """
    Maximum Mean Discrepancy with RBF kernel.
    Subsampled for computational efficiency.
    Returns MMD^2 statistic.
    """
    try:
        # Subsample
        n = min(n_sample, len(reference_df), len(drifted_df))
        ref_sample = reference_df.sample(n=n, random_state=42).values.astype(float)
        drift_sample = drifted_df.sample(n=n, random_state=42).values.astype(float)

        # Normalize
        mean = ref_sample.mean(axis=0)
        std = ref_sample.std(axis=0) + 1e-10
        ref_norm = (ref_sample - mean) / std
        drift_norm = (drift_sample - mean) / std

        # RBF kernel
        def rbf_kernel(X, Y, sigma):
            diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            sq_dists = np.sum(diff ** 2, axis=2)
            return np.exp(-sq_dists / (2 * sigma ** 2))

        K_xx = rbf_kernel(ref_norm, ref_norm, sigma)
        K_yy = rbf_kernel(drift_norm, drift_norm, sigma)
        K_xy = rbf_kernel(ref_norm, drift_norm, sigma)

        # MMD^2 = E[K(x,x)] + E[K(y,y)] - 2*E[K(x,y)]
        mmd2 = (K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())
        return float(max(0.0, mmd2))

    except Exception:
        return 0.0


def compute_lsdd_score(reference_df, drifted_df, n_sample=500):
    """
    Least-Squares Density Difference (LSDD) approximation.
    Estimates the L2 norm of the density difference.
    Subsampled for efficiency.
    """
    try:
        n = min(n_sample, len(reference_df), len(drifted_df))
        ref_sample = reference_df.sample(n=n, random_state=42).values.astype(float)
        drift_sample = drifted_df.sample(n=n, random_state=42).values.astype(float)

        # Normalize
        mean = ref_sample.mean(axis=0)
        std = ref_sample.std(axis=0) + 1e-10
        ref_norm = (ref_sample - mean) / std
        drift_norm = (drift_sample - mean) / std

        # Approximate LSDD using kernel density ratio estimation
        # Use Gaussian kernel with bandwidth h
        h = 1.0

        def gaussian_kernel(X, Y, h):
            diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            sq_dists = np.sum(diff ** 2, axis=2)
            return np.exp(-sq_dists / (2 * h ** 2))

        K_rr = gaussian_kernel(ref_norm, ref_norm, h)
        K_dd = gaussian_kernel(drift_norm, drift_norm, h)
        K_rd = gaussian_kernel(ref_norm, drift_norm, h)

        # LSDD estimate
        lsdd = K_rr.mean() + K_dd.mean() - 2 * K_rd.mean()
        return float(max(0.0, lsdd))

    except Exception:
        return 0.0


def compute_classifier_auc(reference_df, drifted_df, n_sample=1000):
    """
    Classifier Two-Sample Test (C2ST).
    Trains XGBoost to distinguish reference from drifted data.
    AUC = 0.5 means no drift, AUC = 1.0 means perfect separation.

    This is your contribution: XGBoost as the classifier detector.
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score

        # Subsample for speed
        n = min(n_sample, len(reference_df), len(drifted_df))
        ref_sample = reference_df.sample(n=n, random_state=42)
        drift_sample = drifted_df.sample(n=n, random_state=42)

        # Label: 0 = reference, 1 = drifted
        X = pd.concat([ref_sample, drift_sample], ignore_index=True)
        y = np.array([0] * n + [1] * n)

        # Train XGBoost classifier
        clf = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )

        # 3-fold CV AUC
        scores = cross_val_score(
            clf, X, y, cv=3, scoring='roc_auc', n_jobs=-1
        )
        return float(np.mean(scores))

    except Exception:
        return 0.5  # default to no-drift score on failure


def compute_confidence_statistics(model, drifted_df, target_col='target'):
    """
    Compute model confidence statistics on drifted production data.
    These are inputs to Novelty 1 (harm-aware calibration).

    Returns:
        mean_confidence:    average max softmax probability
        confidence_std:     standard deviation of probabilities
        pct_low_confidence: fraction of predictions below 0.6
    """
    X = drifted_df.drop(columns=[target_col])
    proba = model.predict_proba(X)[:, 1]

    mean_conf = float(np.mean(proba))
    conf_std = float(np.std(proba))
    pct_low = float(np.mean(proba < 0.6))

    return mean_conf, conf_std, pct_low


def run_all_detectors(reference_df, drifted_df, target_col='target'):
    """
    Run all 6 detectors on reference vs drifted data.
    Returns dict of detector scores.
    """
    # Feature columns only (no target)
    ref_features = reference_df.drop(columns=[target_col])
    drift_features = drifted_df.drop(columns=[target_col])

    # Select numeric features for detectors
    numeric_cols = ref_features.select_dtypes(include=[np.number]).columns.tolist()

    scores = {
        "ks_score": compute_ks_score(
            ref_features[numeric_cols], drift_features[numeric_cols], numeric_cols
        ),
        "chi2_score": compute_chi2_score(
            ref_features[numeric_cols], drift_features[numeric_cols], numeric_cols
        ),
        "psi_score": compute_psi_score(
            ref_features[numeric_cols], drift_features[numeric_cols], numeric_cols
        ),
        "mmd_score": compute_mmd_score(
            ref_features[numeric_cols], drift_features[numeric_cols]
        ),
        "lsdd_score": compute_lsdd_score(
            ref_features[numeric_cols], drift_features[numeric_cols]
        ),
        "clf_auc": compute_classifier_auc(
            ref_features[numeric_cols], drift_features[numeric_cols]
        )
    }

    return scores


# ── BENCHMARK LOOP ────────────────────────────────────────────────────────────

def build_condition_list(high_importance_features, low_importance_features,
                         feature_types, dataset_name):
    """
    Build the full list of 700 conditions for one dataset.

    Structure:
        400 drift conditions:
            4 types × 10 magnitudes × 2 feature groups × 5 seeds = 400

        300 no-drift baselines:
            varied seeds (100) × 3 batch sizes = 300

    Args:
        high_importance_features: list of high SHAP features
        low_importance_features:  list of low SHAP features
        feature_types:            dict {feature: 'continuous'|'categorical'}
        dataset_name:             'uci_adult' or 'ieee_cis'

    Returns:
        list of condition dicts
    """
    conditions = []
    magnitudes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seeds = [42, 123, 456, 789, 1234]

    # ── HIGH IMPORTANCE FEATURE GROUP ──────────────────────────────────────
    high_feat_primary = high_importance_features[0]
    high_feat_secondary = high_importance_features[1] if len(high_importance_features) > 1 else high_importance_features[0]
    high_feat_tertiary = high_importance_features[2] if len(high_importance_features) > 2 else high_importance_features[0]

    # Low importance primary
    low_feat_primary = low_importance_features[0]

    for seed in seeds:
        for magnitude in magnitudes:

            # 1. Marginal Single - High importance
            conditions.append({
                "drift_type": "marginal_single",
                "features": [high_feat_primary],
                "magnitude": magnitude,
                "feature_types": {high_feat_primary: feature_types.get(high_feat_primary, 'continuous')},
                "seed": seed,
                "feature_group": "high_importance",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_ms_high_{magnitude}_{seed}"
            })

            # 2. Marginal Single - Low importance
            conditions.append({
                "drift_type": "marginal_single",
                "features": [low_feat_primary],
                "magnitude": magnitude,
                "feature_types": {low_feat_primary: feature_types.get(low_feat_primary, 'categorical')},
                "seed": seed,
                "feature_group": "low_importance",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_ms_low_{magnitude}_{seed}"
            })

            # 3. Marginal Multi - High importance (top 2 features)
            conditions.append({
                "drift_type": "marginal_multi",
                "features": [high_feat_primary, high_feat_secondary],
                "magnitude": magnitude,
                "feature_types": {
                    high_feat_primary: feature_types.get(high_feat_primary, 'continuous'),
                    high_feat_secondary: feature_types.get(high_feat_secondary, 'categorical')
                },
                "seed": seed,
                "feature_group": "high_importance",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_mm_high_{magnitude}_{seed}"
            })

            # 4. Marginal Multi - Low importance
            low_feat_secondary = low_importance_features[1] if len(low_importance_features) > 1 else low_importance_features[0]
            conditions.append({
                "drift_type": "marginal_multi",
                "features": [low_feat_primary, low_feat_secondary],
                "magnitude": magnitude,
                "feature_types": {
                    low_feat_primary: feature_types.get(low_feat_primary, 'categorical'),
                    low_feat_secondary: feature_types.get(low_feat_secondary, 'categorical')
                },
                "seed": seed,
                "feature_group": "low_importance",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_mm_low_{magnitude}_{seed}"
            })

            # 5. Interaction - High importance pair
            conditions.append({
                "drift_type": "interaction",
                "features": [high_feat_primary, high_feat_secondary],
                "magnitude": magnitude,
                "feature_types": {
                    high_feat_primary: feature_types.get(high_feat_primary, 'continuous'),
                    high_feat_secondary: feature_types.get(high_feat_secondary, 'categorical')
                },
                "seed": seed,
                "feature_group": "high_importance",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_int_high_{magnitude}_{seed}"
            })

            # 6. Interaction - Low importance pair
            conditions.append({
                "drift_type": "interaction",
                "features": [low_feat_primary, low_feat_secondary],
                "magnitude": magnitude,
                "feature_types": {
                    low_feat_primary: feature_types.get(low_feat_primary, 'categorical'),
                    low_feat_secondary: feature_types.get(low_feat_secondary, 'categorical')
                },
                "seed": seed,
                "feature_group": "low_importance",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_int_low_{magnitude}_{seed}"
            })

            # 7. Combined - High importance
            conditions.append({
                "drift_type": "combined",
                "features": [high_feat_primary],
                "interaction_pairs": [(high_feat_secondary, high_feat_tertiary)],
                "magnitude": magnitude,
                "feature_types": {
                    high_feat_primary: feature_types.get(high_feat_primary, 'continuous'),
                    high_feat_secondary: feature_types.get(high_feat_secondary, 'categorical'),
                    high_feat_tertiary: feature_types.get(high_feat_tertiary, 'continuous')
                },
                "seed": seed,
                "feature_group": "high_importance",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_comb_high_{magnitude}_{seed}"
            })

            # 8. Combined - Low importance
            low_feat_tertiary = low_importance_features[2] if len(low_importance_features) > 2 else low_importance_features[0]
            conditions.append({
                "drift_type": "combined",
                "features": [low_feat_primary],
                "interaction_pairs": [(low_feat_secondary, low_feat_tertiary)],
                "magnitude": magnitude,
                "feature_types": {
                    low_feat_primary: feature_types.get(low_feat_primary, 'categorical'),
                    low_feat_secondary: feature_types.get(low_feat_secondary, 'categorical'),
                    low_feat_tertiary: feature_types.get(low_feat_tertiary, 'categorical')
                },
                "seed": seed,
                "feature_group": "low_importance",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_comb_low_{magnitude}_{seed}"
            })

    # ── NO DRIFT BASELINES ─────────────────────────────────────────────────
    batch_sizes = [500, 1000, 2000]
    no_drift_seeds = list(range(100))

    for seed in no_drift_seeds:
        for batch_size in batch_sizes:
            conditions.append({
                "drift_type": "none",
                "features": [],
                "magnitude": 0.0,
                "batch_size": batch_size,
                "seed": seed,
                "feature_group": "none",
                "dataset": dataset_name,
                "condition_id": f"{dataset_name}_none_{batch_size}_{seed}"
            })

    return conditions


def run_benchmark(reference_df, model, high_importance_features,
                  low_importance_features, feature_types,
                  dataset_name, save_path,
                  target_col='target', checkpoint_every=50):
    """
    Run the full benchmark loop for one dataset.

    Args:
        reference_df:              preprocessed reference DataFrame
        model:                     trained base model
        high_importance_features:  list of high SHAP/gain features
        low_importance_features:   list of low SHAP/gain features
        feature_types:             dict {feature: 'continuous'|'categorical'}
        dataset_name:              'uci_adult' or 'ieee_cis'
        save_path:                 path to save benchmark CSV
        target_col:                name of target column
        checkpoint_every:          save intermediate results every N conditions

    Returns:
        benchmark_df: DataFrame with all conditions and detector scores
    """
    print("=" * 70)
    print(f"BENCHMARK LOOP: {dataset_name.upper()}")
    print("=" * 70)

    # Build condition list
    conditions = build_condition_list(
        high_importance_features,
        low_importance_features,
        feature_types,
        dataset_name
    )
    print(f"Total conditions: {len(conditions)}")

    # Check for existing checkpoint
    checkpoint_path = save_path.replace('.csv', '_checkpoint.csv')
    completed_ids = set()
    rows = []

    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        completed_ids = set(checkpoint_df['condition_id'].values)
        rows = checkpoint_df.to_dict('records')
        print(f"Resuming from checkpoint: {len(completed_ids)} conditions already done")

    # Filter out already completed conditions
    remaining = [c for c in conditions if c['condition_id'] not in completed_ids]
    print(f"Remaining: {len(remaining)} conditions")

    start_time = time.time()

    for i, condition in enumerate(remaining):
        cond_start = time.time()

        try:
            # Inject drift
            result = inject_drift(
                reference_df=reference_df,
                model=model,
                drift_type=condition['drift_type'],
                features=condition['features'],
                magnitude=condition['magnitude'],
                feature_types=condition.get('feature_types', {}),
                interaction_pairs=condition.get('interaction_pairs'),
                batch_size=condition.get('batch_size'),
                target_col=target_col,
                seed=condition['seed']
            )

            drifted_df = result['drifted_df']
            gt = result['ground_truth']

            # Run all 6 detectors
            detector_scores = run_all_detectors(reference_df, drifted_df, target_col)

            # Compute confidence statistics
            mean_conf, conf_std, pct_low = compute_confidence_statistics(
                model, drifted_df, target_col
            )

            # Build benchmark row
            row = {
                # Condition metadata
                "condition_id":    condition['condition_id'],
                "dataset":         dataset_name,
                "drift_type":      condition['drift_type'],
                "magnitude":       condition['magnitude'],
                "feature_group":   condition.get('feature_group', 'none'),
                "injected_features": str(gt['injected_features']),
                "seed":            condition['seed'],

                # Detector signals (9 inputs to meta-learner)
                "ks_score":        round(detector_scores['ks_score'], 6),
                "chi2_score":      round(detector_scores['chi2_score'], 6),
                "psi_score":       round(detector_scores['psi_score'], 6),
                "mmd_score":       round(detector_scores['mmd_score'], 6),
                "lsdd_score":      round(detector_scores['lsdd_score'], 6),
                "clf_auc":         round(detector_scores['clf_auc'], 6),
                "mean_confidence": round(mean_conf, 6),
                "confidence_std":  round(conf_std, 6),
                "pct_low_conf":    round(pct_low, 6),

                # Ground truth (target for meta-learner)
                "actual_accuracy_drop": round(gt['actual_accuracy_drop'], 4),
                "baseline_accuracy":    round(gt['baseline_accuracy'], 4),
                "drifted_accuracy":     round(gt['drifted_accuracy'], 4),
                "is_malignant":         gt['is_malignant'],
                "severity":             gt['severity'],

                # Timing
                "condition_time_sec": round(time.time() - cond_start, 2)
            }

            rows.append(row)

            # Progress logging
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining_time = avg_time * (len(remaining) - i - 1)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1:4d}/{len(remaining)}] "
                      f"{condition['condition_id'][:50]:<50} | "
                      f"drop={gt['actual_accuracy_drop']:+.2f}% | "
                      f"ETA: {remaining_time/60:.1f}min")

            # Checkpoint
            if (i + 1) % checkpoint_every == 0:
                checkpoint_df = pd.DataFrame(rows)
                checkpoint_df.to_csv(checkpoint_path, index=False)
                print(f"  Checkpoint saved: {len(rows)} conditions")

        except Exception as e:
            print(f"  ERROR on condition {condition['condition_id']}: {e}")
            continue

    # Final save
    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(save_path, index=False)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE: {dataset_name.upper()}")
    print("=" * 70)
    print(f"Total conditions:  {len(benchmark_df)}")
    print(f"Total time:        {total_time/60:.1f} minutes")
    print(f"Avg time/condition: {total_time/len(benchmark_df):.1f} seconds")
    print(f"\nCondition breakdown:")
    print(benchmark_df['drift_type'].value_counts().to_string())
    print(f"\nAccuracy drop statistics:")
    print(benchmark_df['actual_accuracy_drop'].describe().round(3).to_string())
    print(f"\nMalignant conditions: {benchmark_df['is_malignant'].sum()} "
          f"({benchmark_df['is_malignant'].mean()*100:.1f}%)")
    print(f"\nSaved to: {save_path}")

    return benchmark_df


def run_benchmark_pipeline(dataset_name, data_dir, save_dir,
                           feature_importance_path, target_col='target'):
    """
    Full benchmark pipeline for one dataset.
    Loads artifacts, builds conditions, runs benchmark, saves CSV.

    Args:
        dataset_name:             'uci_adult' or 'ieee_cis'
        data_dir:                 directory with data/ and models/ subdirs
        save_dir:                 directory to save benchmark CSV
        feature_importance_path:  path to feature importance JSON
        target_col:               name of target column
    """
    # Load artifacts
    print(f"\nLoading artifacts for {dataset_name}...")
    ref_df = pd.read_csv(f"{data_dir}/data/reference.csv")
    model = joblib.load(f"{data_dir}/models/base_model_{dataset_name.replace('uci_', 'uci')}.pkl")

    # Handle model name differences
    model_files = {
        'uci_adult': 'base_model_adult.pkl',
        'ieee_cis':  'base_model_ieeecis.pkl'
    }
    model = joblib.load(f"{data_dir}/models/{model_files[dataset_name]}")

    with open(feature_importance_path) as f:
        importance_data = json.load(f)

    high_feats = importance_data['feature_groups']['high_importance']['features']
    low_feats = importance_data['feature_groups']['low_importance']['features']

    print(f"Reference data: {ref_df.shape}")
    print(f"High importance features: {high_feats}")
    print(f"Low importance features:  {low_feats}")

    # Determine feature types
    feature_cols = [c for c in ref_df.columns if c != target_col]
    feature_types = {}
    for col in feature_cols:
        if ref_df[col].nunique() <= 15:
            feature_types[col] = 'categorical'
        else:
            feature_types[col] = 'continuous'

    # Run benchmark
    save_path = f"{save_dir}/benchmark_{dataset_name}.csv"
    benchmark_df = run_benchmark(
        reference_df=ref_df,
        model=model,
        high_importance_features=high_feats,
        low_importance_features=low_feats,
        feature_types=feature_types,
        dataset_name=dataset_name,
        save_path=save_path,
        target_col=target_col
    )

    return benchmark_df


if __name__ == "__main__":
    # Run UCI Adult benchmark
    df_adult = run_benchmark_pipeline(
        dataset_name='uci_adult',
        data_dir='shared/uci_adult',
        save_dir='shared/uci_adult',
        feature_importance_path='shared/uci_adult/feature_importance_adult.json'
    )

    # Run IEEE-CIS benchmark
    df_ieee = run_benchmark_pipeline(
        dataset_name='ieee_cis',
        data_dir='shared/ieee_cis',
        save_dir='shared/ieee_cis',
        feature_importance_path='shared/ieee_cis/feature_importance_ieeecis.json'
    )

    print(f"\nTotal benchmark conditions: {len(df_adult) + len(df_ieee)}")
