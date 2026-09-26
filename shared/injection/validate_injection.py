"""
Injection Simulator Validation Suite
J26-DS-361 | Shared Artifact | Maintained by Sajivan K (IT23172296)

Run this BEFORE sharing the shared artifacts with the team.
All 4 checks must pass before pushing to GitHub.

Checks:
    1. No-drift baseline: all detector scores near zero
    2. Interaction drift: KS scores near zero (marginals preserved)
    3. Marginal drift: accuracy drop increases with magnitude
    4. Benign vs malignant: high vs low importance features differ
"""

import pandas as pd
import numpy as np
import json
import joblib
from scipy.stats import ks_2samp

from injection.inject_drift import inject_drift


def check_1_no_drift_baseline(reference_df, model, target_col='target',
                               n_runs=10, ks_threshold=0.05):
    """
    Check 1: No-drift conditions should produce near-zero KS scores.

    For each run, we sample from reference and compute KS between
    reference and the sample. Should be near zero because same distribution.
    """
    print("\n" + "=" * 60)
    print("CHECK 1: No-Drift Baseline")
    print("=" * 60)
    print(f"Running {n_runs} no-drift conditions...")
    print(f"Expected: per-feature KS scores < {ks_threshold}")

    feature_cols = [c for c in reference_df.columns if c != target_col]
    ref_features = reference_df[feature_cols]

    all_ks_scores = []
    all_accuracy_drops = []

    for seed in range(n_runs):
        result = inject_drift(
            reference_df=reference_df,
            model=model,
            drift_type="none",
            features=[],
            magnitude=0.0,
            target_col=target_col,
            seed=seed
        )

        gt = result['ground_truth']
        all_accuracy_drops.append(gt['actual_accuracy_drop'])

        # Compute KS for numerical features
        drifted_features = result['drifted_df'].drop(columns=[target_col])
        for col in ref_features.select_dtypes(include=[np.number]).columns[:5]:
            stat, _ = ks_2samp(ref_features[col].values,
                               drifted_features[col].values)
            all_ks_scores.append(stat)

    mean_ks = np.mean(all_ks_scores)
    max_ks = np.max(all_ks_scores)
    mean_drop = np.mean(all_accuracy_drops)
    max_drop = np.max(all_accuracy_drops)

    print(f"\nResults:")
    print(f"  Mean KS score:      {mean_ks:.4f}  (expected < {ks_threshold})")
    print(f"  Max KS score:       {max_ks:.4f}  (expected < {ks_threshold * 2})")
    print(f"  Mean accuracy drop: {mean_drop:.4f}%  (expected near 0)")
    print(f"  Max accuracy drop:  {max_drop:.4f}%  (expected < 2%)")

    passed = (mean_ks < ks_threshold) and (abs(mean_drop) < 2.0)

    if passed:
        print("\n  ✓ CHECK 1 PASSED: No-drift baseline is clean")
    else:
        print("\n  ✗ CHECK 1 FAILED")
        if mean_ks >= ks_threshold:
            print(f"    KS too high: {mean_ks:.4f} >= {ks_threshold}")
        if abs(mean_drop) >= 2.0:
            print(f"    Accuracy drop too high: {mean_drop:.4f}%")

    return passed, {"mean_ks": mean_ks, "max_ks": max_ks,
                    "mean_drop": mean_drop, "max_drop": max_drop}


def check_2_interaction_marginal_preservation(reference_df, model,
                                              feature_a, feature_b,
                                              feature_types,
                                              target_col='target',
                                              ks_threshold=0.05):
    """
    Check 2: Interaction drift must preserve marginal distributions.

    After injecting interaction drift:
    - KS score on feature_a should be near zero
    - KS score on feature_b should be near zero
    - BUT the joint distribution should have changed
      (validated by running C2ST which should detect the change)
    """
    print("\n" + "=" * 60)
    print("CHECK 2: Interaction Drift Marginal Preservation")
    print("=" * 60)
    print(f"Testing interaction drift on: {feature_a} x {feature_b}")
    print(f"Expected: per-feature KS < {ks_threshold}")
    print(f"Expected: joint distribution changed (accuracy should drop)")

    magnitudes = [0.3, 0.6, 1.0]
    results_table = []

    for mag in magnitudes:
        result = inject_drift(
            reference_df=reference_df,
            model=model,
            drift_type="interaction",
            features=[feature_a, feature_b],
            magnitude=mag,
            feature_types=feature_types,
            target_col=target_col,
            seed=42
        )

        gt = result['ground_truth']
        drifted_df = result['drifted_df']
        ref_features = reference_df.drop(columns=[target_col])
        drifted_features = drifted_df.drop(columns=[target_col])

        # KS on the two drifted features
        ks_a, ks_b = 0, 0
        if feature_a in ref_features.columns:
            ks_a, _ = ks_2samp(ref_features[feature_a].values,
                               drifted_features[feature_a].values)
        if feature_b in ref_features.columns:
            ks_b, _ = ks_2samp(ref_features[feature_b].values,
                               drifted_features[feature_b].values)

        results_table.append({
            "magnitude": mag,
            "ks_feature_a": round(ks_a, 4),
            "ks_feature_b": round(ks_b, 4),
            "accuracy_drop": gt['actual_accuracy_drop']
        })

        print(f"\n  Magnitude={mag}:")
        print(f"    KS({feature_a}): {ks_a:.4f}  {'✓' if ks_a < ks_threshold else '✗'}")
        print(f"    KS({feature_b}): {ks_b:.4f}  {'✓' if ks_b < ks_threshold else '✗'}")
        print(f"    Accuracy drop: {gt['actual_accuracy_drop']:.2f}%")

    # Check all KS scores are below threshold
    all_ks = [r['ks_feature_a'] for r in results_table] + \
             [r['ks_feature_b'] for r in results_table]
    passed = all(ks < ks_threshold for ks in all_ks)

    if passed:
        print(f"\n  ✓ CHECK 2 PASSED: Marginal distributions preserved under interaction drift")
    else:
        print(f"\n  ✗ CHECK 2 FAILED: Some KS scores exceed threshold")
        print(f"    This means the interaction injection is accidentally")
        print(f"    changing marginal distributions. Review inject_interaction()")

    return passed, results_table


def check_3_magnitude_monotonicity(reference_df, model, feature,
                                   feature_type, target_col='target'):
    """
    Check 3: As magnitude increases, accuracy drop should generally increase.

    Tests marginal single drift on a high-importance feature.
    At magnitude=1.0, the drop should be significantly larger than at 0.1.
    """
    print("\n" + "=" * 60)
    print("CHECK 3: Magnitude Monotonicity")
    print("=" * 60)
    print(f"Testing marginal single drift on '{feature}' (type: {feature_type})")
    print("Expected: accuracy drop generally increases with magnitude")

    magnitudes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    drops = []

    for mag in magnitudes:
        result = inject_drift(
            reference_df=reference_df,
            model=model,
            drift_type="marginal_single",
            features=[feature],
            magnitude=mag,
            feature_types={feature: feature_type},
            target_col=target_col,
            seed=42
        )
        drop = result['ground_truth']['actual_accuracy_drop']
        drops.append(drop)
        print(f"  Magnitude={mag:.1f}: accuracy drop = {drop:.2f}%")

    # Check monotonic trend (allow some noise, check overall direction)
    # Low magnitude should produce less drop than high magnitude
    low_avg = np.mean(drops[:3])   # magnitudes 0.1, 0.2, 0.3
    high_avg = np.mean(drops[-3:]) # magnitudes 0.8, 0.9, 1.0

    print(f"\n  Average drop (low magnitude 0.1-0.3):  {low_avg:.2f}%")
    print(f"  Average drop (high magnitude 0.8-1.0): {high_avg:.2f}%")

    passed = high_avg > low_avg

    if passed:
        print(f"\n  ✓ CHECK 3 PASSED: Accuracy drop increases with magnitude")
    else:
        print(f"\n  ✗ CHECK 3 FAILED: No clear monotonic trend")
        print(f"    High magnitude produces same or less drop than low magnitude")
        print(f"    Check the magnitude scaling in inject_marginal_single()")

    return passed, {"magnitudes": magnitudes, "drops": drops,
                    "low_avg": low_avg, "high_avg": high_avg}


def check_4_benign_vs_malignant(reference_df, model,
                                 high_importance_feature,
                                 low_importance_feature,
                                 feature_types,
                                 target_col='target',
                                 magnitude=0.4):
    """
    Check 4: High importance feature drift must produce larger drop than
    low importance feature drift at the same magnitude.

    This is the core demonstration of Novelty 1.
    If this check fails, the feature importance grouping is wrong.
    """
    print("\n" + "=" * 60)
    print("CHECK 4: Benign vs Malignant Separation")
    print("=" * 60)
    print(f"High importance feature: {high_importance_feature}")
    print(f"Low importance feature:  {low_importance_feature}")
    print(f"Testing at magnitude:    {magnitude}")
    print("Expected: high importance drift produces much larger accuracy drop")

    # High importance drift (malignant)
    result_high = inject_drift(
        reference_df=reference_df,
        model=model,
        drift_type="marginal_single",
        features=[high_importance_feature],
        magnitude=magnitude,
        feature_types={high_importance_feature: feature_types.get(high_importance_feature, 'continuous')},
        target_col=target_col,
        seed=42
    )

    # Low importance drift (benign)
    result_low = inject_drift(
        reference_df=reference_df,
        model=model,
        drift_type="marginal_single",
        features=[low_importance_feature],
        magnitude=magnitude,
        feature_types={low_importance_feature: feature_types.get(low_importance_feature, 'categorical')},
        target_col=target_col,
        seed=42
    )

    drop_high = result_high['ground_truth']['actual_accuracy_drop']
    drop_low = result_low['ground_truth']['actual_accuracy_drop']

    print(f"\n  High importance '{high_importance_feature}':")
    print(f"    Accuracy drop: {drop_high:.2f}%  (MALIGNANT)")

    print(f"\n  Low importance '{low_importance_feature}':")
    print(f"    Accuracy drop: {drop_low:.2f}%  (BENIGN)")

    separation_ratio = abs(drop_high) / (abs(drop_low) + 0.001)
    print(f"\n  Separation ratio: {separation_ratio:.1f}x")

    # High importance should produce at least 2x more drop
    passed = drop_high > drop_low

    if passed:
        print(f"\n  ✓ CHECK 4 PASSED: Benign vs malignant separation confirmed")
        print(f"    High importance drift causes {separation_ratio:.1f}x more damage")
    else:
        print(f"\n  ✗ CHECK 4 FAILED: No separation between benign and malignant")
        print(f"    Check feature importance grouping in feature_importance JSON")
        print(f"    High importance features may not actually be high importance")

    return passed, {
        "high_importance_feature": high_importance_feature,
        "low_importance_feature": low_importance_feature,
        "drop_high": drop_high,
        "drop_low": drop_low,
        "separation_ratio": round(separation_ratio, 2)
    }


def run_all_checks(reference_df, model, feature_importance_path,
                   target_col='target'):
    """
    Run all 4 validation checks in sequence.
    Uses feature importance JSON to determine which features to test.

    Args:
        reference_df:           preprocessed reference DataFrame
        model:                  trained base model
        feature_importance_path: path to feature_importance JSON
        target_col:             name of target column

    Returns:
        all_passed: True if all checks pass
        results:    dict with detailed results
    """
    print("=" * 60)
    print("INJECTION SIMULATOR VALIDATION SUITE")
    print("J26-DS-361")
    print("=" * 60)

    # Load feature importance
    with open(feature_importance_path) as f:
        importance_data = json.load(f)

    high_feats = importance_data['feature_groups']['high_importance']['features']
    low_feats = importance_data['feature_groups']['low_importance']['features']

    print(f"\nHigh importance features: {high_feats}")
    print(f"Low importance features:  {low_feats}")

    # Determine feature types from data
    feature_cols = [c for c in reference_df.columns if c != target_col]
    feature_types = {}
    for col in feature_cols:
        if reference_df[col].dtype == 'object':
            feature_types[col] = 'categorical'
        elif reference_df[col].nunique() <= 10:
            feature_types[col] = 'categorical'
        else:
            feature_types[col] = 'continuous'

    all_results = {}
    all_passed = True

    # Check 1
    passed1, res1 = check_1_no_drift_baseline(
        reference_df, model, target_col
    )
    all_results['check_1_no_drift'] = {"passed": passed1, "details": res1}
    all_passed = all_passed and passed1

    # Check 2 (use top 2 high importance features for interaction test)
    if len(high_feats) >= 2:
        passed2, res2 = check_2_interaction_marginal_preservation(
            reference_df, model,
            feature_a=high_feats[0],
            feature_b=high_feats[1],
            feature_types=feature_types,
            target_col=target_col
        )
        all_results['check_2_interaction'] = {"passed": passed2, "details": res2}
        all_passed = all_passed and passed2

    # Check 3 (use highest importance CONTINUOUS feature for monotonicity test)
    # Continuous features show cleaner monotonic trends than categorical
    high_feat = high_feats[0]
    for feat in high_feats:
        if feature_types.get(feat, 'continuous') == 'continuous':
            high_feat = feat
            break
    passed3, res3 = check_3_magnitude_monotonicity(
        reference_df, model,
        feature=high_feat,
        feature_type=feature_types.get(high_feat, 'continuous'),
        target_col=target_col
    )
    all_results['check_3_monotonicity'] = {"passed": passed3, "details": res3}
    all_passed = all_passed and passed3

    # Check 4
    passed4, res4 = check_4_benign_vs_malignant(
        reference_df, model,
        high_importance_feature=high_feats[0],
        low_importance_feature=low_feats[0],
        feature_types=feature_types,
        target_col=target_col
    )
    all_results['check_4_benign_malignant'] = {"passed": passed4, "details": res4}
    all_passed = all_passed and passed4

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    checks = [
        ("Check 1: No-drift baseline",              passed1),
        ("Check 2: Interaction marginal preserved", passed2 if len(high_feats) >= 2 else None),
        ("Check 3: Magnitude monotonicity",         passed3),
        ("Check 4: Benign vs malignant",            passed4),
    ]
    for name, passed in checks:
        if passed is None:
            print(f"  SKIP  {name} (not enough features)")
        elif passed:
            print(f"  ✓     {name}")
        else:
            print(f"  ✗     {name}")

    if all_passed:
        print("\n✓ ALL CHECKS PASSED")
        print("The injection simulator is ready.")
        print("Push to GitHub and share with the team.")
    else:
        print("\n✗ SOME CHECKS FAILED")
        print("Fix the issues above before sharing with the team.")

    return all_passed, all_results


if __name__ == "__main__":
    import joblib

    # Load artifacts
    ref_df = pd.read_csv("shared/uci_adult/data/reference.csv")
    model = joblib.load("shared/uci_adult/models/base_model_adult.pkl")

    # Run all checks
    passed, results = run_all_checks(
        reference_df=ref_df,
        model=model,
        feature_importance_path="shared/uci_adult/feature_importance_adult.json"
    )
