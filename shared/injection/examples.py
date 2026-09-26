"""
Injection Simulator Usage Examples
J26-DS-361 | For all team members

Shows exactly how to call inject_drift for each drift type.
Copy these examples into your own notebook.
"""

import pandas as pd
import numpy as np
import joblib
import json
import sys

sys.path.append('J26-DS-361-Intelligent-MLOps-Platform/shared')

from injection.inject_drift import inject_drift


def setup():
    """Load shared artifacts. Run this first."""
    ref_df = pd.read_csv(
        'J26-DS-361-Intelligent-MLOps-Platform/shared/uci_adult/data/reference.csv'
    )
    model = joblib.load(
        'J26-DS-361-Intelligent-MLOps-Platform/shared/uci_adult/models/base_model_adult.pkl'
    )
    with open('J26-DS-361-Intelligent-MLOps-Platform/shared/uci_adult/feature_importance_adult.json') as f:
        importance = json.load(f)

    high_feats = importance['feature_groups']['high_importance']['features']
    low_feats = importance['feature_groups']['low_importance']['features']

    print(f"Reference data: {ref_df.shape}")
    print(f"High importance features: {high_feats}")
    print(f"Low importance features:  {low_feats}")

    return ref_df, model, high_feats, low_feats


# ── EXAMPLE 1: Marginal Single (one feature shifts) ──────────────────────────

def example_marginal_single():
    ref_df, model, high_feats, low_feats = setup()

    print("\n=== EXAMPLE 1: Marginal Single Drift ===")

    result = inject_drift(
        reference_df=ref_df,
        model=model,
        drift_type="marginal_single",
        features=[high_feats[0]],           # drift one high importance feature
        magnitude=0.4,
        feature_types={high_feats[0]: "continuous"},
        target_col='target',
        seed=42
    )

    gt = result['ground_truth']
    print(f"Drift type:      {gt['drift_type']}")
    print(f"Drifted feature: {gt['injected_features']}")
    print(f"Magnitude:       {gt['magnitude']}")
    print(f"Accuracy drop:   {gt['actual_accuracy_drop']:.2f}%")
    print(f"Severity:        {gt['severity']}")
    print(f"Is malignant:    {gt['is_malignant']}")
    print(f"Drifted data shape: {result['drifted_df'].shape}")

    return result


# ── EXAMPLE 2: Interaction Drift (joint relationship breaks) ─────────────────

def example_interaction():
    ref_df, model, high_feats, low_feats = setup()

    print("\n=== EXAMPLE 2: Interaction Drift ===")
    print("Breaks joint relationship between two features")
    print("Marginal distributions are PRESERVED")

    result = inject_drift(
        reference_df=ref_df,
        model=model,
        drift_type="interaction",
        features=[high_feats[0], high_feats[1]],  # pair to break
        magnitude=0.5,
        feature_types={
            high_feats[0]: "continuous",
            high_feats[1]: "categorical"
        },
        target_col='target',
        seed=42
    )

    gt = result['ground_truth']
    print(f"Drift type:         {gt['drift_type']}")
    print(f"Broken pair:        {gt['injected_features']}")
    print(f"Accuracy drop:      {gt['actual_accuracy_drop']:.2f}%")
    print(f"Is malignant:       {gt['is_malignant']}")
    print(f"Per-feature KS:     {gt['per_feature_ks']}")
    print("(KS scores should be near zero - marginals preserved)")

    return result


# ── EXAMPLE 3: Benign Drift (low importance feature) ─────────────────────────

def example_benign_drift():
    ref_df, model, high_feats, low_feats = setup()

    print("\n=== EXAMPLE 3: Benign Drift (same magnitude, low importance) ===")
    print("Same magnitude as Example 1 but on low importance feature")
    print("Expected: much smaller accuracy drop")

    result = inject_drift(
        reference_df=ref_df,
        model=model,
        drift_type="marginal_single",
        features=[low_feats[0]],            # low importance feature
        magnitude=0.4,                      # same magnitude as Example 1
        feature_types={low_feats[0]: "categorical"},
        target_col='target',
        seed=42
    )

    gt = result['ground_truth']
    print(f"Drifted feature: {gt['injected_features']}")
    print(f"Accuracy drop:   {gt['actual_accuracy_drop']:.2f}%")
    print(f"Severity:        {gt['severity']}")
    print(f"Is malignant:    {gt['is_malignant']}")
    print("\nCompare with Example 1 - this is the benign vs malignant separation")

    return result


# ── EXAMPLE 4: Combined Drift ─────────────────────────────────────────────────

def example_combined():
    ref_df, model, high_feats, low_feats = setup()

    print("\n=== EXAMPLE 4: Combined Drift ===")
    print("Marginal shift + interaction break simultaneously")

    result = inject_drift(
        reference_df=ref_df,
        model=model,
        drift_type="combined",
        features=[high_feats[0]],               # marginal shift on this
        interaction_pairs=[(high_feats[1], high_feats[2])],  # break this pair
        magnitude=0.4,
        feature_types={
            high_feats[0]: "continuous",
            high_feats[1]: "continuous",
            high_feats[2]: "categorical"
        },
        target_col='target',
        seed=42
    )

    gt = result['ground_truth']
    print(f"Drift type:        {gt['drift_type']}")
    print(f"Marginal features: {gt['injected_features']}")
    print(f"Interaction pairs: {gt['interaction_pairs']}")
    print(f"Accuracy drop:     {gt['actual_accuracy_drop']:.2f}%")
    print(f"Severity:          {gt['severity']}")

    return result


# ── EXAMPLE 5: No Drift Baseline ─────────────────────────────────────────────

def example_no_drift():
    ref_df, model, high_feats, low_feats = setup()

    print("\n=== EXAMPLE 5: No Drift Baseline ===")
    print("Sample from reference - detectors should see nothing")

    result = inject_drift(
        reference_df=ref_df,
        model=model,
        drift_type="none",
        features=[],
        magnitude=0.0,
        target_col='target',
        seed=42
    )

    gt = result['ground_truth']
    print(f"Accuracy drop:  {gt['actual_accuracy_drop']:.2f}%  (expected ~0)")
    print(f"Is malignant:   {gt['is_malignant']}  (expected False)")

    return result


# ── EXAMPLE 6: How to use ground truth in YOUR research ─────────────────────

def example_using_ground_truth():
    """
    This is the core pattern each team member uses.

    You inject drift -> you get ground truth ->
    you use ground truth as substitute for upstream component outputs.
    """
    ref_df, model, high_feats, low_feats = setup()

    print("\n=== EXAMPLE 6: Using Ground Truth in Research ===")

    result = inject_drift(
        reference_df=ref_df,
        model=model,
        drift_type="interaction",
        features=[high_feats[0], high_feats[1]],
        magnitude=0.3,
        feature_types={high_feats[0]: "continuous", high_feats[1]: "categorical"},
        seed=42
    )

    gt = result['ground_truth']
    drifted_df = result['drifted_df']

    print("\nGround truth (substitute for C1 output):")
    print(f"  drift_type:           {gt['drift_type']}")
    print(f"  actual_accuracy_drop: {gt['actual_accuracy_drop']}%")
    print(f"  severity:             {gt['severity']}")
    print(f"  is_malignant:         {gt['is_malignant']}")

    print("\nGround truth (substitute for C2 output):")
    print(f"  injected_features: {gt['injected_features']}")
    print(f"  (these are the features C2 would identify as root cause)")

    print("\nYour research code runs on the actual drifted data:")
    print(f"  drifted_df shape: {drifted_df.shape}")
    print(f"  Use this as your production batch")
    print(f"  Your component processes this and compares against ground truth")


if __name__ == "__main__":
    print("Running all examples...\n")
    r1 = example_marginal_single()
    r2 = example_interaction()
    r3 = example_benign_drift()
    r4 = example_combined()
    r5 = example_no_drift()
    example_using_ground_truth()

    print("\n" + "=" * 60)
    print("All examples complete.")
    print("\nKey takeaway for each team member:")
    print("  result = inject_drift(ref_df, model, drift_type, features, magnitude, seed)")
    print("  drifted_df   = result['drifted_df']    ← use as production batch")
    print("  ground_truth = result['ground_truth']  ← substitute for upstream outputs")
