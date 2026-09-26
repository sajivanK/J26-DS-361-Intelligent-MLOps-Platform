"""
Drift Type Implementations
J26-DS-361 | Shared Artifact | Maintained by Sajivan K (IT23172296)

Four drift types implemented:
    1. Marginal Single  - one feature distribution shifts
    2. Marginal Multi   - multiple features shift simultaneously
    3. Interaction      - joint relationship breaks, marginals preserved
    4. Combined         - marginal shift + interaction break together
    5. None             - no drift (baseline sampler)

Each function returns a drifted copy of the reference dataframe.
Original dataframe is never modified.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


# ── MARGINAL SINGLE ──────────────────────────────────────────────────────────

def inject_marginal_single(reference_df, feature, magnitude,
                           feature_type, rng):
    """
    Shift the distribution of a single feature.

    For continuous features:
        Shift mean by magnitude * std of the feature.
        Distribution shape is preserved, just translated.

    For categorical features:
        Redistribute category proportions.
        magnitude=1.0 means most rows get the most common category.

    Args:
        reference_df:  clean reference DataFrame
        feature:       column name to drift
        magnitude:     float 0.0 to 1.0
        feature_type:  'continuous' or 'categorical'
        rng:           numpy random generator (seeded)

    Returns:
        drifted_df: copy with the single feature drifted
    """
    drifted_df = reference_df.copy()

    if feature_type == 'continuous':
        feature_std = reference_df[feature].std()
        feature_mean = reference_df[feature].mean()
        shift_amount = magnitude * feature_std

        # Shift all values up by shift_amount
        # Add small noise to make it realistic
        noise = rng.normal(0, feature_std * 0.05, size=len(drifted_df))
        drifted_df[feature] = reference_df[feature] + shift_amount + noise

    elif feature_type == 'categorical':
        categories = reference_df[feature].unique()
        original_probs = (reference_df[feature]
                          .value_counts(normalize=True)
                          .reindex(categories)
                          .fillna(0)
                          .values)

        # Create drifted probabilities by shifting mass to LEAST common category
        # This makes the distribution maximally different from the reference
        # At magnitude=1.0, all mass goes to the least common category
        least_common_idx = np.argmin(original_probs)
        drifted_probs = (1 - magnitude) * original_probs.copy()
        drifted_probs[least_common_idx] += magnitude
        drifted_probs = drifted_probs / drifted_probs.sum()

        drifted_df[feature] = rng.choice(
            categories, size=len(drifted_df), p=drifted_probs
        )

    return drifted_df


# ── MARGINAL MULTI ────────────────────────────────────────────────────────────

def inject_marginal_multi(reference_df, features, magnitude,
                          feature_types, rng):
    """
    Shift distributions of multiple features simultaneously.
    Each feature is shifted independently using marginal_single logic.

    Args:
        reference_df:   clean reference DataFrame
        features:       list of column names to drift
        magnitude:      float 0.0 to 1.0 (same for all features)
        feature_types:  dict of {feature_name: 'continuous'|'categorical'}
        rng:            numpy random generator (seeded)

    Returns:
        drifted_df: copy with all specified features drifted
    """
    drifted_df = reference_df.copy()

    for feature in features:
        ftype = feature_types.get(feature, 'continuous')
        # Apply each feature independently
        temp_df = inject_marginal_single(
            drifted_df, feature, magnitude, ftype, rng
        )
        drifted_df[feature] = temp_df[feature]

    return drifted_df


# ── INTERACTION ───────────────────────────────────────────────────────────────

def inject_interaction(reference_df, feature_a, feature_b,
                       magnitude, rng):
    """
    Break the joint relationship between two features
    while PRESERVING each feature's marginal distribution.

    Method: Value re-pairing (shuffle one column within a subset of rows)

    How it works:
        - Original: (edu=16, occ=Tech), (edu=10, occ=Admin), (edu=14, occ=Sales)
        - After:    (edu=16, occ=Admin), (edu=10, occ=Sales), (edu=14, occ=Tech)
        - edu column: values unchanged, distribution unchanged
        - occ column: values unchanged, distribution unchanged
        - BUT: the correlation between edu and occ is broken

    magnitude controls what fraction of rows get re-paired:
        0.1 = 10% of rows shuffled (weak interaction break)
        1.0 = 100% of rows shuffled (complete independence)

    VALIDATION: KS test on each feature individually should show ~0
    Only multivariate tests (MMD, LSDD, C2ST) should detect this.

    Args:
        reference_df:  clean reference DataFrame
        feature_a:     first feature in the pair
        feature_b:     second feature in the pair (this one gets shuffled)
        magnitude:     float 0.0 to 1.0
        rng:           numpy random generator (seeded)

    Returns:
        drifted_df: copy with interaction drift injected
    """
    drifted_df = reference_df.copy()

    n_rows = len(drifted_df)
    n_rows_to_shuffle = max(1, int(magnitude * n_rows))

    # Select rows to shuffle
    shuffle_idx = rng.choice(n_rows, size=n_rows_to_shuffle, replace=False)

    # Shuffle feature_b values within the selected rows
    # This breaks the joint distribution but preserves the marginal
    shuffled_values = drifted_df[feature_b].iloc[shuffle_idx].values.copy()
    rng.shuffle(shuffled_values)
    drifted_df.iloc[shuffle_idx, drifted_df.columns.get_loc(feature_b)] = \
        shuffled_values

    return drifted_df


def inject_interaction_multi_pair(reference_df, feature_pairs,
                                  magnitude, rng):
    """
    Inject interaction drift across multiple feature pairs.
    Each pair's joint relationship is broken independently.

    Args:
        reference_df:   clean reference DataFrame
        feature_pairs:  list of (feature_a, feature_b) tuples
        magnitude:      float 0.0 to 1.0
        rng:            numpy random generator (seeded)

    Returns:
        drifted_df with all specified feature pair interactions broken
    """
    drifted_df = reference_df.copy()

    for feature_a, feature_b in feature_pairs:
        drifted_df = inject_interaction(
            drifted_df, feature_a, feature_b, magnitude, rng
        )

    return drifted_df


# ── COMBINED ──────────────────────────────────────────────────────────────────

def inject_combined(reference_df, marginal_features, interaction_pairs,
                    magnitude, feature_types, rng):
    """
    Inject both marginal drift AND interaction drift simultaneously.

    Marginal features and interaction pairs should be DIFFERENT features
    to test the classifier's ability to detect combined drift.

    Args:
        reference_df:         clean reference DataFrame
        marginal_features:    list of features for marginal shift
        interaction_pairs:    list of (feature_a, feature_b) for interaction
        magnitude:            float 0.0 to 1.0 (applied to both)
        feature_types:        dict {feature: 'continuous'|'categorical'}
        rng:                  numpy random generator (seeded)

    Returns:
        drifted_df with combined drift
    """
    # First apply marginal drift
    drifted_df = inject_marginal_multi(
        reference_df, marginal_features, magnitude, feature_types, rng
    )

    # Then apply interaction drift on top
    for feature_a, feature_b in interaction_pairs:
        drifted_df = inject_interaction(
            drifted_df, feature_a, feature_b, magnitude, rng
        )

    return drifted_df


# ── NO DRIFT (BASELINE) ───────────────────────────────────────────────────────

def inject_none(reference_df, batch_size, rng):
    """
    No drift baseline: sample a fresh batch from reference data.

    Detectors should return near-zero scores on this.
    Classifier AUC should be ~0.5.
    This is used to calibrate false alarm rates.

    Args:
        reference_df:  clean reference DataFrame
        batch_size:    number of rows to sample
        rng:           numpy random generator (seeded)

    Returns:
        sampled_df: random sample from reference (no drift injected)
    """
    if batch_size >= len(reference_df):
        return reference_df.sample(frac=1, random_state=rng.integers(0, 99999)).reset_index(drop=True)

    return reference_df.sample(
        n=batch_size,
        replace=False,
        random_state=rng.integers(0, 99999)
    ).reset_index(drop=True)


# ── UTILITY ───────────────────────────────────────────────────────────────────

def compute_per_feature_ks(reference_df, drifted_df, continuous_features):
    """
    Compute KS statistic for each continuous feature.
    Used for validation: interaction drift should produce near-zero KS scores.

    Returns:
        dict of {feature: ks_statistic}
    """
    ks_scores = {}
    for feature in continuous_features:
        if feature in reference_df.columns and feature in drifted_df.columns:
            stat, _ = ks_2samp(
                reference_df[feature].values,
                drifted_df[feature].values
            )
            ks_scores[feature] = round(stat, 4)
    return ks_scores
