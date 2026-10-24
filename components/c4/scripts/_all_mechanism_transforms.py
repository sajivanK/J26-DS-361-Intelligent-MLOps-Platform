"""Offline transformations matching the frozen adapter's shared dispatch.

The future caller must verify the complete decision-evidence freeze before
opening future observations. This module does not authorize release.
"""
import numpy as np
import pandas as pd

from _common import require
from c4.data import FEATURE_COLUMNS, TARGET_COLUMN


def transform_future_role(clean, *, config, true_rule, shared, feature_types):
    require(clean.columns.tolist() == [*FEATURE_COLUMNS, TARGET_COLUMN], 'Future schema differs.')
    require(not clean.empty and clean.index.is_unique, 'Future rows must be nonempty and unique.')
    require(pd.api.types.is_integer_dtype(clean.index.dtype) and (clean.index >= 0).all(),
            'Future source IDs are invalid.')
    require(all(pd.api.types.is_numeric_dtype(clean[c]) for c in clean.columns) and
            np.isfinite(clean.to_numpy(dtype=float)).all(), 'Future values must be numeric and finite.')
    require(clean[TARGET_COLUMN].isin([0, 1]).all(), 'Future targets must be binary.')
    require(set(feature_types) == set(FEATURE_COLUMNS), 'Explicit feature types must cover Adult features.')
    require(all(v in {'continuous', 'categorical'} for v in feature_types.values()), 'Invalid feature types.')
    mask = true_rule.mask(clean).rename('true_slice_member')
    require(mask.any() and (~mask).any(), 'Future slice and complement require observations.')
    result = clean.copy(deep=True)
    if config.mechanism == 'none':
        require(config.magnitude == 0 and not config.features and not config.interaction_pairs,
                'Invalid no-drift configuration.')
        return result, mask
    require(int(mask.sum()) >= 2, 'Future injection requires at least two affected rows.')
    affected = clean.loc[mask, list(FEATURE_COLUMNS)].copy()
    rng = np.random.default_rng(config.seed)
    if config.mechanism == 'marginal_single':
        f = config.features[0]
        changed = shared.inject_marginal_single(affected, f, config.magnitude, feature_types[f], rng)
    elif config.mechanism == 'marginal_multi':
        changed = shared.inject_marginal_multi(affected, list(config.features), config.magnitude, feature_types, rng)
    elif config.mechanism == 'interaction':
        changed = shared.inject_interaction_multi_pair(affected, list(config.interaction_pairs), config.magnitude, rng)
    elif config.mechanism == 'combined':
        changed = shared.inject_combined(affected, list(config.features), list(config.interaction_pairs),
                                         config.magnitude, feature_types, rng)
    else:
        raise ValueError('Unsupported future mechanism.')
    require(changed.index.equals(affected.index) and changed.columns.tolist() == list(FEATURE_COLUMNS),
            'Future transformation changed row identity or schema.')
    for feature in FEATURE_COLUMNS:
        result.loc[mask, feature] = changed[feature]
    require(np.isfinite(result.to_numpy(dtype=float)).all(), 'Nonfinite future transformation.')
    pd.testing.assert_series_equal(result[TARGET_COLUMN], clean[TARGET_COLUMN], check_exact=True)
    pd.testing.assert_frame_equal(result.loc[~mask], clean.loc[~mask], check_exact=True)
    return result, mask


def canonical_release_frames(original_role, canonical_clean, saved_clean_bytes,
                             saved_transformed_bytes, *, config, true_rule):
    """Replay with the original parser before restoring canonical source floats."""
    import io
    from c4.scenarios.adapter import transform_role

    require(original_role.index.equals(canonical_clean.index), 'Release source indices differ.')
    expected = original_role.to_csv(index=True, lineterminator='\n').encode('utf-8')
    require(expected == saved_clean_bytes, 'Clean preview is not the exact original-role serialization.')
    replay = transform_role(original_role, config=config, true_slice=true_rule, role='release_validation')
    expected_transformed = replay.frame.to_csv(index=True, lineterminator='\n').encode('utf-8')
    require(expected_transformed == saved_transformed_bytes,
            'Transformed preview differs from deterministic frozen-adapter replay.')
    saved = pd.read_csv(io.BytesIO(saved_transformed_bytes), index_col=0, float_precision='round_trip')
    require(saved.index.equals(canonical_clean.index) and saved.columns.tolist() == canonical_clean.columns.tolist(),
            'Saved release row identity or schema differs.')
    # Preserve source-canonical values except the exact declared perturbed cells.
    membership = replay.true_membership
    result = canonical_clean.copy(deep=True)
    changed_features = set(config.features) | {f for pair in config.interaction_pairs for f in pair}
    unchanged = [c for c in original_role.columns if c not in changed_features]
    pd.testing.assert_frame_equal(replay.frame[unchanged], original_role[unchanged], check_exact=True)
    for feature in changed_features:
        result.loc[membership, feature] = saved.loc[membership, feature]
    pd.testing.assert_series_equal(result[TARGET_COLUMN], canonical_clean[TARGET_COLUMN], check_exact=True)
    pd.testing.assert_frame_equal(result.loc[~membership], canonical_clean.loc[~membership], check_exact=True)
    require(np.isfinite(result.to_numpy(dtype=float)).all(), 'Nonfinite release transformation.')
    return canonical_clean.copy(deep=True), result
