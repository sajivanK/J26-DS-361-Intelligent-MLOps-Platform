"""Offline future-label helpers; never called by the release controller."""
import numpy as np
import pandas as pd

from _common import require, checked_bytes
from c4.data import FEATURE_COLUMNS, TARGET_COLUMN
from c4.models.evaluation import _model_outputs, _metrics


def verify_freeze_dependencies(snapshot):
    require(snapshot.get('complete_decision_evidence_frozen') is True,
            'Complete decision-evidence freeze is required.')
    require(snapshot.get('future_evaluation_executed') is False,
            'Expected the original pre-future freeze.')
    require(snapshot.get('scope') == 'one_admitted_forecast_training_scenario_under_simulated_upstream_evidence',
            'Wrong future research scope.')
    require(bool(snapshot.get('dependencies')), 'Freeze has no dependency manifest.')
    for path, checksum in snapshot['dependencies'].items():
        checked_bytes(path, checksum)


def transform_interaction_future(clean, *, config, true_rule, shared):
    """Same shared interaction mechanism, behind the caller's freeze verification."""
    require(config.mechanism == 'interaction' and not config.features,
            'This scoped evaluator supports interaction drift only.')
    require(clean.columns.tolist() == [*FEATURE_COLUMNS, TARGET_COLUMN], 'Future schema differs.')
    require(not clean.empty and clean.index.is_unique, 'Future rows must be nonempty and unique.')
    require(pd.api.types.is_integer_dtype(clean.index.dtype) and (clean.index >= 0).all(), 'Future source IDs are invalid.')
    require(all(pd.api.types.is_numeric_dtype(clean[c]) for c in clean.columns) and
            np.isfinite(clean.to_numpy(dtype=float)).all(), 'Future values must be numeric and finite.')
    require(clean[TARGET_COLUMN].isin([0, 1]).all(), 'Future targets must be binary.')
    mask = true_rule.mask(clean).rename('true_slice_member')
    require(int(mask.sum()) >= 2 and (~mask).any(), 'Future populations have insufficient rows.')
    affected = clean.loc[mask, list(FEATURE_COLUMNS)].copy()
    drifted = shared.inject_interaction_multi_pair(affected, list(config.interaction_pairs),
                                                  config.magnitude, np.random.default_rng(config.seed))
    require(drifted.index.equals(affected.index) and drifted.columns.tolist() == list(FEATURE_COLUMNS),
            'Future injection changed row identity or schema.')
    transformed = clean.copy(deep=True)
    for feature in FEATURE_COLUMNS:
        transformed.loc[mask, feature] = drifted[feature]
    require(np.isfinite(transformed.to_numpy(dtype=float)).all(), 'Nonfinite future transformation.')
    pd.testing.assert_series_equal(transformed[TARGET_COLUMN], clean[TARGET_COLUMN], check_exact=True)
    pd.testing.assert_frame_equal(transformed.loc[~mask], clean.loc[~mask], check_exact=True)
    return transformed, mask


def future_comparisons(champion, candidate, frame, true_mask, diagnosed_mask):
    for mask in (true_mask, diagnosed_mask):
        require(mask.index.equals(frame.index) and pd.api.types.is_bool_dtype(mask.dtype) and not mask.isna().any(),
                'Future mask indices or booleans differ.')
        require(mask.any() and (~mask).any(), 'Both future populations require observations.')
    X = frame.loc[:, list(FEATURE_COLUMNS)]
    y = frame[TARGET_COLUMN].to_numpy()
    cp, cq = _model_outputs(champion, X)
    npred, nq = _model_outputs(candidate, X)
    masks = {'global': np.ones(len(y), dtype=bool), 'true_injected_slice': true_mask.to_numpy(),
             'true_slice_complement': ~true_mask.to_numpy(),
             'diagnosed_slice_secondary': diagnosed_mask.to_numpy(),
             'diagnosed_complement_secondary': ~diagnosed_mask.to_numpy()}
    comparisons = {}
    for name, mask in masks.items():
        before, after = _metrics(y[mask], cp[mask], cq[mask]), _metrics(y[mask], npred[mask], nq[mask])
        comparisons[name] = {'rows': int(mask.sum()),
            'target_counts': {str(k): int((y[mask] == k).sum()) for k in (0, 1)},
            'champion': before, 'candidate': after,
            'candidate_minus_champion': {k: None if before[k] is None or after[k] is None else after[k]-before[k]
                                        for k in ('accuracy', 'macro_f1', 'auc')}}
    targets = {key: comparisons[population]['candidate_minus_champion']['accuracy'] for key, population in
               [('future_global_margin', 'global'), ('future_slice_margin', 'true_injected_slice'),
                ('future_nonaffected_margin', 'true_slice_complement')]}
    predictions = pd.DataFrame({'source_row_id': frame.index, 'target': y,
        'champion_prediction': cp, 'candidate_prediction': npred,
        'champion_positive_probability': cq, 'candidate_positive_probability': nq})
    return comparisons, targets, predictions
