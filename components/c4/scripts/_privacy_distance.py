"""Explicit mixed-feature distance for development diagnostics.

Continuous distances use generator-fit ranges and are capped at one.
Categoricals use equality, never numerical code differences. Features have
equal weight. This descriptive metric is not a privacy guarantee.
"""
import numpy as np
import pandas as pd


def _values(frame, features):
    if frame.empty or not features or len(set(features)) != len(features):
        raise ValueError('Nonempty data and unique features are required.')
    if not set(features) <= set(frame.columns):
        raise ValueError('A distance feature is missing.')
    if 'target' in features or 'source_row_id' in features:
        raise ValueError('Targets and row IDs cannot be distance features.')
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in features):
        raise ValueError('Processed distance features must be numeric.')
    values = frame.loc[:, list(features)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Distance features must be finite.')
    return values


def fit_distance_spec(fit, feature_types):
    features = tuple(feature_types)
    values = _values(fit, features)
    if any(kind not in {'continuous', 'categorical'} for kind in feature_types.values()):
        raise ValueError('Unknown feature type.')
    return {
        'features': list(features),
        'feature_types': dict(feature_types),
        'continuous_ranges': {
            feature: float(np.ptp(values[:, index]))
            for index, feature in enumerate(features)
            if feature_types[feature] == 'continuous'
        },
        'weights': 'equal_per_feature',
        'continuous_rule': 'min(abs_difference / fit_range, 1)',
        'constant_continuous_rule': '0_if_equal_else_1',
        'categorical_rule': '0_if_equal_else_1',
        'scale_population': 'generator_fit_only',
        'target_included': False,
    }


def nearest_distances(query, reference, spec, chunk_size=128):
    """Return exact nearest distances, limiting the query block size."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('Chunk size must be a positive integer.')
    features = tuple(spec['features'])
    query_values = _values(query, features)
    reference_values = _values(reference, features)
    results = []
    for start in range(0, len(query_values), chunk_size):
        block = query_values[start:start + chunk_size]
        distances = np.zeros((len(block), len(reference_values)), dtype=float)
        for index, feature in enumerate(features):
            left = block[:, index, None]
            right = reference_values[None, :, index]
            if spec['feature_types'][feature] == 'categorical':
                contribution = (left != right).astype(float)
            else:
                scale = spec['continuous_ranges'][feature]
                if not np.isfinite(scale) or scale < 0:
                    raise ValueError('Invalid fitted distance range.')
                contribution = ((left != right).astype(float) if scale == 0
                                else np.minimum(np.abs(left - right) / scale, 1.0))
            distances += contribution
        results.append((distances / len(features)).min(axis=1))
    return np.concatenate(results)


def distance_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Distance summary requires a finite nonempty vector.')
    if ((values < 0) | (values > 1)).any():
        raise ValueError('Distance values must be between zero and one.')
    return {
        'rows': len(values), 'minimum': float(values.min()),
        'p01': float(np.quantile(values, 0.01)),
        'p05': float(np.quantile(values, 0.05)),
        'median': float(np.median(values)), 'mean': float(values.mean()),
        'maximum': float(values.max()),
        'exact_feature_match_rows': int((values == 0).sum()),
    }
