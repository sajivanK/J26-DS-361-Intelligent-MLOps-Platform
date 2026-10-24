"""Verify the development metric independently of Drive artifacts."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[2] / 'scripts' / '_privacy_distance.py'
spec = importlib.util.spec_from_file_location('c4_development_privacy_distance', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_known_mixed_distance_and_target_exclusion():
    fit = pd.DataFrame({'age': [0., 10.], 'sex': [0, 1], 'target': [0, 1]})
    query = pd.DataFrame({'age': [5., 0.], 'sex': [0, 0], 'target': [1, 1]})
    metric = module.fit_distance_spec(fit, {'age': 'continuous', 'sex': 'categorical'})
    np.testing.assert_allclose(module.nearest_distances(query, fit, metric), [.25, 0.])


def test_category_code_magnitude_does_not_change_distance():
    fit = pd.DataFrame({'category': [0, 100]})
    query = pd.DataFrame({'category': [1, 100]})
    metric = module.fit_distance_spec(fit, {'category': 'categorical'})
    np.testing.assert_array_equal(module.nearest_distances(query, fit, metric), [1., 0.])


def test_constant_continuous_and_outside_range_are_bounded():
    fit = pd.DataFrame({'x': [2., 2.], 'y': [0., 10.]})
    query = pd.DataFrame({'x': [2., 3.], 'y': [100., 100.]})
    metric = module.fit_distance_spec(fit, {'x': 'continuous', 'y': 'continuous'})
    np.testing.assert_array_equal(module.nearest_distances(query, fit, metric), [.5, 1.])


def test_chunking_preserves_nearest_results():
    fit = pd.DataFrame({'x': [0., 1., 3.]})
    query = pd.DataFrame({'x': [0., .5, 2., 5.]})
    metric = module.fit_distance_spec(fit, {'x': 'continuous'})
    np.testing.assert_array_equal(module.nearest_distances(query, fit, metric, 1),
                                  module.nearest_distances(query, fit, metric, 100))


def test_nonfinite_features_are_rejected():
    with pytest.raises(ValueError, match='finite'):
        module.fit_distance_spec(pd.DataFrame({'x': [np.nan]}), {'x': 'continuous'})


def test_target_cannot_be_a_distance_feature():
    with pytest.raises(ValueError, match='Targets'):
        module.fit_distance_spec(pd.DataFrame({'target': [0, 1]}), {'target': 'categorical'})


def test_summary_counts_exact_matches():
    result = module.distance_summary([0., .2, .4])
    assert result['exact_feature_match_rows'] == 1
    assert result['median'] == .2
    assert result['rows'] == 3
