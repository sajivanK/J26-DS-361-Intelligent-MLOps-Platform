"""Leakage boundaries and model persistence for actual scenario forecasting."""
import io

import joblib
import numpy as np
import pandas as pd
import pytest

from c4.forecasting.benchmark_n1 import (BenchmarkN1Config, BenchmarkQuantileForecaster,
    TARGETS, forecast_metrics, group_weights, grouped_folds, validate_tables)


def tables():
    ids = [f's{i}' for i in range(24)]
    x = pd.DataFrame({'evidence': np.linspace(-1, 1, 24), 'constant': 1.}, index=ids)
    y = pd.DataFrame({t: .02 * x.evidence + i*.001 for i, t in enumerate(TARGETS)}, index=ids)
    index = pd.DataFrame({'scenario_id': ids, 'group_id': [f'g{i//4}' for i in range(24)], 'split_role': 'forecast_train'})
    return x, y, index


def test_group_weighting_does_not_reward_more_related_variants():
    groups = ['a', 'a', 'a', 'b']
    weights = group_weights(groups)
    assert weights[:3].sum() == pytest.approx(weights[3])


def test_internal_validation_never_splits_related_scenarios():
    _, _, index = tables()
    seen = []
    for train, valid in grouped_folds(index, 3):
        assert not set(index.iloc[train].group_id) & set(index.iloc[valid].group_id)
        seen.extend(valid.tolist())
    assert sorted(seen) == list(range(len(index)))


@pytest.mark.parametrize('role', ['conformal_calibration', 'policy_calibration', 'outer_test'])
def test_other_folds_cannot_fit_forecasting_models(role):
    x, y, index = tables()
    index.loc[0, 'split_role'] = role
    with pytest.raises(ValueError, match='forecast_train'):
        BenchmarkQuantileForecaster('linear').fit(x, y, index)


def test_future_target_cannot_become_input_feature():
    x, y, index = tables()
    x[TARGETS[0]] = y[TARGETS[0]]
    with pytest.raises(ValueError, match='Future targets'):
        validate_tables(x, y, index)


def test_duplicate_scenario_id_is_rejected():
    x, y, index = tables()
    index.loc[1, 'scenario_id'] = index.loc[0, 'scenario_id']
    with pytest.raises(ValueError, match='uniqueness'):
        validate_tables(x, y, index)


def test_nonfinite_training_target_is_rejected():
    x, y, index = tables()
    y.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite fractions'):
        validate_tables(x, y, index)


def test_validation_variation_cannot_change_training_feature_filter():
    x, y, index = tables()
    x['validation_only'] = [0.]*20 + [100.]*4
    model = BenchmarkQuantileForecaster('linear').fit(x.iloc[:20], y.iloc[:20], index.iloc[:20])
    assert 'validation_only' in model.constant_columns
    original = model.predict(x.iloc[20:])
    changed = x.iloc[20:].copy()
    changed['validation_only'] = -500.
    pd.testing.assert_frame_equal(original, model.predict(changed))


@pytest.mark.parametrize('method', ['linear', 'lightgbm'])
def test_actual_models_reload_with_identical_predictions(method):
    x, y, index = tables()
    model = BenchmarkQuantileForecaster(method, BenchmarkN1Config(n_estimators=5, min_child_samples=2)).fit(x, y, index)
    expected = model.predict(x)
    memory = io.BytesIO()
    joblib.dump(model, memory)
    memory.seek(0)
    restored = joblib.load(memory)
    pd.testing.assert_frame_equal(expected, restored.predict(x), check_exact=True)
    for target in TARGETS:
        assert expected[target+'__lower'].le(expected[target+'__median']).all()
        assert expected[target+'__median'].le(expected[target+'__upper']).all()
    metrics = forecast_metrics(y, expected, index.group_id)
    assert all(np.isfinite(metrics[t]['equal_group_weighted']['mae']) for t in TARGETS)


def test_prediction_schema_rejects_accidental_target_column():
    x, y, index = tables()
    model = BenchmarkQuantileForecaster('linear').fit(x, y, index)
    contaminated = x.assign(future_global_margin=y[TARGETS[0]])
    with pytest.raises(ValueError, match='columns/order'):
        model.predict(contaminated)
