"""N1 research models for the audited 75-feature scenario representation.

This module is separate from earlier engineering fixtures. It does not perform
conformal calibration, infer privacy approval, or authorize model deployment.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

TARGETS = ('future_global_margin', 'future_slice_margin', 'future_nonaffected_margin')
QUANTILES = (.05, .5, .95)


@dataclass(frozen=True)
class BenchmarkN1Config:
    n_estimators: int = 100
    num_leaves: int = 7
    max_depth: int = 3
    min_child_samples: int = 10
    learning_rate: float = .05
    reg_lambda: float = 1.
    linear_alpha: float = .001
    random_state: int = 42
    n_jobs: int = 1
    cv_splits: int = 5

    def __post_init__(self):
        for name in ('n_estimators', 'num_leaves', 'max_depth', 'min_child_samples', 'n_jobs', 'cv_splits'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if type(self.random_state) is not int or not 0 <= self.random_state <= 2**32-1:
            raise ValueError('random_state must be an unsigned integer.')
        if self.cv_splits < 2 or self.num_leaves < 2:
            raise ValueError('At least two CV splits and two leaves are required.')
        for name in ('learning_rate', 'reg_lambda', 'linear_alpha'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        if self.learning_rate == 0 or self.linear_alpha == 0:
            raise ValueError('Positive learning rate and baseline regularization are required.')


def group_weights(groups):
    s = pd.Series(list(groups), dtype='object')
    if s.empty or not s.map(lambda g: isinstance(g, str) and bool(g)).all():
        raise ValueError('Nonempty string group IDs are required.')
    counts = s.value_counts()
    weights = np.array([1. / counts[g] for g in s], dtype=float)
    return weights / weights.mean()


def validate_tables(features, targets, index):
    if features.empty or not features.columns.is_unique or not features.index.is_unique:
        raise ValueError('Nonempty features with unique columns and scenario IDs are required.')
    if set(TARGETS) & set(features.columns):
        raise ValueError('Future targets cannot be input features.')
    if not all(pd.api.types.is_numeric_dtype(features[c]) for c in features) or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError('Features must be numeric and finite.')
    if list(targets.columns) != list(TARGETS) or not features.index.equals(targets.index):
        raise ValueError('Training targets must match feature scenario IDs and target order.')
    y = targets.to_numpy(dtype=float)
    if not np.isfinite(y).all() or (np.abs(y) > 1).any():
        raise ValueError('Future training margins must be finite fractions in [-1, 1].')
    if index['scenario_id'].tolist() != features.index.tolist() or not index['scenario_id'].is_unique:
        raise ValueError('Scenario index order or uniqueness differs.')
    if not index['split_role'].eq('forecast_train').all():
        raise ValueError('Only forecast_train rows may fit N1.')
    if not index['group_id'].map(lambda g: isinstance(g, str) and bool(g)).all():
        raise ValueError('Valid training group IDs are required.')


def grouped_folds(index, n_splits):
    if not index['split_role'].eq('forecast_train').all() or not index['scenario_id'].is_unique:
        raise ValueError('Grouped validation requires unique forecast-training scenarios.')
    if index.group_id.nunique() < n_splits:
        raise ValueError('Insufficient nonempty training groups for the specified CV.')
    for train, valid in GroupKFold(n_splits=n_splits).split(index, groups=index.group_id):
        if set(index.iloc[train].group_id) & set(index.iloc[valid].group_id):
            raise ValueError('A group crossed the internal validation boundary.')
        yield train, valid


class BenchmarkQuantileForecaster:
    def __init__(self, method, config=None):
        if method not in {'lightgbm', 'linear'}:
            raise ValueError('Unknown forecasting method.')
        self.method = method
        self.config = config or BenchmarkN1Config()
        self.models = {}

    def fit(self, features, targets, index):
        if self.models:
            raise ValueError('Do not refit an existing model version.')
        validate_tables(features, targets, index)
        self.input_columns = features.columns.tolist()
        self.selected_columns = features.columns[features.nunique(dropna=False) > 1].tolist()
        self.constant_columns = [c for c in self.input_columns if c not in self.selected_columns]
        if not self.selected_columns:
            raise ValueError('No varying decision-time features in this training fold.')
        weights = group_weights(index.group_id)
        x = features.loc[:, self.selected_columns]
        self.scaler = None
        if self.method == 'linear':
            self.scaler = StandardScaler().fit(x, sample_weight=weights)
            x = self.scaler.transform(x)
        models = {}
        for target in TARGETS:
            for q in QUANTILES:
                if self.method == 'lightgbm':
                    from lightgbm import LGBMRegressor
                    model = LGBMRegressor(objective='quantile', alpha=q,
                        n_estimators=self.config.n_estimators, num_leaves=self.config.num_leaves,
                        max_depth=self.config.max_depth, min_child_samples=self.config.min_child_samples,
                        learning_rate=self.config.learning_rate, reg_lambda=self.config.reg_lambda,
                        random_state=self.config.random_state, n_jobs=self.config.n_jobs,
                        deterministic=True, force_col_wise=True, verbosity=-1)
                else:
                    model = QuantileRegressor(quantile=q, alpha=self.config.linear_alpha, solver='highs')
                model.fit(x, targets[target].to_numpy(), sample_weight=weights)
                models[(target, q)] = model
        self.models = models
        self.training_scenario_ids = features.index.tolist()
        self.training_group_ids = sorted(index.group_id.unique().tolist())
        return self

    def predict(self, features):
        if not self.models:
            raise ValueError('Forecaster is not fitted.')
        if features.columns.tolist() != self.input_columns:
            raise ValueError('Prediction feature columns/order differ; targets and IDs must be separate.')
        if not all(pd.api.types.is_numeric_dtype(features[c]) for c in features) or not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError('Prediction features must be numeric and finite.')
        x = features.loc[:, self.selected_columns]
        if self.scaler is not None:
            x = self.scaler.transform(x)
        out = pd.DataFrame(index=features.index)
        for target in TARGETS:
            raw = np.column_stack([self.models[(target, q)].predict(x) for q in QUANTILES])
            if not np.isfinite(raw).all():
                raise ValueError('Nonfinite forecast.')
            ordered = np.clip(np.sort(raw, axis=1), -1., 1.)
            for k, label in enumerate(('lower', 'median', 'upper')):
                out[target + '__raw_' + label] = raw[:, k]
                out[target + '__' + label] = ordered[:, k]
            out[target + '__crossing_corrected'] = (np.diff(raw, axis=1) < 0).any(axis=1)
        return out


def forecast_metrics(targets, forecasts, groups):
    if not targets.index.equals(forecasts.index) or len(groups) != len(targets):
        raise ValueError('Metric targets, forecasts and groups must align.')
    weights = group_weights(groups)
    result = {}
    for target in TARGETS:
        y = targets[target].to_numpy(dtype=float)
        point = forecasts[target + '__median'].to_numpy(dtype=float)
        report = {}
        for name, w in [('scenario_weighted', np.ones(len(y))), ('equal_group_weighted', weights)]:
            mse = float(np.average((y-point)**2, weights=w))
            denominator = float(np.sum(w * (y-np.average(y, weights=w))**2))
            report[name] = {'mae': float(np.average(np.abs(y-point), weights=w)),
                'rmse': float(np.sqrt(mse)),
                'r2': None if denominator == 0 else float(1.-np.sum(w*(y-point)**2)/denominator),
                'pinball': {}, 'raw_pinball': {}}
            for q, label in zip(QUANTILES, ('lower', 'median', 'upper')):
                for prefix, key in [('', 'pinball'), ('raw_', 'raw_pinball')]:
                    residual = y-forecasts[target + '__' + prefix + label].to_numpy(dtype=float)
                    report[name][key][str(q)] = float(np.average(np.maximum(q*residual, (q-1)*residual), weights=w))
        report['raw_quantile_crossing_rows'] = int(forecasts[target + '__crossing_corrected'].sum())
        result[target] = report
    return result
