import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('c4_meta_features_test', scripts / '_scenario_meta.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.fixture
def inputs():
    upstream = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/valid_upstream_input.json').read_text())
    metrics = {'accuracy': .8, 'macro_f1': .7, 'auc': .9}
    evidence = {'report': {'comparisons': {p: {'rows': 10,
        'champion': metrics.copy(), 'candidate': metrics.copy(),
        'candidate_minus_champion': {'accuracy': 0., 'macro_f1': 0., 'auc': 0.}}
        for p in ('global', 'affected_slice', 'non_affected')}}}
    return upstream, evidence


def test_future_outcomes_cannot_enter_feature_row(inputs):
    upstream, evidence = inputs
    baseline = helpers.feature_row(upstream, evidence)
    upstream['future_global_margin'] = -.99
    evidence['future_targets'] = {'future_slice_margin': .99}
    assert helpers.feature_row(upstream, evidence) == baseline
    assert not set(helpers.TARGET_COLUMNS) & set(baseline)


def test_scenario_factors_and_identifiers_are_not_features(inputs):
    upstream, evidence = inputs
    baseline = helpers.feature_row(upstream, evidence)
    evidence.update(seed=42, mechanism='interaction', true_slice_support=.5, group_id='g', scenario_id='s')
    assert helpers.feature_row(upstream, evidence) == baseline


def test_negative_future_targets_are_retained():
    values = dict(zip(helpers.TARGET_COLUMNS, [-.0086, -.0025, -.0142]))
    result = helpers.validated_targets({'evaluation_role': 'sealed_future', 'primary_slice_basis': 'true_injected_slice',
        'target_units': 'accuracy_margin_fraction', 'primary_targets': values})
    assert result == values


def test_missing_future_target_is_rejected():
    with pytest.raises(ValueError, match='Exactly three'):
        helpers.validated_targets({'evaluation_role': 'sealed_future', 'primary_slice_basis': 'true_injected_slice',
            'target_units': 'accuracy_margin_fraction', 'primary_targets': {'future_global_margin': 0.}})


def test_undefined_auc_has_explicit_missing_indicator(inputs):
    upstream, evidence = inputs
    evidence['report']['comparisons']['affected_slice']['champion']['auc'] = None
    row = helpers.feature_row(upstream, evidence)
    assert row['c4_affected_slice_champion_auc'] == 0.
    assert row['c4_affected_slice_champion_auc_missing'] == 1.
    assert row['c4_global_champion_auc_missing'] == 0.


def test_nonfinite_decision_feature_is_rejected(inputs):
    upstream, evidence = inputs
    upstream['c1_evidence']['predicted_harm'] = float('nan')
    with pytest.raises(ValueError, match='finite and numeric'):
        helpers.feature_row(upstream, evidence)
