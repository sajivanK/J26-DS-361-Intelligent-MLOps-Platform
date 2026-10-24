import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from c4.data import FEATURE_COLUMNS
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import SliceRule

scripts = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('c4_future_helpers_test', scripts / '_future_research.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.fixture
def setup():
    frame = pd.DataFrame({f: np.arange(8, dtype=float) for f in FEATURE_COLUMNS}, index=np.arange(100, 108))
    frame['target'] = [0, 0, 0, 1, 1, 1, 1, 0]
    config = ScenarioConfig(mechanism='interaction', magnitude=.5,
        interaction_pairs=(('marital_status', 'relationship'),), seed=123)
    rule = SliceRule(name='true', feature='age', operator='le', value=3.)
    def inject(features, pairs, magnitude, rng):
        result = features.copy()
        result['relationship'] = rng.permutation(result['relationship'].to_numpy())
        return result
    return frame, config, rule, SimpleNamespace(inject_interaction_multi_pair=inject)


def test_future_transform_repeats_with_fixed_seed(setup):
    frame, config, rule, shared = setup
    first, _ = helpers.transform_interaction_future(frame, config=config, true_rule=rule, shared=shared)
    second, _ = helpers.transform_interaction_future(frame, config=config, true_rule=rule, shared=shared)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_future_transform_preserves_ids_labels_and_complement(setup):
    frame, config, rule, shared = setup
    original = frame.copy(deep=True)
    transformed, mask = helpers.transform_interaction_future(frame, config=config, true_rule=rule, shared=shared)
    assert transformed.index.equals(frame.index)
    pd.testing.assert_series_equal(transformed['target'], frame['target'], check_exact=True)
    pd.testing.assert_frame_equal(transformed.loc[~mask], frame.loc[~mask], check_exact=True)
    pd.testing.assert_frame_equal(frame, original, check_exact=True)


def test_future_rejects_unimplemented_mechanism(setup):
    frame, _, rule, shared = setup
    config = ScenarioConfig(mechanism='none', magnitude=0., seed=123)
    with pytest.raises(ValueError, match='interaction drift only'):
        helpers.transform_interaction_future(frame, config=config, true_rule=rule, shared=shared)


def test_future_primary_targets_use_true_not_diagnosed_slice(setup):
    frame, _, rule, _ = setup
    class Model:
        classes_ = np.array([0, 1])
        def __init__(self, label): self.label = label
        def predict(self, X): return np.full(len(X), self.label)
        def predict_proba(self, X): return np.tile([.8, .2] if self.label == 0 else [.2, .8], (len(X), 1))
    diagnosed = frame['age'] <= 5
    comparisons, targets, _ = helpers.future_comparisons(Model(0), Model(1), frame, rule.mask(frame), diagnosed)
    assert targets == {'future_global_margin': 0., 'future_slice_margin': -.5, 'future_nonaffected_margin': .5}
    assert comparisons['diagnosed_slice_secondary']['rows'] == 6
    assert comparisons['true_injected_slice']['rows'] == 4


def test_incomplete_decision_freeze_is_rejected():
    with pytest.raises(ValueError, match='Complete decision-evidence freeze'):
        helpers.verify_freeze_dependencies({'complete_decision_evidence_frozen': False})


def test_changed_frozen_dependency_is_rejected(tmp_path):
    path = tmp_path / 'evidence.json'
    path.write_text('changed')
    snapshot = {'complete_decision_evidence_frozen': True, 'future_evaluation_executed': False,
        'scope': 'one_admitted_forecast_training_scenario_under_simulated_upstream_evidence',
        'dependencies': {str(path): '0' * 64}}
    with pytest.raises(ValueError, match='Checksum mismatch'):
        helpers.verify_freeze_dependencies(snapshot)
