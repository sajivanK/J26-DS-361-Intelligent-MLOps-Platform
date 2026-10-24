"""Future dispatch and exact preview replay must match the frozen adapter."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from _all_mechanism_transforms import transform_future_role, canonical_release_frames
from c4.data import FEATURE_COLUMNS
from c4.scenarios.config import ScenarioConfig, CONTINUOUS_FEATURES
from c4.scenarios.slices import SliceRule
import c4.scenarios.adapter as adapter


class SharedFunctions:
    def inject_marginal_multi(self, frame, features, magnitude, feature_types, rng):
        result = frame.copy()
        for f in features:
            result[f] = result[f] + magnitude * rng.uniform(.1, .9)
        return result

    def inject_interaction_multi_pair(self, frame, pairs, magnitude, rng):
        result = frame.copy()
        for _, second in pairs:
            result[second] = rng.permutation(result[second].to_numpy())
        return result

    def inject_combined(self, frame, features, pairs, magnitude, types, rng):
        result = self.inject_marginal_multi(frame, features, magnitude, types, rng)
        return self.inject_interaction_multi_pair(result, pairs, magnitude, rng)


@pytest.fixture
def setup(monkeypatch):
    frame = pd.DataFrame({f: np.arange(4, dtype=float) for f in FEATURE_COLUMNS}, index=[10, 20, 30, 40])
    frame['age'] = [-2., -1., 1., 2.]
    frame['target'] = [0, 1, 0, 1]
    rule = SliceRule(name='test_slice', feature='age', operator='le', value=0.)
    types = {f: 'continuous' if f in CONTINUOUS_FEATURES else 'categorical' for f in FEATURE_COLUMNS}
    shared = SharedFunctions()
    monkeypatch.setattr(adapter, '_load_shared_functions', lambda: (shared, 'd'*64))
    monkeypatch.setattr(adapter, 'load_adult_feature_metadata',
                        lambda: SimpleNamespace(feature_types=types, source_sha256='f'*64))
    return frame, rule, types, shared


def configuration(mechanism):
    return ScenarioConfig(mechanism=mechanism, magnitude=0. if mechanism == 'none' else .5,
        features=('education_num', 'hours_per_week') if mechanism in {'marginal_multi', 'combined'} else (),
        interaction_pairs=(('marital_status', 'relationship'),) if mechanism in {'interaction', 'combined'} else (),
        seed=123)


@pytest.mark.parametrize('mechanism', ['none', 'marginal_multi', 'interaction', 'combined'])
def test_future_matches_adapter_and_preserves_roles(setup, mechanism):
    clean, rule, types, shared = setup
    original = clean.copy(deep=True)
    config = configuration(mechanism)
    expected = adapter.transform_role(clean, config=config, true_slice=rule, role='release_validation')
    actual, mask = transform_future_role(clean, config=config, true_rule=rule, shared=shared, feature_types=types)
    pd.testing.assert_frame_equal(actual, expected.frame, check_exact=True)
    pd.testing.assert_series_equal(mask, expected.true_membership, check_exact=True)
    pd.testing.assert_frame_equal(clean, original, check_exact=True)
    pd.testing.assert_frame_equal(actual.loc[~mask], clean.loc[~mask], check_exact=True)
    pd.testing.assert_series_equal(actual.target, clean.target, check_exact=True)
    again, _ = transform_future_role(clean, config=config, true_rule=rule, shared=shared, feature_types=types)
    pd.testing.assert_frame_equal(actual, again, check_exact=True)


def test_invalid_row_identity_is_rejected(setup):
    clean, rule, types, shared = setup
    clean.index = [10, 10, 30, 40]
    with pytest.raises(ValueError, match='unique'):
        transform_future_role(clean, config=configuration('interaction'), true_rule=rule, shared=shared, feature_types=types)


def test_shared_function_cannot_change_indices(setup, monkeypatch):
    clean, rule, types, shared = setup
    monkeypatch.setattr(shared, 'inject_interaction_multi_pair', lambda f, *a: f.reset_index(drop=True))
    with pytest.raises(ValueError, match='row identity'):
        transform_future_role(clean, config=configuration('interaction'), true_rule=rule, shared=shared, feature_types=types)


def test_nonfinite_shared_output_is_rejected(setup, monkeypatch):
    clean, rule, types, shared = setup
    def bad(frame, *args):
        result = frame.copy()
        result.iloc[0, 0] = np.inf
        return result
    monkeypatch.setattr(shared, 'inject_interaction_multi_pair', bad)
    with pytest.raises(ValueError, match='Nonfinite'):
        transform_future_role(clean, config=configuration('interaction'), true_rule=rule, shared=shared, feature_types=types)


def test_exact_replay_restores_canonical_nonperturbed_values(setup):
    clean, rule, _, _ = setup
    config = configuration('combined')
    expected = adapter.transform_role(clean, config=config, true_slice=rule, role='release_validation')
    canonical = clean.copy()
    canonical.loc[10, 'age'] = np.nextafter(canonical.loc[10, 'age'], -np.inf)
    a, b = canonical_release_frames(clean, canonical,
        clean.to_csv(index=True, lineterminator='\n').encode(),
        expected.frame.to_csv(index=True, lineterminator='\n').encode(), config=config, true_rule=rule)
    assert a.loc[10, 'age'] == b.loc[10, 'age'] == canonical.loc[10, 'age']
    assert b.loc[10, 'education_num'] == expected.frame.loc[10, 'education_num']


def test_changed_transformation_cannot_pass_replay(setup):
    clean, rule, _, _ = setup
    config = configuration('combined')
    transformed = adapter.transform_role(clean, config=config, true_slice=rule, role='release_validation').frame
    transformed.loc[10, 'education_num'] += .01
    with pytest.raises(ValueError, match='deterministic'):
        canonical_release_frames(clean, clean, clean.to_csv(index=True, lineterminator='\n').encode(),
            transformed.to_csv(index=True, lineterminator='\n').encode(), config=config, true_rule=rule)


def test_changed_clean_serialization_is_rejected(setup):
    clean, rule, _, _ = setup
    with pytest.raises(ValueError, match='original-role serialization'):
        canonical_release_frames(clean, clean, b'incorrect', b'incorrect',
                                  config=configuration('none'), true_rule=rule)
