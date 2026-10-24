import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c4.data import FEATURE_COLUMNS
from c4.experiments.upstream_adapters import adapt_evidence_pair, require_no_mock_sources


@pytest.fixture
def inputs():
    payload = json.loads((Path(__file__).resolve().parents[1] / 'fixtures' /
                          'valid_upstream_input.json').read_text())
    clean = pd.DataFrame({feature: np.zeros(4) for feature in FEATURE_COLUMNS}, index=[10, 11, 12, 13])
    clean['age'] = [-2., -1., 1., 2.]
    clean['target'] = [0, 1, 0, 1]
    payload['c1_evidence']['batch_size'] = 4
    payload['c2_diagnosis']['affected_slice_rule'] = 'age <= 0'
    payload['c2_diagnosis']['slice_support'] = .5
    return payload, clean


def adapt(inputs, **changes):
    payload, clean = inputs
    options = {'c1_origin': 'mock', 'c2_origin': 'mock', **changes}
    return adapt_evidence_pair(payload['c1_evidence'], payload['c2_diagnosis'], clean, **options)


def test_mock_origins_and_clean_membership_are_preserved(inputs):
    result = adapt(inputs)
    assert result['contains_mock_evidence'] is True
    assert result['diagnosed_membership'] == [True, True, False, False]
    assert result['source_row_ids'] == [10, 11, 12, 13]
    assert result['candidate_training_authorized'] is False
    assert result['release_decision'] is None


def test_mock_evidence_is_blocked_from_real_evidence_path(inputs):
    with pytest.raises(ValueError, match='Mock'):
        require_no_mock_sources(adapt(inputs))


def test_fake_engine_cannot_be_declared_real(inputs):
    inputs[0]['c1_evidence']['engine_version'] = 'fake-1.1'
    with pytest.raises(ValueError, match='Fake'):
        adapt(inputs, c1_origin='real')


def test_batch_count_mismatch_is_rejected(inputs):
    inputs[0]['c1_evidence']['batch_size'] = 5
    with pytest.raises(ValueError, match='batch_size'):
        adapt(inputs)


def test_sealed_future_role_is_rejected(inputs):
    with pytest.raises(ValueError, match='Sealed-future'):
        adapt(inputs, role='sealed_future')


def test_nonfinite_payload_is_rejected(inputs):
    inputs[0]['c1_evidence']['harm_interval'] = [float('-inf'), float('inf')]
    with pytest.raises(ValueError, match='finite'):
        adapt(inputs)


def test_empty_complement_is_rejected(inputs):
    inputs[0]['c2_diagnosis']['affected_slice_rule'] = 'age <= 100'
    with pytest.raises(ValueError, match='Both diagnosed'):
        adapt(inputs)


def test_declared_support_is_reported_not_silently_rewritten(inputs):
    inputs[0]['c2_diagnosis']['slice_support'] = .4
    result = adapt(inputs)
    assert result['computed_slice_support'] == .5
    assert result['declared_slice_support'] == .4
    assert result['slice_support_difference'] == pytest.approx(-.1)
