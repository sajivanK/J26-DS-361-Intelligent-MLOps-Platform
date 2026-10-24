"""A batch cannot silently add groups or consume calibration/test groups."""
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from run_fixed_training_batch import planned_group_ids


def manifest():
    return {'groups': [
        {'group_id': 'train-a', 'split_role': 'forecast_train'},
        {'group_id': 'train-b', 'split_role': 'forecast_train'},
        {'group_id': 'conformal', 'split_role': 'conformal_calibration'},
        {'group_id': 'policy', 'split_role': 'policy_calibration'},
        {'group_id': 'outer', 'split_role': 'outer_test'}]}


def test_requested_order_is_preserved_without_extra_groups():
    assert planned_group_ids(manifest(), ['train-b', 'train-a']) == ['train-b', 'train-a']


def test_duplicate_requested_group_is_rejected():
    with pytest.raises(ValueError, match='unique'):
        planned_group_ids(manifest(), ['train-a', 'train-a'])


@pytest.mark.parametrize('group', ['conformal', 'policy', 'outer'])
def test_calibration_and_test_groups_cannot_enter_training_batch(group):
    with pytest.raises(ValueError, match='forecasting-training'):
        planned_group_ids(manifest(), ['train-a', group])


def test_unplanned_group_is_rejected():
    with pytest.raises(ValueError, match='missing'):
        planned_group_ids(manifest(), ['unknown'])
