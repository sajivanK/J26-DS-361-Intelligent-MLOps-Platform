"""Combining snapshots must not duplicate scenarios or mix research folds."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from audit_combined_training_snapshot import combine_tables, TARGET_COLUMNS


def tables(sid, group):
    f = pd.DataFrame(np.zeros((1, 75)), columns=[f'f{k}' for k in range(75)], index=[sid])
    t = pd.DataFrame([[-.01, 0., -.02]], columns=TARGET_COLUMNS, index=[sid])
    i = pd.DataFrame({'scenario_id': [sid], 'group_id': [group], 'split_role': ['forecast_train']})
    return f, t, i


def test_combined_tables_preserve_negative_targets_and_separate_ids():
    a, b = tables('a', 'group-a'), tables('b', 'group-b')
    f, t, i = combine_tables([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
    assert f.shape == (2, 75) and t.shape == (2, 3)
    assert t.iloc[0, 0] == -.01
    assert set(TARGET_COLUMNS).isdisjoint(f.columns)
    assert i.group_id.tolist() == ['group-a', 'group-b']


def test_duplicate_scenario_across_snapshots_is_rejected():
    a = tables('same', 'group-a')
    with pytest.raises(ValueError, match='Duplicate scenario'):
        combine_tables([a[0], a[0]], [a[1], a[1]], [a[2], a[2]])


def test_nontraining_snapshot_cannot_enter_combination():
    f, t, i = tables('a', 'group-a')
    i['split_role'] = 'outer_test'
    with pytest.raises(ValueError, match='Nontraining'):
        combine_tables([f], [t], [i])


def test_mismatched_feature_schema_is_rejected():
    a, b = tables('a', 'group-a'), tables('b', 'group-b')
    b[0].rename(columns={'f0': 'different'}, inplace=True)
    with pytest.raises(ValueError, match='schema differs'):
        combine_tables([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
