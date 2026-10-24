import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'scripts'))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

helpers = load('c4_training_group_test', root / 'scripts/_training_group.py')
builder = load('c4_group_manifest_test', root / 'scripts/_benchmark_manifest.py')

@pytest.fixture
def manifest():
    return builder.build_manifest(json.loads((root / 'configs/benchmark_protocol.json').read_text()))


def test_group_has_all_37_frozen_conditions(manifest):
    rows = helpers.training_group_cases(manifest)
    assert len(rows) == 37
    assert len({r['group_id'] for r in rows}) == 1
    assert sum(r['is_no_drift_control'] for r in rows) == 1
    assert sum(r['repair_condition'] == 'targeted_repair' for r in rows) == 19
    assert sum(r['repair_condition'] == 'label_corrupted_repair' for r in rows) == 18


def test_targeted_base_precedes_corruption_and_shares_generator_seed(manifest):
    rows = helpers.training_group_cases(manifest)
    bases = {}
    for row in rows:
        if row['repair_condition'] == 'targeted_repair':
            bases[row['shared_base_package_id']] = row
        else:
            base = bases[row['shared_base_package_id']]
            assert base['generator_seed'] == row['generator_seed']
            assert base['drift_seeds'] == row['drift_seeds']


def test_calibration_group_is_blocked(manifest):
    group = next(g['group_id'] for g in manifest['groups'] if g['split_role'] == 'conformal_calibration')
    with pytest.raises(ValueError, match='forecasting-training group'):
        helpers.training_group_cases(manifest, group)


def test_corruption_is_fixed_reproducible_and_changes_labels_only():
    base = pd.DataFrame({'age': np.linspace(-2, 2, 500), 'target': np.arange(500) % 2})
    first, mask = helpers.corrupt_labels(base, fraction=.2, seed=123)
    second, _ = helpers.corrupt_labels(base, fraction=.2, seed=123)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    pd.testing.assert_series_equal(first['age'], base['age'], check_exact=True)
    assert int(mask['label_flipped'].sum()) == 100
    assert int((first['target'] != base['target']).sum()) == 100
    assert (base['target'].to_numpy() == np.arange(500) % 2).all()
