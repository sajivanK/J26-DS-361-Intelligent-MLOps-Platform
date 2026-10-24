import copy
import importlib.util
import json
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('c4_full_manifest_builder', root / 'scripts/_benchmark_manifest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope='module')
def protocol():
    return json.loads((root / 'configs/benchmark_protocol.json').read_text())


def test_planned_counts_and_controls(protocol):
    result = module.build_manifest(protocol)
    assert len(result['groups']) == 80
    assert len(result['scenarios']) == 2960
    assert sum(s['is_no_drift_control'] for s in result['scenarios']) == 80
    assert result['experiments_executed'] == 0


def test_exact_four_way_group_counts(protocol):
    result = module.build_manifest(protocol)
    counts = {role: sum(g['split_role'] == role for g in result['groups'])
              for role in protocol['scenario_split']['per_namespace']}
    assert counts == {'forecast_train': 32, 'conformal_calibration': 20,
                      'policy_calibration': 12, 'outer_test': 16}


def test_related_variants_never_cross_folds(protocol):
    result = module.build_manifest(protocol)
    for group in result['groups']:
        rows = [s for s in result['scenarios'] if s['group_id'] == group['group_id']]
        assert len(rows) == 37
        assert {s['split_role'] for s in rows} == {group['split_role']}


def test_manifest_repeats_exactly(protocol):
    assert module.build_manifest(protocol) == module.build_manifest(protocol)


def test_repair_variants_share_base_generator_but_remain_distinct(protocol):
    rows = module.build_manifest(protocol)['scenarios'][:2]
    assert rows[0]['shared_base_package_id'] == rows[1]['shared_base_package_id']
    assert rows[0]['generator_seed'] == rows[1]['generator_seed']
    assert rows[0]['scenario_id'] != rows[1]['scenario_id']
    assert rows[0]['label_flip_fraction'] != rows[1]['label_flip_fraction']


def test_invalid_split_budget_is_rejected(protocol):
    changed = copy.deepcopy(protocol)
    changed['scenario_split']['per_namespace']['outer_test'] = 1
    with pytest.raises(ValueError, match='sum'):
        module.build_manifest(changed)


def test_duplicate_namespaces_are_rejected(protocol):
    changed = copy.deepcopy(protocol)
    changed['replication']['seed_namespaces'] = [42, 42]
    with pytest.raises(ValueError, match='replication'):
        module.build_manifest(changed)


def test_future_seeds_are_distinct_from_decision_time_seeds(protocol):
    for row in module.build_manifest(protocol)['scenarios']:
        assert len(set(row['drift_seeds'].values())) == 3
