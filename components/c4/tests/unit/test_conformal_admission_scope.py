"""Ensure conformal admission cannot cross folds or alter frozen N1."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


def helper():
    common = ModuleType('_common')
    common.C4_ROOT = Path('/unused')
    common.require = lambda condition, message: None if condition else (_ for _ in ()).throw(ValueError(message))
    common.checked_bytes = lambda *args: b''
    common.checked_json = lambda *args: ({}, '')
    training = ModuleType('_training_group')
    training.corrupt_labels = lambda *args, **kwargs: None
    path = Path(__file__).resolve().parents[2] / 'scripts/_conformal_group.py'
    spec = importlib.util.spec_from_file_location('conformal_scope_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'_common': common, '_training_group': training}):
        spec.loader.exec_module(module)
    return module


def manifest():
    groups = [{'group_id': f'cal-{i:02d}', 'split_role': 'conformal_calibration'} for i in range(20)]
    groups += [{'group_id': role, 'split_role': role} for role in ('forecast_train','policy_calibration','outer_test')]
    rows = []
    for group in groups:
        gid = group['group_id']
        for m in ('marginal_multi','interaction','combined'):
            for strength in (.2,.5,.8):
                for d in ('exact_slice','broadened_slice'):
                    for repair in ('targeted_repair','label_corrupted_repair'):
                        base = f'{gid}-{m}-{strength}-{d}'
                        rows.append({'scenario_id': base+'-'+repair, 'group_id': gid, 'split_role': group['split_role'],
                            'mechanism': m, 'magnitude': strength, 'diagnosis_condition': d, 'repair_condition': repair,
                            'shared_base_package_id': base, 'is_no_drift_control': False})
        rows.append({'scenario_id': gid+'-control', 'group_id': gid, 'split_role': group['split_role'],
            'mechanism': 'none', 'magnitude': 0., 'diagnosis_condition': 'exact_slice', 'repair_condition': 'targeted_repair',
            'shared_base_package_id': gid+'-control', 'is_no_drift_control': True})
    return {'groups': groups, 'scenarios': rows}


def test_five_batches_exhaust_exactly_the_frozen_twenty_groups():
    h, m = helper(), manifest()
    batches = [h.batch_groups(m, n) for n in range(1,6)]
    flat = [group for batch in batches for group in batch]
    assert all(len(b) == 4 for b in batches)
    assert len(flat) == len(set(flat)) == 20
    assert flat == [g['group_id'] for g in m['groups'] if g['split_role']=='conformal_calibration']


@pytest.mark.parametrize('role', ['forecast_train','policy_calibration','outer_test'])
def test_nonconformal_group_is_rejected(role):
    with pytest.raises(ValueError, match='Only a frozen conformal'):
        helper().conformal_group_cases(manifest(), role)


def test_group_fold_in_manifest_cannot_disagree_with_scenario_fold():
    m = manifest()
    m['scenarios'][0]['split_role'] = 'outer_test'
    with pytest.raises(ValueError, match='37 conformal cases'):
        helper().conformal_group_cases(m, 'cal-00')


def test_duplicate_scenario_ids_are_rejected():
    m = manifest()
    m['scenarios'][1]['scenario_id'] = m['scenarios'][0]['scenario_id']
    with pytest.raises(ValueError, match='Duplicate scenario'):
        helper().conformal_group_cases(m, 'cal-00')


def test_label_corruption_follows_its_exact_targeted_base():
    rows = helper().conformal_group_cases(manifest(), 'cal-00')
    seen = set()
    for row in rows:
        if row['repair_condition']=='targeted_repair':
            seen.add(row['shared_base_package_id'])
        else:
            assert row['shared_base_package_id'] in seen


def test_changed_factorial_condition_is_rejected():
    m = manifest()
    m['scenarios'][0]['magnitude'] = .3
    with pytest.raises(ValueError, match='factorial'):
        helper().conformal_group_cases(m, 'cal-00')


def test_duplicate_group_assignment_is_rejected():
    m = manifest()
    m['groups'].append(copy.deepcopy(m['groups'][0]))
    with pytest.raises(ValueError, match='Duplicate group'):
        helper().conformal_groups(m)


def test_n1_guard_rejects_a_different_model_checkpoint():
    h = helper()
    h.checked_json = lambda directory, stem='metadata': ({}, 'wrong-digest')
    with pytest.raises(ValueError, match='N1 specification or fitted-model'):
        h.verify_n1(Path('/unused'))


def test_n1_guard_checks_saved_model_bytes():
    h = helper()
    binding = {'training_metadata_sha256': 'training', 'code_sha256': {'src/c4/forecasting/benchmark_n1.py':'code'},
               'protocol_freeze_sha256': 'protocol'}
    spec = {'binding': binding}
    run = {'specification_sha256': h.N1_SPEC_HASH, 'n2_calibrated': False, 'outer_test_evaluated': False,
           'release_authorized': False, 'internal_grouped_validation_complete': True,
           'model_reload_prediction_equality': True, 'training_metadata_sha256': 'training',
           'files': {'lightgbm.joblib':'model-hash'}}
    h.checked_json = lambda directory, stem='metadata': (spec,h.N1_SPEC_HASH) if stem=='specification' else (run,h.N1_RUN_HASH)
    def read(path, expected):
        if path.name=='lightgbm.joblib':
            raise ValueError('Checksum mismatch')
        return b''
    h.checked_bytes = read
    with pytest.raises(ValueError, match='Checksum mismatch'):
        h.verify_n1(Path('/unused'))
