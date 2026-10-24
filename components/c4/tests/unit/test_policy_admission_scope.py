"""Ensure policy admission cannot cross folds or alter frozen N1."""
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
    path = Path(__file__).resolve().parents[2] / 'scripts/_policy_group.py'
    spec = importlib.util.spec_from_file_location('policy_scope_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'_common': common, '_training_group': training}):
        spec.loader.exec_module(module)
    return module


def manifest():
    groups = [{'group_id': f'cal-{i:02d}', 'split_role': 'policy_calibration'} for i in range(12)]
    groups += [{'group_id': role, 'split_role': role} for role in ('forecast_train','conformal_calibration','outer_test')]
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


def test_three_batches_exhaust_exactly_the_frozen_twelve_groups():
    h, m = helper(), manifest()
    batches = [h.batch_groups(m, n) for n in range(1,4)]
    flat = [group for batch in batches for group in batch]
    assert all(len(b) == 4 for b in batches)
    assert len(flat) == len(set(flat)) == 12
    assert flat == [g['group_id'] for g in m['groups'] if g['split_role']=='policy_calibration']


@pytest.mark.parametrize('role', ['forecast_train','conformal_calibration','outer_test'])
def test_nonpolicy_group_is_rejected(role):
    with pytest.raises(ValueError, match='Only a frozen policy'):
        helper().policy_group_cases(manifest(), role)


def test_group_fold_in_manifest_cannot_disagree_with_scenario_fold():
    m = manifest()
    m['scenarios'][0]['split_role'] = 'outer_test'
    with pytest.raises(ValueError, match='37 policy cases'):
        helper().policy_group_cases(m, 'cal-00')


def test_duplicate_scenario_ids_are_rejected():
    m = manifest()
    m['scenarios'][1]['scenario_id'] = m['scenarios'][0]['scenario_id']
    with pytest.raises(ValueError, match='Duplicate scenario'):
        helper().policy_group_cases(m, 'cal-00')


def test_label_corruption_follows_its_exact_targeted_base():
    rows = helper().policy_group_cases(manifest(), 'cal-00')
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
        helper().policy_group_cases(m, 'cal-00')


def test_duplicate_group_assignment_is_rejected():
    m = manifest()
    m['groups'].append(copy.deepcopy(m['groups'][0]))
    with pytest.raises(ValueError, match='Duplicate group'):
        helper().policy_groups(m)


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


def test_changed_n2_checkpoint_blocks_policy_admission():
    h = helper()
    h.checked_json = lambda *args: ({}, 'wrong')
    with pytest.raises(ValueError, match='N2 checkpoint'):
        h.verify_n2_n3(Path('/unused'))


def test_changed_n3_checkpoint_blocks_policy_admission():
    h = helper()
    n2 = {'informative_95_percent_intervals': False, 'release_authorized': False, 'source_sha256': {}}
    n2_hash = 'd40e55f7934934cf8708b4b77cb7b4a3205f66b1d30540daa31846e682cacc80'
    h.checked_json = lambda directory, *args: (n2, n2_hash) if directory.name=='n2-joint-v1' else ({},'wrong')
    with pytest.raises(ValueError, match='N3 checkpoint'):
        h.verify_n2_n3(Path('/unused'))


def test_policy_guard_never_reads_calibration_outcome_tables():
    h = helper()
    n2_hash = 'd40e55f7934934cf8708b4b77cb7b4a3205f66b1d30540daa31846e682cacc80'
    n3_hash = 'fa9a8eb674c0fc387e854c71316b57863f81bd79ab6cec9f935b926aa8c60c3a'
    n2 = {'informative_95_percent_intervals': False, 'release_authorized': False, 'source_sha256': {}, 'n1_binding': {}}
    n3 = {'n2_metadata_sha256': n2_hash, 'n1_binding': {}, 'release_authorized': False,
          'policy_calibrated': False, 'source_sha256': {}, 'specification_sha256': 'spec'}
    h.checked_json = lambda directory, *args: (n2, n2_hash) if directory.name=='n2-joint-v1' else (n3,n3_hash)
    reads = []
    h.checked_bytes = lambda path, expected: reads.append(path.name)
    h.verify_n2_n3(Path('/unused'))
    assert reads == ['specification.json']
