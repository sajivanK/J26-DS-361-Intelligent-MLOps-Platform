"""Boundary tests for separately frozen outer admission; no TVAE fitting."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


def helper():
    common = ModuleType('_common')
    common.C4_ROOT = Path('/unused')
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    common.require = require
    common.sha = lambda b: hashlib.sha256(b).hexdigest()
    common.checked_bytes = lambda *args: b''
    common.checked_json = lambda *args: ({}, '')
    training = ModuleType('_training_group')
    training.corrupt_labels = lambda *args, **kwargs: None
    spec = importlib.util.spec_from_file_location('outer_admission_under_test', SCRIPTS / '_outer_group.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'_common': common, '_training_group': training}):
        spec.loader.exec_module(module)
    return module, common


def manifest():
    groups = [{'group_id': f'outer-{i:02d}', 'split_role': 'outer_test'} for i in range(16)]
    groups += [{'group_id': role, 'split_role': role}
               for role in ('forecast_train', 'conformal_calibration', 'policy_calibration')]
    rows = []
    for group in groups:
        gid = group['group_id']
        for mechanism in ('marginal_multi', 'interaction', 'combined'):
            for magnitude in (.2, .5, .8):
                for diagnosis in ('exact_slice', 'broadened_slice'):
                    for repair in ('targeted_repair', 'label_corrupted_repair'):
                        base = f'{gid}-{mechanism}-{magnitude}-{diagnosis}'
                        rows.append({'scenario_id': base + '-' + repair, 'group_id': gid,
                            'split_role': group['split_role'], 'mechanism': mechanism, 'magnitude': magnitude,
                            'diagnosis_condition': diagnosis, 'repair_condition': repair,
                            'shared_base_package_id': base, 'is_no_drift_control': False})
        rows.append({'scenario_id': gid + '-control', 'group_id': gid, 'split_role': group['split_role'],
                     'mechanism': 'none', 'magnitude': 0., 'diagnosis_condition': 'exact_slice',
                     'repair_condition': 'targeted_repair', 'shared_base_package_id': gid + '-control',
                     'is_no_drift_control': True})
    return {'groups': groups, 'scenarios': rows}


def test_four_batches_cover_all_sixteen_in_manifest_order_once():
    h, _ = helper()
    m = manifest()
    batches = [h.batch_groups(m, i) for i in range(1, 5)]
    flat = sum(batches, [])
    assert all(len(b) == 4 for b in batches)
    assert len(flat) == len(set(flat)) == 16
    assert flat == [g['group_id'] for g in m['groups'] if g['split_role'] == 'outer_test']
    assert sum(len(h.outer_group_cases(m, g)) for g in flat) == 592


@pytest.mark.parametrize('number', [0, 5, -1, True, 1.0])
def test_invalid_batch_rejected(number):
    h, _ = helper()
    with pytest.raises(ValueError, match='1 through 4'):
        h.batch_groups(manifest(), number)


@pytest.mark.parametrize('role', ['forecast_train', 'conformal_calibration', 'policy_calibration'])
def test_other_fold_rejected(role):
    h, _ = helper()
    with pytest.raises(ValueError, match='Only a frozen outer'):
        h.outer_group_cases(manifest(), role)


@pytest.mark.parametrize('change', ['duplicate_id', 'wrong_fold', 'wrong_magnitude', 'missing_case', 'bad_control'])
def test_invalid_case_inventory_rejected(change):
    h, _ = helper()
    m = manifest()
    if change == 'duplicate_id':
        m['scenarios'][1]['scenario_id'] = m['scenarios'][0]['scenario_id']
    elif change == 'wrong_fold':
        m['scenarios'][0]['split_role'] = 'forecast_train'
    elif change == 'wrong_magnitude':
        m['scenarios'][0]['magnitude'] = .3
    elif change == 'missing_case':
        m['scenarios'].pop(0)
    else:
        m['scenarios'][36]['mechanism'] = 'combined'
    with pytest.raises(ValueError):
        h.outer_group_cases(m, 'outer-00')


def test_duplicate_group_rejected():
    h, _ = helper()
    m = manifest()
    m['groups'].append(copy.deepcopy(m['groups'][0]))
    with pytest.raises(ValueError, match='Duplicate group'):
        h.outer_groups(m)


def test_corrupted_variant_always_follows_its_base():
    h, _ = helper()
    seen = set()
    for row in h.outer_group_cases(manifest(), 'outer-00'):
        if row['repair_condition'] == 'targeted_repair':
            seen.add(row['shared_base_package_id'])
        else:
            assert row['shared_base_package_id'] in seen


@pytest.mark.parametrize('stage', ['scoped_candidate_validation-v1', 'pre_release_freeze-v1', 'future_research-v1'])
def test_admission_guard_rejects_downstream_artifact(tmp_path, stage):
    h, _ = helper()
    m = manifest()
    run = h.runs_directory(tmp_path) / m['scenarios'][0]['scenario_id']
    (run / stage).mkdir(parents=True)
    with pytest.raises(ValueError, match='Admission-only scope violated'):
        h.assert_admission_only(tmp_path, m)


def test_completed_admission_directory_is_allowed_without_downstream_stages(tmp_path):
    h, _ = helper()
    m = manifest()
    (h.runs_directory(tmp_path) / m['scenarios'][0]['scenario_id']).mkdir(parents=True)
    h.assert_admission_only(tmp_path, m)


def test_initial_freeze_rejects_existing_outer_claim(tmp_path):
    h, common = helper()
    m = manifest()
    (h.runs_directory(tmp_path) / m['scenarios'][0]['scenario_id']).mkdir(parents=True)
    with patch.dict(sys.modules, {'_common': common}):
        with pytest.raises(ValueError, match='predates admission freeze'):
            h.assert_no_outer_artifacts(tmp_path, m)


def frozen_fixture(h):
    return {'purpose': 'frozen_outer_admission_only_protocol', 'admission_execution_authorized': True,
            'outer_candidate_execution_authorized': False, 'outer_future_execution_authorized': False,
            'production_release_authorized': False, 'source_sha256': {'code': 'same'},
            'binding': {'n1_binding': {'model': 'unchanged'}}}


def configure_guard(h):
    frozen = frozen_fixture(h)
    h.checked_json = lambda *args: (frozen, 'freeze-hash')
    h.source_hashes = lambda: {'code': 'same'}
    h.current_bindings = lambda root: {'n1_binding': {'model': 'unchanged'}}
    h.load_manifest = lambda root: (manifest(), 'original', 'manifest')
    return frozen


def test_verify_reuses_frozen_original_n1_binding(tmp_path):
    h, _ = helper()
    configure_guard(h)
    assert h.verify_n1(tmp_path) == {'model': 'unchanged'}


@pytest.mark.parametrize('change', ['model', 'code', 'candidate_authorized', 'future_authorized', 'release_authorized'])
def test_changed_frozen_binding_or_scope_rejected(tmp_path, change):
    h, _ = helper()
    frozen = configure_guard(h)
    if change == 'model':
        h.current_bindings = lambda root: {'n1_binding': {'model': 'changed'}}
    elif change == 'code':
        h.source_hashes = lambda: {'code': 'changed'}
    else:
        key = {'candidate_authorized': 'outer_candidate_execution_authorized',
               'future_authorized': 'outer_future_execution_authorized',
               'release_authorized': 'production_release_authorized'}[change]
        frozen[key] = True
    with pytest.raises(ValueError):
        h.verify_outer_protocol(tmp_path)


def test_batch_child_commands_only_invoke_admission_runner():
    tree = ast.parse((SCRIPTS / 'run_outer_admission_batch.py').read_text())
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert 'scripts/run_outer_group_admissions.py' in strings
    assert not any('evaluate_outer_case_future.py' in s or 'validate_outer_candidate.py' in s for s in strings)


def test_cli_flags_declared_for_expected_stage():
    expected = {'freeze_outer_admission_protocol.py': {'--freeze'},
                'run_outer_admission_batch.py': {'--batch-number', '--batch-version', '--execute-admissions'},
                'run_outer_group_admissions.py': {'--group-id', '--execute-admissions'},
                'run_outer_case_admission.py': {'--scenario-id', '--group-id', '--execute-selected'},
                'audit_outer_admissions.py': {'--save'}}
    for name, flags in expected.items():
        tree = ast.parse((SCRIPTS / name).read_text())
        args = {call.args[0].value for call in ast.walk(tree) if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument'
                and call.args and isinstance(call.args[0], ast.Constant)}
        assert flags <= args, name


def test_no_development_loader_that_forbids_outer_artifacts_is_called():
    tree = ast.parse((SCRIPTS / '_outer_group.py').read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert '_evaluation_engine_inputs' not in imports
    assert '_comparison_inputs' not in imports
    assert '_feature_ablation_inputs' not in imports


def load_cli(filename, common, outer, additional=None):
    spec = importlib.util.spec_from_file_location('outer_cli_' + filename[:-3], SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    modules = {'_common': common, '_outer_group': outer, **(additional or {})}
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def cli_common(tmp_path):
    import argparse
    import json
    common = ModuleType('_common')
    common.C4_ROOT = tmp_path / 'c4'
    def parser(description):
        p = argparse.ArgumentParser(description=description)
        p.add_argument('--data-root', type=Path)
        return p
    common.parser = parser
    common.resolve_data_root = lambda value: tmp_path
    common.name = str
    common.git_context = lambda: {'git_revision': 'test-only', 'working_tree_dirty': False}
    common.utc_now = lambda: 'test-only'
    common.sha = lambda b: hashlib.sha256(b).hexdigest()
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    common.require = require
    common.checked_bytes = lambda *args: b''
    common.checked_json = lambda *args: ({}, '')
    def save_json(path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(document, sort_keys=True).encode()
        with path.open('xb') as file:
            file.write(data)
        digest = common.sha(data)
        path.with_suffix('.sha256').write_text(digest)
        return digest
    common.save_json = save_json
    return common


def test_freeze_cli_plan_writes_nothing(tmp_path, monkeypatch, capsys):
    h, _ = helper()
    common = cli_common(tmp_path)
    h.current_bindings = lambda root: {'n1_binding': {'unchanged': True}}
    h.load_manifest = lambda root: (manifest(), 'original', 'manifest')
    h.source_hashes = lambda: {'code.py': 'hash'}
    module = load_cli('freeze_outer_admission_protocol.py', common, h)
    monkeypatch.setattr(sys, 'argv', ['freeze_outer_admission_protocol.py'])
    module.main()
    assert 'PLAN ONLY' in capsys.readouterr().out
    assert not list(tmp_path.rglob('*.json'))


def test_freeze_cli_saves_only_admission_authorization(tmp_path, monkeypatch):
    import json
    h, _ = helper()
    common = cli_common(tmp_path)
    h.current_bindings = lambda root: {'n1_binding': {'unchanged': True}}
    h.load_manifest = lambda root: (manifest(), 'original', 'manifest')
    h.source_hashes = lambda: {'code.py': 'hash'}
    module = load_cli('freeze_outer_admission_protocol.py', common, h)
    monkeypatch.setattr(sys, 'argv', ['freeze_outer_admission_protocol.py', '--freeze'])
    module.main()
    payload = json.loads((h.protocol_directory(tmp_path) / 'freeze.json').read_text())
    assert payload['planned_cases'] == 592
    assert payload['admission_execution_authorized'] is True
    assert payload['outer_candidate_execution_authorized'] is False
    assert payload['outer_future_execution_authorized'] is False
    with pytest.raises(ValueError, match='freeze exists'):
        module.main()


def test_freeze_cli_rejects_dirty_git_tree(tmp_path, monkeypatch):
    h, _ = helper()
    common = cli_common(tmp_path)
    common.git_context = lambda: {'working_tree_dirty': True}
    h.current_bindings = lambda root: {}
    h.load_manifest = lambda root: (manifest(), 'original', 'manifest')
    h.source_hashes = lambda: {}
    module = load_cli('freeze_outer_admission_protocol.py', common, h)
    monkeypatch.setattr(sys, 'argv', ['freeze_outer_admission_protocol.py', '--freeze'])
    with pytest.raises(ValueError, match='Commit outer admission code'):
        module.main()
    assert not list(tmp_path.rglob('*.json'))


def batch_cli(tmp_path):
    h, _ = helper()
    common = cli_common(tmp_path)
    m = manifest()
    h.outer_freeze_hash = lambda root: 'outer-freeze'
    h.verify_n1 = lambda root: {'protocol_freeze_sha256': 'original'}
    h.import_preflight = lambda: None
    group = ModuleType('run_outer_group_admissions')
    group.existing_record = lambda *args: None
    common.checked_json = lambda path, stem='metadata': (
        ({'scenario_manifest_sha256': 'manifest', 'full_execution_enabled': False}, 'original')
        if stem == 'freeze' else (m, 'manifest'))
    module = load_cli('run_outer_admission_batch.py', common, h, {'run_outer_group_admissions': group})
    return module, h, common


def test_batch_cli_plan_has_four_fixed_groups_and_no_writes(tmp_path, monkeypatch, capsys):
    module, _, _ = batch_cli(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['run_outer_admission_batch.py', '--batch-number', '4'])
    module.main()
    output = capsys.readouterr().out
    assert 'TOTAL GROUPS: 16; PLANNED CASES: 592' in output
    assert 'BATCH: 4 / 4' in output
    assert 'outer-12' in output and 'outer-15' in output
    assert 'outer-00' not in output
    assert not list(tmp_path.rglob('*.json'))


def test_import_failure_precedes_batch_claim(tmp_path, monkeypatch):
    module, h, common = batch_cli(tmp_path)
    # The runner pins the same dependencies as the existing admission system.
    for name in ['_outer_group.py', 'run_outer_group_admissions.py', 'run_outer_case_admission.py',
                 '_training_group.py', '_scoped_benchmark.py', '_common.py', '_privacy_distance.py',
                 'audit_development_repair.py', 'check_repair_distribution.py', 'check_repair_privacy.py']:
        path = common.C4_ROOT / 'scripts' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic test dependency\n')
    def fail():
        raise OSError('synthetic DLL import failure')
    module.import_preflight = fail
    monkeypatch.setattr(sys, 'argv', ['run_outer_admission_batch.py', '--batch-number', '1', '--execute-admissions'])
    with pytest.raises(OSError, match='DLL import'):
        module.main()
    assert not (h.runs_directory(tmp_path) / 'outer_admission_batches/outer-admission-b01-v1').exists()


def test_incomplete_case_is_preserved_and_rejected(tmp_path):
    h, _ = helper()
    common = cli_common(tmp_path)
    h.outer_freeze_hash = lambda root: 'outer'
    h.import_preflight = lambda: None
    h.verify_n1 = lambda root: {}
    module = load_cli('run_outer_group_admissions.py', common, h)
    row = manifest()['scenarios'][0]
    run = h.runs_directory(tmp_path) / row['scenario_id']
    run.mkdir(parents=True)
    marker = run / 'claim.json'
    marker.write_text('preserve this incomplete claim')
    with pytest.raises(ValueError, match='Incomplete scenario claim'):
        module.existing_record(tmp_path, row, 'original')
    assert marker.read_text() == 'preserve this incomplete claim'
