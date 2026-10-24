"""Verify scheduling changes preserve record checks and completed groups."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


def common_module(tmp_path):
    m = ModuleType('_common')
    m.C4_ROOT = tmp_path / 'c4'
    m.sha = lambda b: hashlib.sha256(b).hexdigest()
    def require(value, message):
        if not value:
            raise ValueError(message)
    m.require = require
    def checked_bytes(path, expected):
        data = Path(path).read_bytes()
        require(m.sha(data) == expected, 'Checksum mismatch: ' + str(path))
        return data
    m.checked_bytes = checked_bytes
    def checked_json(directory, stem='metadata'):
        directory = Path(directory)
        digest = (directory / (stem+'.sha256')).read_text().strip()
        return json.loads(checked_bytes(directory / (stem+'.json'), digest)), digest
    m.checked_json = checked_json
    def save_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        with path.open('xb') as f:
            f.write(data)
        digest = m.sha(data)
        path.with_suffix('.sha256').write_text(digest)
        return digest
    m.save_json = save_json
    m.utc_now = lambda: 'test-only'
    m.git_context = lambda: {'working_tree_dirty': False, 'git_revision': 'test-only'}
    m.name = str
    m.resolve_data_root = lambda value: tmp_path
    def parser(description):
        p = argparse.ArgumentParser(description=description)
        p.add_argument('--data-root', type=Path)
        return p
    m.parser = parser
    return m


def outer_module(tmp_path):
    m = ModuleType('_outer_group')
    m.VERSION = 'adult-simulation-pp1-v2'
    m.outer_freeze_hash = lambda root: 'd8815e1cc433daa45963ede0f9075e7d2ec4038dd2b9f6785cf38cd7ffa859b9'
    m.source_hashes = lambda: {'old': 'source'}
    m.load_manifest = lambda root: ({}, 'original', 'manifest')
    m.protocol_directory = lambda root: root / 'old-protocol'
    m.assert_admission_only = lambda *args: None
    m.runs_directory = lambda root: root / 'c4_artifacts/benchmark_runs/adult' / m.VERSION
    m.import_preflight = lambda: None
    m.verify_outer_protocol = lambda root: ({'binding': {'n1_binding': {'model': 'same'}},
                                            'source_sha256': {'old': 'source'}}, m.outer_freeze_hash(root))
    return m


def load(filename, modules):
    spec = importlib.util.spec_from_file_location('portable_test_' + filename[:-3], SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def setup(tmp_path):
    c, o = common_module(tmp_path), outer_module(tmp_path)
    h = load('_outer_admission_verification.py', {'_common': c, '_outer_group': o})
    return c, o, h


def create_record(tmp_path, common, outer, status='READY_FOR_SCOPED_CANDIDATE_VALIDATION'):
    row = {'scenario_id': 'outer-example', 'group_id': 'outer-group', 'split_role': 'outer_test',
           'mechanism': 'interaction', 'magnitude': .2, 'diagnosis_condition': 'exact_slice',
           'repair_condition': 'targeted_repair', 'is_no_drift_control': False}
    run = outer.runs_directory(tmp_path) / row['scenario_id']
    package = tmp_path / 'c4_artifacts/offline_repair/adult/example-repair'
    package.mkdir(parents=True)
    (package / 'synthetic_unvalidated.csv').write_bytes(b'example CSV bytes')
    (package / 'generator.pkl').write_bytes(b'example generator bytes')
    generation = {'synthetic_csv_sha256': common.sha(b'example CSV bytes'),
                  'generator_sha256': common.sha(b'example generator bytes')}
    generation_hash = common.save_json(package / 'metadata.json', generation)
    privacy_hash = common.save_json(package / 'privacy_diagnostics.json', {
        'generation_metadata_sha256': generation_hash,
        'summaries': {'synthetic_to_generator_fit': {}, 'diagnosed_internal_validation_to_generator_fit': {}}})
    screen = {'screen_status': 'PASS', 'tail_distance_ratio': 1.1}
    admission = {**screen, 'scenario': row, 'freeze_sha256': 'original', 'n1_binding': {'model': 'same'},
        'outer_admission_freeze_sha256': outer.outer_freeze_hash(tmp_path), 'artifact_version': 'example-repair',
        'generation_metadata_sha256': generation_hash, 'privacy_report_sha256': privacy_hash,
        'simulated_c2_diagnosis': {'diagnosis_stability': .8}, 'simulation_admission': status,
        'candidate_training_executed': False, 'future_evaluation_executed': False}
    common.save_json(run / 'admission.json', admission)
    protocol_dir = tmp_path / 'c4_artifacts/benchmark_protocols/adult' / outer.VERSION
    protocol_dir.mkdir(parents=True)
    data = b'{}'
    (protocol_dir / 'protocol.json').write_bytes(data)
    common.save_json(protocol_dir / 'freeze.json', {'protocol_sha256': common.sha(data)})
    screen_module = ModuleType('_scoped_benchmark')
    screen_module.frozen_tail_screen = lambda *args, **kwargs: screen.copy()
    return row, run, package, screen_module


def test_record_verification_does_not_repeat_full_shared_verification(tmp_path):
    c, o, h = setup(tmp_path)
    row, _, _, screen = create_record(tmp_path, c, o)
    def forbidden(*args):
        raise AssertionError('Record-level check must not reread all shared checkpoints')
    h.full_boundary = forbidden
    with patch.dict(sys.modules, {'_scoped_benchmark': screen}):
        result = h.existing_record(tmp_path, row, 'original', {'model': 'same'})
    assert result['scenario_id'] == 'outer-example'
    assert result['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION'


@pytest.mark.parametrize('filename', ['synthetic_unvalidated.csv', 'generator.pkl'])
def test_modified_repair_bytes_still_rejected(tmp_path, filename):
    c, o, h = setup(tmp_path)
    row, _, package, screen = create_record(tmp_path, c, o)
    (package / filename).write_bytes(b'modified')
    with patch.dict(sys.modules, {'_scoped_benchmark': screen}):
        with pytest.raises(ValueError, match='Checksum mismatch'):
            h.existing_record(tmp_path, row, 'original', {'model': 'same'})


def test_wrong_n1_binding_still_rejected(tmp_path):
    c, o, h = setup(tmp_path)
    row, _, _, screen = create_record(tmp_path, c, o)
    with patch.dict(sys.modules, {'_scoped_benchmark': screen}):
        with pytest.raises(ValueError, match='different N1'):
            h.existing_record(tmp_path, row, 'original', {'model': 'changed'})


def test_changed_screen_still_rejected(tmp_path):
    c, o, h = setup(tmp_path)
    row, _, _, screen = create_record(tmp_path, c, o)
    screen.frozen_tail_screen = lambda *args, **kwargs: {'screen_status': 'FAIL', 'tail_distance_ratio': .5}
    with patch.dict(sys.modules, {'_scoped_benchmark': screen}):
        with pytest.raises(ValueError, match='screen differs'):
            h.existing_record(tmp_path, row, 'original', {'model': 'same'})


def test_incomplete_claim_still_rejected_without_deletion(tmp_path):
    c, o, h = setup(tmp_path)
    row = {'scenario_id': 'incomplete'}
    run = o.runs_directory(tmp_path) / 'incomplete'
    c.save_json(run / 'claim.json', {'preserve': True})
    with pytest.raises(ValueError, match='Incomplete scenario claim'):
        h.existing_record(tmp_path, row, 'original', {'model': 'same'})
    assert c.checked_json(run, 'claim')[0] == {'preserve': True}


def test_claimed_admission_decision_must_match_recomputed_screen(tmp_path):
    c, o, h = setup(tmp_path)
    row, _, _, screen = create_record(tmp_path, c, o, status='BLOCKED_ESCALATE')
    with patch.dict(sys.modules, {'_scoped_benchmark': screen}):
        with pytest.raises(ValueError, match='status/execution flags'):
            h.existing_record(tmp_path, row, 'original', {'model': 'same'})


def test_revision_preserves_original_and_refuses_overwrite(tmp_path):
    c, o, h = setup(tmp_path)
    h.source_hashes = lambda: {'new': 'source'}
    digest = h.freeze_revision(tmp_path)
    saved, actual = c.checked_json(h.revision_directory(tmp_path), 'freeze')
    assert actual == digest
    assert saved['original_outer_admission_freeze_sha256'] == h.ORIGINAL_OUTER_FREEZE
    assert saved['record_checksums_and_screen_recomputation_retained'] is True
    assert saved['outer_future_execution_authorized'] is False
    with pytest.raises(ValueError, match='revision exists'):
        h.freeze_revision(tmp_path)


def test_revision_checks_changed_source(tmp_path):
    c, o, h = setup(tmp_path)
    h.source_hashes = lambda: {'new': 'source'}
    h.freeze_revision(tmp_path)
    h.source_hashes = lambda: {'new': 'changed'}
    with pytest.raises(ValueError, match='revision differs'):
        h.verify_revision(tmp_path, full=True)


def test_revision_checks_changed_model_binding(tmp_path):
    c, o, h = setup(tmp_path)
    h.source_hashes = lambda: {'new': 'source'}
    h.freeze_revision(tmp_path)
    h.full_boundary = lambda root: ({'binding': {'n1_binding': {'model': 'changed'}}}, h.ORIGINAL_OUTER_FREEZE)
    with pytest.raises(ValueError, match='revision differs'):
        h.verify_revision(tmp_path, full=True)


def test_shared_checkpoint_failures_propagate(tmp_path):
    _, o, h = setup(tmp_path)
    # The imported function is replaced to simulate the existing cryptographic guard failing.
    def fail(root):
        raise ValueError('Frozen model checksum changed')
    h.verify_outer_protocol = fail
    with pytest.raises(ValueError, match='checksum changed'):
        h.full_boundary(tmp_path)


def test_completed_group_reuse_never_launches_case_training(tmp_path, monkeypatch):
    c, o, h = setup(tmp_path)
    row = {'scenario_id': 'complete', 'group_id': 'g'}
    record = {'scenario_id': 'complete', 'admission_sha256': 'admission'}
    # Selection/accounting of 37 rows is tested by original scope tests; here
    # verify the actual completed-group CLI path does no subprocess work.
    o.outer_group_cases = lambda manifest, group: [row]
    h.existing_record = lambda *args: record
    calls = []
    h.verify_revision = lambda root, full=False: (calls.append(full) or
        ({'binding': {'n1_binding': {'protocol_freeze_sha256': 'original'}}}, 'revision'))
    group_dir = o.runs_directory(tmp_path) / 'group_admissions/g/v1'
    c.save_json(group_dir / 'group_admissions.json', {'outer_admission_freeze_sha256': o.outer_freeze_hash(tmp_path),
        'freeze_sha256': 'original', 'records': [record]})
    def read(directory, stem='metadata'):
        if stem == 'freeze':
            return {'scenario_manifest_sha256': 'manifest', 'full_execution_enabled': False}, 'original'
        if stem == 'scenario_manifest':
            return {}, 'manifest'
        return c.checked_json(directory, stem)
    modules = {'_common': c, '_outer_group': o, '_outer_admission_verification': h}
    module = load('run_outer_group_admissions_portable.py', modules)
    module.checked_json = read
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed group must not launch case training')
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run_outer_group_admissions_portable.py', '--group-id', 'g', '--execute-admissions'])
    module.main()
    assert calls == [True, True]


def test_batch_completed_groups_do_not_launch_children_and_keep_boundary_checks(tmp_path, monkeypatch):
    c, o, h = setup(tmp_path)
    groups = ['g1', 'g2', 'g3', 'g4']
    o.batch_groups = lambda manifest, number: groups
    o.outer_group_cases = lambda manifest, group: []
    original_binding = {'model': 'same'}
    checkpoint = {'binding': {'n1_binding': original_binding}}
    calls = []
    h.verify_revision = lambda root, full=False: (calls.append(full) or (checkpoint, 'revision'))
    h.source_hashes = lambda: {}
    h.verify_report = lambda root, group, rows, digest, binding: {
        'group_id': group, 'admitted': 2, 'blocked': 35, 'admission_sha256': group, 'status': 'ADMISSION_COMPLETE'}
    for group in groups:
        c.save_json(o.runs_directory(tmp_path) / 'group_admissions' / group / 'v1/group_admissions.json', {'complete': True})
    module = load('run_outer_admission_batch_portable.py',
                  {'_common': c, '_outer_group': o, '_outer_admission_verification': h})
    def forbidden(*args):
        raise AssertionError('Completed groups must be reused')
    module.run_group = forbidden
    monkeypatch.setattr(sys, 'argv', ['run_outer_admission_batch_portable.py', '--execute-admissions'])
    module.main()
    assert calls == [True] * 9  # Initial binding plus before/after each of four groups.
    saved, _ = c.checked_json(o.runs_directory(tmp_path) / 'outer_admission_batches/outer-admission-b01-portable-v2', 'batch_results')
    assert saved['admitted'] == 8 and saved['blocked'] == 140
    assert saved['future_evaluation_executed_by_batch'] is False


def test_resume_rejects_changed_group_order(tmp_path):
    c, o, h = setup(tmp_path)
    claim = {'group_ids_in_fixed_order': ['g2', 'g1'], 'n1_binding': {'model': 'same'},
             'protocol_freeze_sha256': 'original', 'fold': 'outer_test',
             'outer_admission_freeze_sha256': h.ORIGINAL_OUTER_FREEZE}
    c.save_json(o.runs_directory(tmp_path) / 'outer_admission_batches/old/claim.json', claim)
    o.batch_groups = lambda *args: []
    o.outer_group_cases = lambda *args: []
    module = load('run_outer_admission_batch_portable.py',
                  {'_common': c, '_outer_group': o, '_outer_admission_verification': h})
    with pytest.raises(ValueError, match='Previous batch identity'):
        module.validate_resume(tmp_path, 'old', ['g1', 'g2'], {'model': 'same'}, 'original')


def test_resume_does_not_modify_old_claim(tmp_path):
    c, o, h = setup(tmp_path)
    claim = {'group_ids_in_fixed_order': ['g1'], 'n1_binding': {'model': 'same'},
             'protocol_freeze_sha256': 'original', 'fold': 'outer_test',
             'outer_admission_freeze_sha256': h.ORIGINAL_OUTER_FREEZE}
    directory = o.runs_directory(tmp_path) / 'outer_admission_batches/old'
    c.save_json(directory / 'claim.json', claim)
    before = (directory / 'claim.json').read_bytes()
    o.batch_groups = lambda *args: []
    o.outer_group_cases = lambda *args: []
    module = load('run_outer_admission_batch_portable.py',
                  {'_common': c, '_outer_group': o, '_outer_admission_verification': h})
    result = module.validate_resume(tmp_path, 'old', ['g1'], {'model': 'same'}, 'original')
    assert result['attempt'] == 'old'
    assert (directory / 'claim.json').read_bytes() == before


def test_original_case_cli_is_retained():
    tree = ast.parse((SCRIPTS / 'run_outer_group_admissions_portable.py').read_text())
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert 'scripts/run_outer_case_admission.py' in strings
    assert not any('evaluate_outer_case_future' in s for s in strings)
