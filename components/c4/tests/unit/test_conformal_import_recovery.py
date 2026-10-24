"""Recovery must preserve evidence and cannot change a failed experiment's scope."""
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


def helper():
    common = ModuleType('_common')
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def checked_bytes(path, expected):
        data = Path(path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, 'Checksum mismatch')
        return data
    common.require = require
    common.sha = lambda data: hashlib.sha256(data).hexdigest()
    common.checked_bytes = checked_bytes
    path = Path(__file__).resolve().parents[2] / 'scripts/_conformal_recovery.py'
    spec = importlib.util.spec_from_file_location('isolated_conformal_recovery', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'_common': common}):
        spec.loader.exec_module(module)
    return module


def context(h):
    row = {'scenario_id': h.SCENARIO, 'group_id': h.GROUP, 'split_role': 'conformal_calibration',
           'generator_seed': h.SEED, 'repair_condition': 'targeted_repair', 'label_flip_fraction': 0.}
    binding = {'run_metadata_sha256': 'frozen-model'}
    claim = {'scenario': copy.deepcopy(row), 'n1_binding': binding,
             'full_execution_enabled': False, 'candidate_training_executed': False, 'future_evaluation_executed': False}
    failure = {'scenario_id': h.SCENARIO, 'exception_type': 'OSError',
               'message': '[WinError 126] Error loading torch/lib/shm.dll or one of its dependencies.'}
    return row, binding, claim, failure


def test_original_pre_fit_import_failure_can_be_recovered():
    h=helper()
    row, binding, claim, failure=context(h)
    h.validate_interrupted_case(claim, failure, row, binding)


@pytest.mark.parametrize('field,value', [('scenario_id','other'),('group_id','other'),
    ('split_role','forecast_train'),('generator_seed',999)])
def test_recovery_cannot_change_original_case_or_seed(field,value):
    h=helper()
    row,binding,claim,failure=context(h)
    row[field]=value
    claim['scenario']=copy.deepcopy(row)
    with pytest.raises(ValueError,match='restricted'):
        h.validate_interrupted_case(claim,failure,row,binding)


def test_other_failure_type_cannot_be_treated_as_pre_fit_import_failure():
    h=helper()
    row,binding,claim,failure=context(h)
    failure['exception_type']='ValueError'
    with pytest.raises(ValueError,match='observed PyTorch'):
        h.validate_interrupted_case(claim,failure,row,binding)


def test_future_execution_blocks_this_recovery_scope():
    h=helper()
    row,binding,claim,failure=context(h)
    claim['future_evaluation_executed']=True
    with pytest.raises(ValueError,match='scope'):
        h.validate_interrupted_case(claim,failure,row,binding)


def test_original_evidence_can_be_checked_without_modifying_it(tmp_path):
    h=helper()
    path=tmp_path/'failure.json'
    path.write_bytes(b'original failure evidence')
    hashes=h.retain_files([path])
    h.verify_retained(hashes)
    assert path.read_bytes()==b'original failure evidence'


def test_mutated_original_evidence_is_rejected(tmp_path):
    h=helper()
    path=tmp_path/'preparation.csv'
    path.write_bytes(b'original')
    hashes=h.retain_files([path])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Checksum mismatch'):
        h.verify_retained(hashes)
