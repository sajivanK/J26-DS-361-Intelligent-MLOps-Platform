import copy
import importlib.util
import json
import sys
from pathlib import Path
import pytest

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'scripts'))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

helpers = load('c4_admitted_group_test', root / 'scripts/_admitted_training.py')
builder = load('c4_admitted_manifest_test', root / 'scripts/_benchmark_manifest.py')

@pytest.fixture
def report():
    manifest = builder.build_manifest(json.loads((root / 'configs/benchmark_protocol.json').read_text()))
    records = []
    for row in manifest['scenarios']:
        if row['group_id'] != helpers.GROUP: continue
        admitted = row['mechanism'] == 'interaction' and (row['magnitude'] == .2 or
                   (row['magnitude'] == .5 and row['diagnosis_condition'] == 'broadened_slice'))
        records.append({**row, 'simulation_admission': 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if admitted else 'BLOCKED_ESCALATE'})
    return {'group_id': helpers.GROUP, 'cases': 37, 'records': records}


def test_all_six_admitted_cases_are_selected_without_blocked_cases(report):
    selected = helpers.admitted_rows(report)
    assert len(selected) == 6
    assert all(r['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' for r in selected)
    assert {r['repair_condition'] for r in selected} == {'targeted_repair', 'label_corrupted_repair'}


def test_selection_cannot_depend_on_future_results(report):
    baseline = [r['scenario_id'] for r in helpers.admitted_rows(report)]
    for number, record in enumerate(report['records']): record['future_global_margin'] = -1. if number % 2 else 1.
    assert [r['scenario_id'] for r in helpers.admitted_rows(report)] == baseline


def test_nontraining_case_is_rejected(report):
    report['records'][0]['split_role'] = 'outer_test'
    with pytest.raises(ValueError, match='forbidden scenario fold'):
        helpers.admitted_rows(report)


def test_duplicate_group_record_is_rejected(report):
    report['records'][1] = copy.deepcopy(report['records'][0])
    with pytest.raises(ValueError, match='incomplete or duplicated'):
        helpers.admitted_rows(report)
