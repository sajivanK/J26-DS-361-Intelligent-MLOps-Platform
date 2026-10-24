import copy
import importlib.util
import json
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_module('c4_panel_selection', root / 'scripts/_scoped_benchmark.py')
builder = load_module('c4_panel_manifest', root / 'scripts/_benchmark_manifest.py')


@pytest.fixture
def manifest():
    protocol = json.loads((root / 'configs/benchmark_protocol.json').read_text())
    return builder.build_manifest(protocol)


def test_panel_is_exactly_seven_cases_in_one_training_group(manifest):
    rows = helpers.training_admission_panel(manifest)
    assert len(rows) == 7
    assert len({row['group_id'] for row in rows}) == 1
    assert {row['split_role'] for row in rows} == {'forecast_train'}


def test_panel_has_six_middle_conditions_and_one_control(manifest):
    rows = helpers.training_admission_panel(manifest)
    assert sum(row['is_no_drift_control'] for row in rows) == 1
    assert {row['magnitude'] for row in rows if not row['is_no_drift_control']} == {.5}
    assert {row['repair_condition'] for row in rows} == {'targeted_repair'}


def test_panel_retains_the_original_failed_scenario(manifest):
    selected = helpers.select_training_scenario(manifest)
    assert selected['scenario_id'] in {row['scenario_id'] for row in helpers.training_admission_panel(manifest)}


def test_incomplete_panel_is_rejected(manifest):
    changed = copy.deepcopy(manifest)
    control = next(row for row in helpers.training_admission_panel(changed) if row['is_no_drift_control'])
    changed['scenarios'] = [row for row in changed['scenarios'] if row['scenario_id'] != control['scenario_id']]
    with pytest.raises(ValueError, match='exact seven'):
        helpers.training_admission_panel(changed)
