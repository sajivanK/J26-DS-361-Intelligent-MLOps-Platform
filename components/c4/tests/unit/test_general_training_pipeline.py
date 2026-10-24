"""Completed-case reuse verifies group identity and saved dependencies."""
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import run_general_admitted_training_pipeline as pipeline


@pytest.fixture
def completed(tmp_path, monkeypatch):
    record = {'scenario_id': 'adult-ns123-r006-interaction-high-exact_slice-targeted_repair',
              'group_id': 'adult-ns123-r006', 'admission_sha256': 'a'*64}
    run = tmp_path / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / record['scenario_id']
    output = run / 'future_research-v1'
    output.mkdir(parents=True)
    (output / 'outcomes.json').write_text('{}', encoding='utf-8')
    outcomes = {'decision_freeze_sha256': 'f'*64,
        'scenario': {**record, 'split_role': 'forecast_train'},
        'files': {'future_predictions.csv': 'p'*64, 'future_membership.csv': 'm'*64}}
    snapshot = {'marker': 'original-freeze'}
    calls = []

    def checked_json(directory, stem):
        if stem == 'freeze':
            return snapshot, 'f'*64
        assert stem == 'outcomes'
        return outcomes, 'o'*64

    monkeypatch.setattr(pipeline, 'checked_json', checked_json)
    monkeypatch.setattr(pipeline, 'verify_freeze_dependencies', lambda s: calls.append(('freeze', s)))
    monkeypatch.setattr(pipeline, 'checked_bytes', lambda p, h: calls.append((Path(p).name, h)))
    return tmp_path, record, outcomes, calls


def test_completed_case_uses_requested_group_and_checks_every_saved_file(completed):
    root, record, outcomes, calls = completed
    result = pipeline.completed_case(root, record)
    assert result['group_id'] == 'adult-ns123-r006'
    assert result['scenario_id'] == record['scenario_id']
    assert result['admission_sha256'] == record['admission_sha256']
    assert calls[0][0] == 'freeze'
    assert {c[0] for c in calls[1:]} == set(outcomes['files'])


@pytest.mark.parametrize('field', ['group_id', 'scenario_id'])
def test_case_from_another_group_or_scenario_is_not_reused(completed, field):
    root, record, outcomes, _ = completed
    outcomes['scenario'][field] = 'different'
    with pytest.raises(ValueError, match='links differ'):
        pipeline.completed_case(root, record)


def test_tampered_saved_predictions_stop_reuse(completed, monkeypatch):
    root, record, _, _ = completed

    def fail_checksum(path, expected):
        raise ValueError('Checksum mismatch: ' + str(path))

    monkeypatch.setattr(pipeline, 'checked_bytes', fail_checksum)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        pipeline.completed_case(root, record)


def test_unfinished_case_is_pending_not_silently_completed(tmp_path):
    assert pipeline.completed_case(tmp_path, {'scenario_id': 'pending'}) is None
