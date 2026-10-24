"""Admission selection must follow planned factors, never future performance."""
from pathlib import Path
import sys
from copy import deepcopy

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from _general_training import select_admitted_cases


@pytest.fixture
def documents():
    group = 'adult-ns123-r006'
    rows = []
    for mechanism in ('marginal_multi', 'interaction', 'combined'):
        for strength, magnitude in [('low', .2), ('medium', .5), ('high', .8)]:
            for diagnosis in ('exact_slice', 'broadened_slice'):
                for repair in ('targeted_repair', 'label_corrupted_repair'):
                    rows.append(dict(scenario_id=f'{group}-{mechanism}-{strength}-{diagnosis}-{repair}',
                        group_id=group, split_role='forecast_train', mechanism=mechanism,
                        magnitude=magnitude, diagnosis_condition=diagnosis, repair_condition=repair,
                        is_no_drift_control=False, shared_base_package_id=f'{mechanism}-{strength}-{diagnosis}'))
    rows.append(dict(scenario_id=group+'-none-control', group_id=group, split_role='forecast_train',
        mechanism='none', magnitude=0., diagnosis_condition='exact_slice', repair_condition='targeted_repair',
        is_no_drift_control=True, shared_base_package_id='control'))
    records = []
    for row in rows:
        # Select one case of each mechanism, including the no-drift control.
        admitted = row['is_no_drift_control'] or (row['magnitude'] == .2 and
                    row['diagnosis_condition'] == 'exact_slice' and row['repair_condition'] == 'targeted_repair')
        records.append({**row, 'screen_status': 'PASS' if admitted else 'FAIL',
            'simulation_admission': 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if admitted else 'BLOCKED_ESCALATE',
            'admission_sha256': 'a'*64})
    return {'scenarios': rows}, {'group_id': group, 'cases': 37, 'records': records,
                                 'scoped_admissible_cases': 4, 'blocked_cases': 33}, group


def test_supports_all_mechanisms_and_control(documents):
    selected = select_admitted_cases(*documents)
    assert {r['mechanism'] for r in selected} == {'marginal_multi', 'interaction', 'combined', 'none'}
    assert len(selected) == 4


def test_future_values_do_not_select_cases(documents):
    manifest, report, group = documents
    before = [r['scenario_id'] for r in select_admitted_cases(*documents)]
    for r in report['records']:
        r['future_global_margin'] = -1 if r['screen_status'] == 'PASS' else 1
    assert [r['scenario_id'] for r in select_admitted_cases(manifest, report, group)] == before


def test_duplicate_admission_is_rejected(documents):
    manifest, report, group = documents
    report['records'][-1] = deepcopy(report['records'][0])
    with pytest.raises(ValueError, match='Duplicate'):
        select_admitted_cases(manifest, report, group)


def test_calibration_group_is_rejected(documents):
    manifest, report, group = documents
    for r in manifest['scenarios']:
        r['split_role'] = 'conformal_calibration'
    with pytest.raises(ValueError, match='forecasting-training'):
        select_admitted_cases(manifest, report, group)


def test_altered_condition_is_rejected(documents):
    manifest, report, group = documents
    report['records'][0]['magnitude'] = .9
    with pytest.raises(ValueError, match='factor/fold'):
        select_admitted_cases(manifest, report, group)


def test_unknown_admission_status_is_rejected(documents):
    manifest, report, group = documents
    report['records'][0]['simulation_admission'] = 'AUTO_DEPLOY'
    with pytest.raises(ValueError, match='Unknown admission'):
        select_admitted_cases(manifest, report, group)


def test_screen_failure_cannot_be_admitted(documents):
    manifest, report, group = documents
    report['records'][0]['screen_status'] = 'FAIL'
    with pytest.raises(ValueError, match='passing screen'):
        select_admitted_cases(manifest, report, group)


def test_totals_must_match_actual_selection(documents):
    manifest, report, group = documents
    report['scoped_admissible_cases'] = 10
    with pytest.raises(ValueError, match='totals'):
        select_admitted_cases(manifest, report, group)
