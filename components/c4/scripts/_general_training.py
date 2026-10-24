"""Manifest-bound selection for additional forecasting-training groups."""

from _common import checked_json, require
from _training_group import training_group_cases


def select_admitted_cases(manifest, report, group_id):
    planned = training_group_cases(manifest, group_id)
    require(report['group_id'] == group_id and report['cases'] == 37,
            'Wrong group or incomplete admission count.')
    records = report['records']
    by_id = {r['scenario_id']: r for r in records}
    require(len(records) == len(by_id) == 37, 'Duplicate or missing admission cases.')
    require(set(by_id) == {r['scenario_id'] for r in planned}, 'Admission IDs differ from manifest.')
    selected = []
    fields = ('group_id', 'split_role', 'mechanism', 'magnitude',
              'diagnosis_condition', 'repair_condition', 'is_no_drift_control')
    allowed = {'READY_FOR_SCOPED_CANDIDATE_VALIDATION', 'BLOCKED_ESCALATE'}
    for row in planned:
        record = by_id[row['scenario_id']]
        require(all(record[k] == row[k] for k in fields), 'Admission factor/fold differs from manifest.')
        require(record['simulation_admission'] in allowed, 'Unknown admission status.')
        if record['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION':
            require(record['screen_status'] == 'PASS', 'Admitted case has no passing screen.')
            selected.append(record)
    require(report['scoped_admissible_cases'] == len(selected) and
            report['blocked_cases'] == 37 - len(selected), 'Admission totals differ.')
    # No future targets, metrics or outcomes are used to decide membership.
    return sorted(selected, key=lambda r: r['scenario_id'])


def general_group_context(root, group_id, scenario_id):
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    freeze, freeze_hash = checked_json(frozen_dir, 'freeze')
    manifest, manifest_hash = checked_json(frozen_dir, 'scenario_manifest')
    require(manifest_hash == freeze['scenario_manifest_sha256'], 'Frozen manifest link differs.')
    require(freeze['full_execution_enabled'] is False, 'Unexpected execution scope.')
    run_root = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2'
    report, report_hash = checked_json(run_root / 'group_admissions' / group_id / 'v1', 'group_admissions')
    require(report['freeze_sha256'] == freeze_hash, 'Group protocol differs.')
    selected = select_admitted_cases(manifest, report, group_id)
    matches = [r for r in selected if r['scenario_id'] == scenario_id]
    require(len(matches) == 1, 'Scenario is not admitted in the requested training group.')
    record = matches[0]
    admission, checksum = checked_json(run_root / scenario_id, 'admission')
    planned = next(r for r in manifest['scenarios'] if r['scenario_id'] == scenario_id)
    require(checksum == record['admission_sha256'] and admission['scenario'] == planned and
            admission['freeze_sha256'] == freeze_hash and admission['simulation_admission'] ==
            'READY_FOR_SCOPED_CANDIDATE_VALIDATION', 'Scenario admission/provenance differs.')
    return report, report_hash, record


def group_context(root, scenario_id):
    """Resolve the group from the frozen manifest, never from an ID substring."""
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    manifest, _ = checked_json(frozen_dir, 'scenario_manifest')
    matches = [r for r in manifest['scenarios'] if r['scenario_id'] == scenario_id]
    require(len(matches) == 1, 'Scenario is missing or duplicated in the manifest.')
    return general_group_context(root, matches[0]['group_id'], scenario_id)
