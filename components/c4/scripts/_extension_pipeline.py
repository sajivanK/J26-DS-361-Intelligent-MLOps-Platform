"""Manifest-bound selection for additional extension-calibration groups."""

from _common import checked_json, require
from _extension_group import extension_group_cases, verify_n1, VERSION, EXTENSION_HASH, MANIFEST_HASH, load_extension
from _conformal_group import N1_SPEC_HASH, N1_RUN_HASH
from pathlib import Path
import json
from _common import checked_bytes


def select_admitted_cases(manifest, report, group_id):
    require(report.get('extension_freeze_sha256') == EXTENSION_HASH, 'Extension admission report binding differs.')
    planned = extension_group_cases(manifest, group_id)
    require(report['group_id'] == group_id and report['cases'] == 37,
            'Wrong group or incomplete admission count.')
    records = report['records']
    by_id = {r['scenario_id']: r for r in records}
    require(len(records) == len(by_id) == 37, 'Duplicate or missing admission cases.')
    require(set(by_id) == {r['scenario_id'] for r in planned}, 'Admission IDs differ from manifest.')
    require(report['n1_binding']['specification_sha256'] == N1_SPEC_HASH and
            report['n1_binding']['run_metadata_sha256'] == N1_RUN_HASH, 'Admission links to different N1.')
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
    n1_binding = verify_n1(root)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    freeze, freeze_hash, protocol, manifest, manifest_hash, extension_hash = load_extension(root)
    run_root = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2'
    report, report_hash = checked_json(run_root / 'group_admissions' / group_id / 'v1', 'group_admissions')
    require(report['freeze_sha256'] == freeze_hash == n1_binding['protocol_freeze_sha256'] and
            report['n1_binding'] == n1_binding, 'Group protocol/N1 binding differs.')
    selected = select_admitted_cases(manifest, report, group_id)
    matches = [r for r in selected if r['scenario_id'] == scenario_id]
    require(len(matches) == 1, 'Scenario is not admitted in the requested extension group.')
    record = matches[0]
    admission, checksum = checked_json(run_root / scenario_id, 'admission')
    planned = next(r for r in manifest['scenarios'] if r['scenario_id'] == scenario_id)
    require(checksum == record['admission_sha256'] and admission['scenario'] == planned and
            admission['freeze_sha256'] == freeze_hash and admission['simulation_admission'] ==
            'READY_FOR_SCOPED_CANDIDATE_VALIDATION', 'Scenario admission/provenance differs.')
    require(admission.get('extension_freeze_sha256') == EXTENSION_HASH, 'Scenario extension link differs.')
    require(admission['n1_binding'] == n1_binding, 'Scenario N1 binding differs.')
    return report, report_hash, record


def group_context(root, scenario_id):
    """Resolve the group from the frozen manifest, never from an ID substring."""
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    _, _, _, manifest, _, _ = load_extension(root)
    matches = [r for r in manifest['scenarios'] if r['scenario_id'] == scenario_id]
    require(len(matches) == 1, 'Scenario is missing or duplicated in the manifest.')
    return general_group_context(root, matches[0]['group_id'], scenario_id)


def extension_support(nonempty_groups):
    """Frozen alpha=.05 finite-sample rank; no empty-group replacements."""
    from decimal import Decimal, ROUND_CEILING
    require(type(nonempty_groups) is int and 0 <= nonempty_groups <= 20, 'Invalid nonempty extension group count.')
    rank = int((Decimal(nonempty_groups + 1) * Decimal('0.95')).to_integral_value(rounding=ROUND_CEILING))
    return {'nonempty_groups': nonempty_groups, 'quantile_rank': rank,
            'finite_adjustment_available': rank <= nonempty_groups,
            'insufficient_support_fallback': 'bounds_minus1_plus1_and_disable_AUTO_DEPLOY'}


def verify_freeze_dependencies(snapshot):
    require(snapshot.get('complete_decision_evidence_frozen') is True and
            snapshot.get('future_evaluation_executed') is False, 'Complete original pre-future freeze is required.')
    require(snapshot.get('scope') == 'one_admitted_extension_scenario_under_simulated_upstream_evidence' and
            snapshot['scenario']['split_role'] == 'conformal_extension', 'Wrong extension freeze scope.')
    require(snapshot['n1_binding']['specification_sha256'] == N1_SPEC_HASH and
            snapshot['n1_binding']['run_metadata_sha256'] == N1_RUN_HASH, 'Wrong N1 model/specification binding.')
    require(snapshot.get('extension_freeze_sha256') == EXTENSION_HASH, 'Wrong extension evidence freeze.')
    dependencies = snapshot.get('dependencies', {})
    require(bool(dependencies), 'Missing frozen dependencies.')
    require(any(Path(p).name == 'freeze.json' and digest == EXTENSION_HASH for p,digest in dependencies.items()),
            'Extension specification is absent from frozen dependencies.')
    n1_dir = Path(snapshot['n1_run_directory'])
    required = {str(n1_dir / 'metadata.json'): N1_RUN_HASH,
                str(n1_dir.parent / 'specification.json'): N1_SPEC_HASH}
    model_metadata = json.loads(checked_bytes(n1_dir / 'metadata.json', N1_RUN_HASH))
    for filename, expected in model_metadata['files'].items():
        required[str(n1_dir / filename)] = expected
    require(all(dependencies.get(path) == expected for path, expected in required.items()),
            'Extension freeze must pin all original N1 artifacts.')
    require(snapshot.get('n1_forecasts_frozen') is True, 'Pre-future N1 forecasts are missing.')
    forecast_path = snapshot['n1_forecasts_path']
    require(dependencies.get(forecast_path) == snapshot['n1_forecasts_sha256'], 'N1 forecast dependency is missing.')
    for path, checksum in dependencies.items():
        checked_bytes(path, checksum)
