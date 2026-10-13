"""Plan/run a fixed seven-case admission feasibility panel in one training group.

No outcome-based selection, threshold changes, candidate training or future
evaluation. Completed cases are verified and reused, never regenerated.
"""
import subprocess
import sys

from _common import parser, resolve_data_root, checked_json, require, save_json, utc_now, git_context, C4_ROOT
from _scoped_benchmark import training_admission_panel


def main():
    p = parser(__doc__)
    p.add_argument('--execute-panel', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    frozen_directory = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    frozen, freeze_hash = checked_json(frozen_directory, 'freeze')
    manifest, manifest_hash = checked_json(frozen_directory, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'], 'Frozen manifest differs.')
    require(frozen['status'] == 'frozen_specification_execution_disabled'
            and frozen['full_execution_enabled'] is False, 'Unexpected benchmark scope.')
    rows = training_admission_panel(manifest)
    run_root = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2'
    print('GROUP:', rows[0]['group_id'], 'FOLD: forecast_train')
    print('FIXED PANEL CASES:', len(rows))
    pending = []
    cached = {}
    for row in rows:
        case_directory = run_root / row['scenario_id']
        if (case_directory / 'admission.json').exists():
            report, checksum = checked_json(case_directory, 'admission')
            require(report['scenario'] == row and report['freeze_sha256'] == freeze_hash,
                    'Cached result does not match the frozen scenario.')
            require(report['candidate_training_executed'] is False
                    and report['future_evaluation_executed'] is False, 'Unexpected executed later stage.')
            cached[row['scenario_id']] = (report, checksum)
            print('REUSE VERIFIED RESULT:', row['scenario_id'], report['simulation_admission'])
        else:
            require(not case_directory.exists(),
                    f'Incomplete claimed scenario exists: {row["scenario_id"]}; inspect it before continuing.')
            pending.append(row)
            print('PLANNED NEW CASE:', row['scenario_id'])
    print('NEW CASES TO EXECUTE:', len(pending))
    print('FULL BENCHMARK / CANDIDATE / FUTURE EXECUTION: DISABLED')
    if not args.execute_panel:
        print('PLAN ONLY: no new experiments or report writes.')
        return
    output = run_root / 'admission_panels' / 'middle-strength-targeted-v1'
    if output.exists():
        raise FileExistsError('Panel was already claimed; inspect its report or failure instead of retrying.')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'claim.json', {
        'purpose': 'fixed_training_admission_feasibility_panel', 'created_at_utc': utc_now(), **git_context(),
        'freeze_sha256': freeze_hash, 'scenario_manifest_sha256': manifest_hash,
        'scenario_ids': [row['scenario_id'] for row in rows],
        'scope': 'one_training_group_not_representative_benchmark_results',
        'threshold_changed': False, 'future_evaluation_authorized': False,
    })
    try:
        for row in pending:
            print('EXECUTING FIXED PANEL CASE:', row['scenario_id'], flush=True)
            subprocess.run([sys.executable, str(C4_ROOT / 'scripts/run_first_benchmark_admission.py'),
                            '--data-root', str(root), '--scenario-id', row['scenario_id'],
                            '--execute-selected'], check=True)
        records = []
        for row in rows:
            report, checksum = checked_json(run_root / row['scenario_id'], 'admission')
            require(report['scenario'] == row and report['freeze_sha256'] == freeze_hash,
                    'Result differs from frozen panel.')
            records.append({
                'scenario_id': row['scenario_id'], 'mechanism': row['mechanism'],
                'diagnosis_condition': row['diagnosis_condition'], 'is_no_drift_control': row['is_no_drift_control'],
                'screen_status': report['screen_status'], 'tail_distance_ratio': report['tail_distance_ratio'],
                'simulation_admission': report['simulation_admission'],
                'artifact_version': report['artifact_version'], 'admission_sha256': checksum,
            })
        admitted = sum(row['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' for row in records)
        result = {
            'purpose': 'training_admission_panel_results', 'created_at_utc': utc_now(), **git_context(),
            'freeze_sha256': freeze_hash, 'group_id': rows[0]['group_id'], 'records': records,
            'cases': len(records), 'reused_cases': len(cached), 'new_cases': len(pending),
            'scoped_admissible_cases': admitted, 'blocked_cases': len(records) - admitted,
            'candidate_training_executed': False, 'future_evaluation_executed': False,
            'full_benchmark_execution': False,
            'limitations': ['One related training group; not seven independent replications.',
                            'Admission results are not future safety or forecasting results.',
                            'Every preselected case was retained; no replacement seed or changed threshold.'],
        }
        checksum = save_json(output / 'panel.json', result)
        for record in records:
            print('PANEL RESULT:', record)
        print('SCOPED ADMISSIBLE CASES:', admitted, '/', len(records))
        print('REPORT:', output / 'panel.json')
        print('REPORT SHA256:', checksum)
        print('CANDIDATE / FUTURE EVALUATION EXECUTED: False')
    except Exception as error:
        save_json(output / 'failure.json', {
            'purpose': 'training_panel_execution_failure', 'created_at_utc': utc_now(),
            'exception_type': type(error).__name__, 'message': str(error),
            'future_evaluation_executed': False,
        })
        raise


if __name__ == '__main__':
    main()
