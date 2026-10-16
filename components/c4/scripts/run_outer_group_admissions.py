"""Plan or run admission for all 37 cases in one frozen outer-test group.

No C4 candidate or future evaluator is called. Completed cases are reused.
"""
import subprocess
import sys
from pathlib import Path

from _common import parser, resolve_data_root, checked_json, checked_bytes, require, save_json, sha, utc_now, git_context, C4_ROOT
from _outer_group import outer_freeze_hash, import_preflight, outer_group_cases, verify_n1


def existing_record(root, row, freeze_hash):
    run = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / row['scenario_id']
    if not run.exists(): return None
    require((run / 'admission.json').is_file(),
            'Incomplete scenario claim found; preserve it and resolve the recorded failure before continuing: ' + str(run))
    admission, admission_hash = checked_json(run, 'admission')
    require(admission['scenario'] == row and admission['freeze_sha256'] == freeze_hash, 'Existing scenario/protocol differs.')
    require(admission.get('n1_binding') == verify_n1(root) and admission.get('outer_admission_freeze_sha256') == outer_freeze_hash(root), 'Existing outer case links to different N1 evidence.')
    package = root / 'c4_artifacts/offline_repair/adult' / admission['artifact_version']
    generation, generation_hash = checked_json(package)
    privacy, privacy_hash = checked_json(package, 'privacy_diagnostics')
    require(generation_hash == admission['generation_metadata_sha256'] and
            privacy_hash == admission['privacy_report_sha256'] and
            privacy['generation_metadata_sha256'] == generation_hash, 'Existing admission artifact links differ.')
    import json
    from _scoped_benchmark import frozen_tail_screen
    protocol_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    protocol_freeze, _ = checked_json(protocol_dir, 'freeze')
    protocol = json.loads(checked_bytes(protocol_dir / 'protocol.json', protocol_freeze['protocol_sha256']))
    screen = frozen_tail_screen(protocol, privacy['summaries']['synthetic_to_generator_fit'],
                                privacy['summaries']['diagnosed_internal_validation_to_generator_fit'],
                                protocol_kind='benchmark')
    require(all(admission.get(key) == value for key, value in screen.items()),
            'Saved admission screen differs from recomputation.')
    accepted = screen['screen_status'] == 'PASS' and admission['simulated_c2_diagnosis']['diagnosis_stability'] >= .50
    expected_status = 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if accepted else 'BLOCKED_ESCALATE'
    require(admission['simulation_admission'] == expected_status and
            admission['candidate_training_executed'] is False and admission['future_evaluation_executed'] is False,
            'Saved admission status/execution flags differ.')
    checked_bytes(package / 'synthetic_unvalidated.csv', generation['synthetic_csv_sha256'])
    checked_bytes(package / 'generator.pkl', generation['generator_sha256'])
    if generation.get('corruption_variant'):
        checked_bytes(package / 'corruption_mask.csv', generation['corruption_variant']['mask_sha256'])
    return {'scenario_id': row['scenario_id'], 'group_id': row['group_id'], 'split_role': row['split_role'],
            'mechanism': row['mechanism'], 'magnitude': row['magnitude'], 'diagnosis_condition': row['diagnosis_condition'],
            'repair_condition': row['repair_condition'], 'is_no_drift_control': row['is_no_drift_control'],
            'screen_status': admission['screen_status'], 'tail_distance_ratio': admission['tail_distance_ratio'],
            'simulation_admission': admission['simulation_admission'], 'artifact_version': admission['artifact_version'],
            'admission_sha256': admission_hash}


def main():
    p = parser(__doc__)
    p.add_argument('--group-id', required=True, help='Explicit frozen outer-test group; excludes forecasting training, conformal calibration and policy calibration.')
    p.add_argument('--execute-admissions', action='store_true', help='Run all remaining cases in this one group, without adaptive retries.')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    n1_binding = verify_n1(root)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    frozen, freeze_hash = checked_json(frozen_dir, 'freeze')
    manifest, manifest_hash = checked_json(frozen_dir, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'] and frozen['full_execution_enabled'] is False,
            'Wrong frozen scope or manifest.')
    require(freeze_hash == n1_binding['protocol_freeze_sha256'], 'N1 and outer protocol differ.')
    rows = outer_group_cases(manifest, args.group_id)
    group = rows[0]['group_id']
    cached = {row['scenario_id']: result for row in rows if
              (result := existing_record(root, row, freeze_hash)) is not None}
    pending = [row for row in rows if row['scenario_id'] not in cached]
    output = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2/group_admissions' / group / 'v1'
    print('GROUP:', group, 'FOLD: outer_test')
    print('PLANNED CASES:', len(rows), 'EXISTING VERIFIED CASES:', len(cached), 'NEW CASES:', len(pending))
    print('NEW TVAE FITS:', sum(row['repair_condition'] == 'targeted_repair' for row in pending))
    print('PAIRED LABEL-CORRUPTED CONDITIONS:', sum(row['repair_condition'] == 'label_corrupted_repair' for row in pending))
    print('FULL BENCHMARK EXECUTION: DISABLED; THIS STAGE: ADMISSION ONLY')
    if (output / 'group_admissions.json').exists():
        saved, saved_hash = checked_json(output, 'group_admissions')
        require(len(cached) == len(rows) and saved['outer_admission_freeze_sha256'] == outer_freeze_hash(root) and saved['freeze_sha256'] == freeze_hash and saved['records'] == [cached[row['scenario_id']] for row in rows],
                'Completed group report differs from verified cases.')
        print('GROUP ALREADY COMPLETE; REPORT:', output / 'group_admissions.json')
        print('REPORT SHA256:', saved_hash)
        return
    require(not output.exists(), 'Incomplete batch claim exists; preserve failure/log files for diagnosis.')
    if not args.execute_admissions:
        for row in pending: print('PENDING:', row['scenario_id'])
        print('PLAN ONLY. Use --execute-admissions to run this fixed group once.')
        return
    import_preflight()
    output.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), C4_ROOT / 'scripts/run_outer_case_admission.py', C4_ROOT / 'scripts/_outer_group.py']
    code_hashes = {path.name: sha(path.read_bytes()) for path in code_paths}
    save_json(output / 'claim.json', {'purpose': 'fixed_outer_group_admission_claim', 'created_at_utc': utc_now(), **git_context(),
        'group_id': group, 'n1_binding': n1_binding, 'outer_admission_freeze_sha256': outer_freeze_hash(root), 'freeze_sha256': freeze_hash, 'scenario_manifest_sha256': manifest_hash,
        'scenario_ids': [r['scenario_id'] for r in rows], 'reused_scenario_ids': list(cached),
        'source_sha256': code_hashes, 'log_filename_scheme': 'first_12_SHA256_hex-admission.log', 'candidate_training_invoked_by_batch': False, 'future_evaluation_invoked_by_batch': False})
    current = None
    try:
        for number, row in enumerate(pending, 1):
            current = row['scenario_id']
            print(f'CASE {number}/{len(pending)}: {current}', flush=True)
            log_path = output / (sha(current.encode('utf-8'))[:12] + '-admission.log')
            with log_path.open('x', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, str(C4_ROOT / 'scripts/run_outer_case_admission.py'),
                    '--data-root', str(root), '--group-id', group, '--scenario-id', current, '--execute-selected'],
                    stdout=log, stderr=subprocess.STDOUT, text=True)
            require(result.returncode == 0, 'Case execution failed; inspect ' + str(log_path))
            cached[current] = existing_record(root, row, freeze_hash)
            print('RESULT:', cached[current]['simulation_admission'], 'RATIO:', cached[current]['tail_distance_ratio'], flush=True)
        records = [cached[row['scenario_id']] for row in rows]
        for path in code_paths: checked_bytes(path, code_hashes[path.name])
        checked_bytes(frozen_dir / 'freeze.json', freeze_hash)
        admitted = sum(r['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' for r in records)
        report = {'purpose': 'fixed_outer_group_admission_results', 'created_at_utc': utc_now(), **git_context(),
            'group_id': group, 'n1_binding': n1_binding, 'outer_admission_freeze_sha256': outer_freeze_hash(root), 'freeze_sha256': freeze_hash, 'cases': len(records), 'records': records,
            'new_cases': len(pending), 'reused_cases': len(rows)-len(pending), 'scoped_admissible_cases': admitted,
            'blocked_cases': len(records)-admitted, 'full_benchmark_execution': False,
            'candidate_training_invoked_by_batch': False, 'future_evaluation_invoked_by_batch': False,
            'scope_note': 'One related outer group; admission only, no independent future performance measured yet.',
            'case_log_sha256': {path.name: sha(path.read_bytes()) for path in output.glob('*.log')}}
        checksum = save_json(output / 'group_admissions.json', report)
        print('GROUP CASES COMPLETE:', len(records))
        print('SCOPED ADMISSIBLE CASES:', admitted, '/', len(records))
        print('BLOCKED CASES:', len(records)-admitted)
        print('REPORT:', output / 'group_admissions.json')
        print('REPORT SHA256:', checksum)
        print('THIS BATCH INVOKED C4 CANDIDATE / FUTURE EVALUATION: False')
    except Exception as error:
        save_json(output / 'failure.json', {'created_at_utc': utc_now(), 'scenario_id': current,
            'exception_type': type(error).__name__, 'message': str(error),
            'scope': 'No automatic replacement seeds or threshold changes; preserve failure evidence.'})
        raise


if __name__ == '__main__':
    main()
