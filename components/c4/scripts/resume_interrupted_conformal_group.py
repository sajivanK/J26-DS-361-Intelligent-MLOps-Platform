"""Finish the interrupted r012 group, preserving its original claim, logs and failure."""
import subprocess
import sys
from pathlib import Path

from _common import C4_ROOT, checked_bytes, checked_json, git_context, name, parser, require, resolve_data_root, save_json, sha, utc_now
from _conformal_group import VERSION, conformal_group_cases, verify_n1
from _conformal_recovery import GROUP, SCENARIO, retain_files, verify_retained
from run_conformal_group_admissions import existing_record


def main():
    p = parser(__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume-version', type=name, default='import-resume-v1')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    n1 = verify_n1(root)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult' / VERSION
    frozen, frozen_hash = checked_json(frozen_dir, 'freeze')
    manifest, manifest_hash = checked_json(frozen_dir, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'] and frozen_hash == n1['protocol_freeze_sha256'] and
            frozen['full_execution_enabled'] is False, 'Original protocol/manifest/N1 scope differs.')
    rows = conformal_group_cases(manifest, GROUP)
    run_root = root / 'c4_artifacts/benchmark_runs/adult' / VERSION
    original = run_root / 'group_admissions' / GROUP / 'v1'
    claim, claim_hash = checked_json(original, 'claim')
    failure, failure_hash = checked_json(original, 'failure')
    require(claim['group_id'] == GROUP and claim['freeze_sha256'] == frozen_hash and
            claim['scenario_manifest_sha256'] == manifest_hash and claim['n1_binding'] == n1 and
            claim['scenario_ids'] == [r['scenario_id'] for r in rows], 'Original group claim differs.')
    require(failure['scenario_id'] == SCENARIO, 'Different group interruption; inspect before using this recovery.')
    for filename, expected in claim['source_sha256'].items():
        checked_bytes(C4_ROOT / 'scripts' / filename, expected)
    cached = {r['scenario_id']: result for r in rows if (result := existing_record(root, r, frozen_hash)) is not None}
    require(SCENARIO in cached, 'Recover the interrupted case before resuming this group.')
    pending = [r for r in rows if r['scenario_id'] not in cached]
    print('GROUP:', GROUP, 'FOLD: conformal_calibration')
    print('VERIFIED EXISTING ADMISSIONS:', len(cached), 'REMAINING CASES:', len(pending))
    print('ORIGINAL CLAIM / FAILURE / LOGS: PRESERVED; N1 / SEEDS / SCREEN: UNCHANGED')
    if (original / 'group_admissions.json').exists():
        report, digest = checked_json(original, 'group_admissions')
        require(not pending and report['records'] == [cached[r['scenario_id']] for r in rows] and report['n1_binding'] == n1,
                'Completed group differs from verified cases.')
        print('GROUP ALREADY COMPLETE:', original / 'group_admissions.json')
        print('REPORT SHA256:', digest)
        return
    require(len(cached) >= 17, 'The original sixteen completed records and recovered case must be retained.')
    if not args.execute:
        print('PLAN ONLY. Use --execute to run only remaining cases in their original fixed order.')
        return
    output = original.parent / args.resume_version
    require(not output.exists(), 'Resume attempt exists; preserve it and inspect before another attempt.')
    original_hashes = retain_files([p for p in original.rglob('*') if p.is_file()])
    code_paths = [Path(__file__), *(C4_ROOT / 'scripts' / f for f in
        ('_conformal_recovery.py', '_conformal_group.py', 'run_conformal_case_admission.py', 'run_conformal_group_admissions.py'))]
    code_hashes = {str(p): sha(p.read_bytes()) for p in code_paths}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'claim.json', {'purpose': 'resume_original_conformal_group_after_import_recovery',
        'created_at_utc': utc_now(), **git_context(), 'group_id': GROUP, 'original_claim_sha256': claim_hash,
        'original_failure_sha256': failure_hash, 'preserved_original_files': original_hashes,
        'pending_scenario_ids': [r['scenario_id'] for r in pending], 'source_sha256': code_hashes,
        'n1_binding': n1, 'automatic_retries': False, 'outcome_based_selection': False})
    current = None
    try:
        for number, row in enumerate(pending, 1):
            current = row['scenario_id']
            verify_retained(original_hashes)
            require(verify_n1(root) == n1, 'N1 changed during group resume.')
            for path, digest in code_hashes.items():
                checked_bytes(path, digest)
            print(f'REMAINING CASE {number}/{len(pending)}:', current, flush=True)
            log = output / (sha(current.encode())[:12] + '-admission.log')
            with log.open('x', encoding='utf-8') as file:
                result = subprocess.run([sys.executable, str(C4_ROOT / 'scripts/run_conformal_case_admission.py'),
                    '--data-root', str(root), '--group-id', GROUP, '--scenario-id', current, '--execute-selected'],
                    stdout=file, stderr=subprocess.STDOUT, text=True)
            require(result.returncode == 0, 'Case failed; preserve artifacts and inspect ' + str(log))
            cached[current] = existing_record(root, row, frozen_hash)
            require(cached[current] is not None, 'Missing verified case admission.')
            print('RESULT:', cached[current]['simulation_admission'], 'RATIO:', cached[current]['tail_distance_ratio'], flush=True)
        records = [cached[r['scenario_id']] for r in rows]
        admitted = sum(r['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' for r in records)
        require(verify_n1(root) == n1, 'N1 changed before completing the group.')
        verify_retained(original_hashes)
        for path, digest in code_hashes.items():
            checked_bytes(path, digest)
        report = {'purpose': 'fixed_conformal_group_admission_results', 'created_at_utc': utc_now(), **git_context(),
            'group_id': GROUP, 'n1_binding': n1, 'freeze_sha256': frozen_hash, 'cases': 37, 'records': records,
            'new_cases': len(pending), 'reused_cases': 37-len(pending), 'scoped_admissible_cases': admitted,
            'blocked_cases': 37-admitted, 'full_benchmark_execution': False,
            'candidate_training_invoked_by_batch': False, 'future_evaluation_invoked_by_batch': False,
            'recovery': {'original_claim_sha256': claim_hash, 'original_failure_sha256': failure_hash,
                         'resume_claim_path': str(output / 'claim.json'), 'original_files_unchanged': True},
            'case_log_sha256': {p.name: sha(p.read_bytes()) for p in original.glob('*.log')},
            'resume_case_log_sha256': {p.name: sha(p.read_bytes()) for p in output.glob('*.log')}}
        digest = save_json(original / 'group_admissions.json', report)
        save_json(output / 'completion.json', {'created_at_utc': utc_now(), 'group_admissions_sha256': digest,
                                               'original_files_unchanged': True})
        print('GROUP CASES COMPLETE: 37; ADMITTED:', admitted, 'BLOCKED:', 37-admitted)
        print('REPORT:', original / 'group_admissions.json')
        print('REPORT SHA256:', digest)
        print('CANDIDATE / FUTURE / N2 EXECUTED: False')
    except Exception as error:
        save_json(output / 'failure.json', {'created_at_utc': utc_now(), 'scenario_id': current,
            'exception_type': type(error).__name__, 'message': str(error), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
