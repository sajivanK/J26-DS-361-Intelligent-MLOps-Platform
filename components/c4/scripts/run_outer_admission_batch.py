"""Plan/run one fixed four-group outer admission batch; never fit N1 or read future outcomes."""
import subprocess
import sys
from pathlib import Path

from _common import C4_ROOT, checked_bytes, checked_json, git_context, name, parser, require, resolve_data_root, save_json, sha, utc_now
from _outer_group import outer_freeze_hash, import_preflight, VERSION, batch_groups, outer_group_cases, outer_groups, verify_n1
from run_outer_group_admissions import existing_record


def verify_report(root, group, rows, freeze_hash, n1_binding):
    directory = root / 'c4_artifacts/benchmark_runs/adult' / VERSION / 'group_admissions' / group / 'v1'
    report, digest = checked_json(directory, 'group_admissions')
    require(report['outer_admission_freeze_sha256'] == outer_freeze_hash(root) and report['group_id'] == group and report['freeze_sha256'] == freeze_hash and
            report['n1_binding'] == n1_binding, 'Outer admission identity/provenance differs.')
    expected = [existing_record(root, row, freeze_hash) for row in rows]
    require(all(r is not None for r in expected) and report['records'] == expected, 'Outer records differ from actual artifacts.')
    admitted = sum(r['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION' for r in expected)
    require(report['cases'] == 37 and report['scoped_admissible_cases'] == admitted and
            report['blocked_cases'] == 37-admitted, 'Admission accounting differs.')
    return {'group_id': group, 'admitted': admitted, 'blocked': 37-admitted,
            'admission_sha256': digest, 'status': 'ADMISSION_COMPLETE', 'candidate_pipeline_executed_by_batch': False,
            'future_evaluation_executed_by_batch': False}


def main():
    p = parser(__doc__)
    p.add_argument('--batch-number', type=int, default=1, help='1 through 4: four groups each in frozen manifest order.')
    p.add_argument('--batch-version', type=name, default=None, help='Optional fresh attempt ID; never overwrite earlier attempts.')
    p.add_argument('--execute-admissions', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    n1_binding = verify_n1(root)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult' / VERSION
    freeze, freeze_hash = checked_json(frozen_dir, 'freeze')
    manifest, manifest_hash = checked_json(frozen_dir, 'scenario_manifest')
    require(manifest_hash == freeze['scenario_manifest_sha256'] and
            freeze_hash == n1_binding['protocol_freeze_sha256'] and freeze['full_execution_enabled'] is False,
            'Outer and frozen N1 protocol links differ.')
    all_groups = outer_groups(manifest)
    groups = batch_groups(manifest, args.batch_number)
    for group in all_groups:
        outer_group_cases(manifest, group)
    run_root = root / 'c4_artifacts/benchmark_runs/adult' / VERSION
    print('FROZEN N1 SPECIFICATION / MODEL FILE CHECKSUMS: PASS')
    print('FOLD: outer_test; TOTAL GROUPS: 16; PLANNED CASES: 592')
    print('BATCH:', args.batch_number, '/ 4; GROUPS: 4; CASES: 148')
    print('MAXIMUM NEW TVAE FITS: 76; PAIRED CORRUPTED VARIANTS: 72')
    planned = {}
    for group in groups:
        rows = outer_group_cases(manifest, group)
        planned[group] = rows
        present = (run_root / 'group_admissions' / group / 'v1/group_admissions.json').exists()
        print('GROUP:', group, 'FOLD: outer_test CASES: 37',
              'ADMISSION_REPORT_PRESENT' if present else 'PENDING_ADMISSION')
    print('ORDER: frozen manifest order; admission rule / seeds / N1 models unchanged')
    print('THIS STAGE: ADMISSION ONLY; NO C4 CANDIDATE / FUTURE / N2 FITTING')
    if not args.execute_admissions:
        print('PLAN ONLY. Use --execute-admissions after reviewing this fixed batch.')
        return
    attempt = args.batch_version or f'outer-admission-b{args.batch_number:02d}-v1'
    output = run_root / 'outer_admission_batches' / attempt
    require(not output.exists(), 'Batch attempt exists; preserve it and inspect before recovery.')
    code_paths = [Path(__file__), *(C4_ROOT / 'scripts' / f for f in (
        '_outer_group.py', 'run_outer_group_admissions.py', 'run_outer_case_admission.py',
        '_training_group.py', '_scoped_benchmark.py', '_common.py', '_privacy_distance.py',
        'audit_development_repair.py', 'check_repair_distribution.py', 'check_repair_privacy.py'))]
    hashes = {str(p): sha(p.read_bytes()) for p in code_paths}
    import_preflight()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'claim.json', {'purpose': 'fixed_outer_admission_batch', 'created_at_utc': utc_now(),
        **git_context(), 'batch_number': args.batch_number, 'group_ids_in_fixed_order': groups,
        'fold': 'outer_test', 'n1_binding': n1_binding, 'outer_admission_freeze_sha256': outer_freeze_hash(root), 'protocol_freeze_sha256': freeze_hash,
        'scenario_manifest_sha256': manifest_hash, 'source_sha256': hashes,
        'outcome_based_selection': False, 'automatic_retries': False, 'release_authorized': False})
    records = []
    current = None
    try:
        for current in groups:
            require(verify_n1(root) == n1_binding, 'N1 changed during outer admissions.')
            for path, digest in hashes.items():
                checked_bytes(Path(path), digest)
            checked_bytes(frozen_dir / 'freeze.json', freeze_hash)
            checked_bytes(frozen_dir / 'scenario_manifest.json', manifest_hash)
            report = run_root / 'group_admissions' / current / 'v1/group_admissions.json'
            if not report.exists():
                print('RUNNING:', current, 'admission', flush=True)
                log = output / (sha(current.encode())[:12] + '-admission.log')
                with log.open('x', encoding='utf-8') as file:
                    result = subprocess.run([sys.executable, str(C4_ROOT / 'scripts/run_outer_group_admissions.py'),
                        '--data-root', str(root), '--group-id', current, '--execute-admissions'],
                        stdout=file, stderr=subprocess.STDOUT, text=True)
                require(result.returncode == 0, 'Admission stage failed; preserve artifacts and inspect ' + str(log))
            record = verify_report(root, current, planned[current], freeze_hash, n1_binding)
            records.append(record)
            print('ADMISSION COMPLETE:', current, 'ADMITTED:', record['admitted'], 'BLOCKED:', record['blocked'], flush=True)
        require(verify_n1(root) == n1_binding, 'N1 changed before batch completion.')
        for path, digest in hashes.items():
            checked_bytes(Path(path), digest)
        summary = {'purpose': 'outer_admission_batch_results', 'created_at_utc': utc_now(), **git_context(),
            'fold': 'outer_test', 'group_ids_in_fixed_order': groups, 'records': records,
            'admission_cases': 148, 'admitted': sum(r['admitted'] for r in records),
            'blocked': sum(r['blocked'] for r in records), 'n1_binding': n1_binding, 'outer_admission_freeze_sha256': outer_freeze_hash(root),
            'protocol_freeze_sha256': freeze_hash, 'N1_refitted': False, 'N2_calibrated': False,
            'candidate_pipeline_executed_by_batch': False, 'future_evaluation_executed_by_batch': False,
            'outcome_based_selection': False, 'release_authorized': False,
            'case_log_sha256': {p.name: sha(p.read_bytes()) for p in output.glob('*.log')}}
        digest = save_json(output / 'batch_results.json', summary)
        print('OUTER ADMISSION BATCH COMPLETE: 4 GROUPS / 148 CASES')
        print('ADMITTED:', summary['admitted'], 'BLOCKED:', summary['blocked'])
        print('REPORT:', output / 'batch_results.json')
        print('REPORT SHA256:', digest)
        print('OUTER ADMISSION ONLY COMPLETE; CANDIDATE / FUTURE EXECUTION REMAINS DISABLED')
    except Exception as error:
        save_json(output / 'failure.json', {'created_at_utc': utc_now(), 'group_id': current,
            'exception_type': type(error).__name__, 'message': str(error),
            'completed_group_records': records, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
