"""Plan/resume outer admissions with visible progress and boundary-level shared verification."""
import os
import subprocess
import sys
from pathlib import Path
from _common import (C4_ROOT, checked_json, checked_bytes, parser, require, resolve_data_root,
                     save_json, sha, name, git_context, utc_now)
from _outer_group import VERSION, batch_groups, outer_group_cases, load_manifest, runs_directory, import_preflight
from _outer_admission_verification import (freeze_revision, verify_revision, verify_report,
                                           source_hashes, ORIGINAL_OUTER_FREEZE)


def run_group(root, group, output):
    script = C4_ROOT / 'scripts/run_outer_group_admissions_portable.py'
    log_path = output / (sha(group.encode())[:12] + '-admission.log')
    command = [sys.executable, '-u', str(script), '--data-root', str(root),
               '--group-id', group, '--execute-admissions']
    print('RUNNING:', group, 'admission; streaming progress below', flush=True)
    with log_path.open('x', encoding='utf-8') as log:
        environment = dict(os.environ, PYTHONIOENCODING='utf-8')
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1,
                              env=environment) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            returncode = process.wait()
        log.write('EXIT CODE: ' + str(returncode) + '\n')
    require(returncode == 0, 'Group failed; preserve claims and inspect ' + str(log_path))


def validate_resume(root, version, groups, n1_binding, original_hash):
    if version is None:
        return None
    directory = runs_directory(root) / 'outer_admission_batches' / version
    claim, digest = checked_json(directory, 'claim')
    require(claim['group_ids_in_fixed_order'] == groups and claim['n1_binding'] == n1_binding and
            claim['protocol_freeze_sha256'] == original_hash and claim['fold'] == 'outer_test' and
            claim['outer_admission_freeze_sha256'] == ORIGINAL_OUTER_FREEZE,
            'Previous batch identity/model/freeze differs. No adaptive group changes permitted.')
    return {'attempt': version, 'claim_sha256': digest,
            'final_report_present': (directory / 'batch_results.json').exists(),
            'failure_report_present': (directory / 'failure.json').exists()}


def main():
    p = parser(__doc__)
    p.add_argument('--batch-number', type=int, default=1)
    p.add_argument('--batch-version', type=name, default=None)
    p.add_argument('--resume-from', type=name, default=None)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--freeze-spec', action='store_true')
    actions.add_argument('--execute-admissions', action='store_true')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    if args.freeze_spec:
        require(args.resume_from is None and args.batch_version is None,
                '--freeze-spec defines the scheduling revision only; omit attempt arguments.')
        freeze_revision(root)
        return
    checkpoint, revision_hash = verify_revision(root, full=True)
    manifest, original_hash, manifest_hash = load_manifest(root)
    groups = batch_groups(manifest, args.batch_number)
    binding = checkpoint['binding']['n1_binding']
    previous = validate_resume(root, args.resume_from, groups, binding, original_hash)
    runs = runs_directory(root)
    for group in groups:
        present = (runs / 'group_admissions' / group / 'v1/group_admissions.json').exists()
        print('GROUP:', group, 'COMPLETE REPORT PRESENT' if present else 'PENDING', flush=True)
    print('FOLD: outer_test; FIXED BATCH:', args.batch_number, '/4; GROUPS: 4; CASES: 148')
    print('FULL SHARED CHECKPOINT VERIFICATION: before/after each group; record checksums retained')
    print('SAME ADMISSION RULE / SEEDS; ORIGINAL CASE RUNNER UNCHANGED; NO FUTURE ACCESS')
    if not args.execute_admissions:
        print('PLAN ONLY. Stop the old batch safely before executing this fresh attempt.')
        return
    attempt = args.batch_version or f'outer-admission-b{args.batch_number:02d}-portable-v2'
    output = runs / 'outer_admission_batches' / attempt
    require(not output.exists(), 'Attempt exists; preserve it and choose a fresh attempt version after diagnosis.')
    git = git_context()
    require(not git['working_tree_dirty'], 'Commit portable admission code before execution.')
    import_preflight()
    output.mkdir(parents=True, exist_ok=False)
    code = source_hashes()
    save_json(output / 'claim.json', {
        'purpose': 'portable_outer_admission_batch', 'created_at_utc': utc_now(), **git,
        'batch_number': args.batch_number, 'group_ids_in_fixed_order': groups, 'fold': 'outer_test',
        'n1_binding': binding, 'protocol_freeze_sha256': original_hash,
        'outer_admission_freeze_sha256': ORIGINAL_OUTER_FREEZE,
        'verification_scheduling_freeze_sha256': revision_hash, 'source_sha256': code,
        'scenario_manifest_sha256': manifest_hash, 'previous_attempt': previous,
        'outcome_based_selection': False, 'automatic_retries': False, 'release_authorized': False,
    })
    records = []
    current = None
    try:
        for current in groups:
            # A verified boundary before each group. No group is selected by its result.
            verify_revision(root, full=True)
            report_path = runs / 'group_admissions' / current / 'v1/group_admissions.json'
            if report_path.exists():
                print('REUSING COMPLETE GROUP WITHOUT TVAE REFIT:', current, flush=True)
            else:
                run_group(root, current, output)
            record = verify_report(root, current, outer_group_cases(manifest, current), original_hash, binding)
            verify_revision(root, full=True)
            records.append(record)
            print('ADMISSION COMPLETE:', current, 'ADMITTED:', record['admitted'], 'BLOCKED:', record['blocked'], flush=True)
        for path, digest in code.items():
            checked_bytes(C4_ROOT / path, digest)
        result = {
            'purpose': 'outer_admission_batch_results', 'created_at_utc': utc_now(), **git,
            'fold': 'outer_test', 'group_ids_in_fixed_order': groups, 'records': records,
            'admission_cases': 148, 'admitted': sum(r['admitted'] for r in records),
            'blocked': sum(r['blocked'] for r in records), 'n1_binding': binding,
            'protocol_freeze_sha256': original_hash, 'outer_admission_freeze_sha256': ORIGINAL_OUTER_FREEZE,
            'verification_scheduling_freeze_sha256': revision_hash, 'previous_attempt': previous,
            'N1_refitted': False, 'N2_calibrated': False,
            'candidate_pipeline_executed_by_batch': False, 'future_evaluation_executed_by_batch': False,
            'outcome_based_selection': False, 'release_authorized': False,
            'case_log_sha256': {f.name: sha(f.read_bytes()) for f in output.glob('*.log')},
        }
        digest = save_json(output / 'batch_results.json', result)
        print('OUTER ADMISSION BATCH COMPLETE: 4 GROUPS / 148 CASES')
        print('ADMITTED:', result['admitted'], 'BLOCKED:', result['blocked'])
        print('REPORT:', output / 'batch_results.json')
        print('REPORT SHA256:', digest)
    except Exception as error:
        save_json(output / 'failure.json', {'created_at_utc': utc_now(), 'group_id': current,
            'exception_type': type(error).__name__, 'message': str(error),
            'completed_group_records': records, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
