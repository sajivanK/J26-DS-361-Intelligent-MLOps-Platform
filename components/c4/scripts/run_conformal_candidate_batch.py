"""Run an explicit list of frozen conformal groups without outcome-based selection.

Admission failures remain records. Technical errors stop the entire batch.
There are no replacement seeds, automatic retries, or threshold changes.
"""
import subprocess
import sys
from pathlib import Path

from _common import (parser, name, resolve_data_root, checked_json, checked_bytes,
                     require, sha, save_json, utc_now, git_context, C4_ROOT)
from _conformal_group import conformal_group_cases, batch_groups, conformal_groups, verify_n1
from _conformal_pipeline import conformal_support
from _conformal_pipeline import select_admitted_cases
from run_conformal_group_admissions import existing_record

VERSION = 'adult-simulation-pp1-v2'


def planned_group_ids(manifest, requested):
    require(bool(requested) and len(requested) == len(set(requested)),
            'Requested groups must be nonempty and unique.')
    mapping = {g['group_id']: g for g in manifest['groups']}
    require(len(mapping) == len(manifest['groups']), 'Duplicate manifest groups.')
    for group in requested:
        require(group in mapping, 'Requested group is missing from the frozen manifest.')
        require(mapping[group]['split_role'] == 'conformal_calibration',
                'Only conformal-calibration groups are allowed in this batch.')
    return list(requested)


def verify_admissions(root, manifest, group, freeze_hash):
    run_root = root / 'c4_artifacts/benchmark_runs/adult' / VERSION
    report, checksum = checked_json(run_root / 'group_admissions' / group / 'v1', 'group_admissions')
    require(report['freeze_sha256'] == freeze_hash, 'Admission protocol differs.')
    admitted = select_admitted_cases(manifest, report, group)
    planned = conformal_group_cases(manifest, group)
    actual = [existing_record(root, row, freeze_hash) for row in planned]
    require(all(r is not None for r in actual) and report['records'] == actual,
            'Admission records do not match verified repair artifacts.')
    return report, checksum, admitted


def verified_meta(root, group, admission_hash, pipeline_hash):
    directory = root / 'c4_artifacts/meta_datasets/adult' / VERSION / (group + '-conformal-v1')
    metadata, checksum = checked_json(directory)
    require(metadata['group_id'] == group and metadata['fold'] == 'conformal_calibration' and
            metadata['group_admissions_sha256'] == admission_hash and
            metadata['group_pipeline_sha256'] == pipeline_hash, 'Meta-dataset links differ.')
    for filename, expected in metadata['files'].items():
        checked_bytes(directory / filename, expected)
    return metadata, checksum


def main():
    p = parser(__doc__)
    p.add_argument('--batch-number', type=int, default=None)
    p.add_argument('--all-groups', action='store_true', help='Complete all 20 frozen conformal groups in manifest order; reuse verified stages.')
    p.add_argument('--batch-version', type=name, default=None)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    require(not (args.all_groups and args.batch_number is not None), 'Use either --all-groups or --batch-number.')
    if not args.all_groups and args.batch_number is None:
        args.batch_number = 1
    root = resolve_data_root(args.data_root)
    n1_binding = verify_n1(root)
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult' / VERSION
    frozen, freeze_hash = checked_json(frozen_dir, 'freeze')
    manifest, manifest_hash = checked_json(frozen_dir, 'scenario_manifest')
    require(manifest_hash == frozen['scenario_manifest_sha256'] and
            frozen['full_execution_enabled'] is False, 'Frozen protocol/manifest scope differs.')
    require(freeze_hash == n1_binding['protocol_freeze_sha256'], 'Frozen N1 and conformal protocols differ.')
    all_reports = {}
    counts = []
    for gid in conformal_groups(manifest):
        report, digest = checked_json(root / 'c4_artifacts/benchmark_runs/adult' / VERSION / 'group_admissions' / gid / 'v1', 'group_admissions')
        selected = select_admitted_cases(manifest, report, gid)
        require(report['n1_binding'] == n1_binding and report['freeze_sha256'] == freeze_hash, 'Conformal admission scope differs.')
        counts.append(len(selected))
        all_reports[gid] = digest
    require(sum(counts) == 64 and sum(n > 0 for n in counts) == 17, 'Complete admission inventory differs from reported 64 cases / 17 nonempty groups.')
    support = conformal_support(17)
    print('ALL CONFORMAL ADMISSIONS VERIFIED: 20 GROUPS / 740 CASES / 64 ADMITTED / 676 BLOCKED')
    print('NONEMPTY GROUPS: 17; 95% RANK: 18; FINITE N2 ADJUSTMENT AVAILABLE: False')
    print('FROZEN FALLBACK REQUIRED: [-1, 1] MARGIN BOUNDS; AUTO_DEPLOY DISABLED')
    groups = planned_group_ids(manifest, conformal_groups(manifest) if args.all_groups else batch_groups(manifest, args.batch_number))
    run_root = root / 'c4_artifacts/benchmark_runs/adult' / VERSION
    scripts = [Path(__file__), *(C4_ROOT / 'scripts' / f for f in [
        'run_conformal_group_admissions.py', 'run_conformal_case_admission.py',
        '_conformal_group.py', '_common.py', '_training_group.py', '_conformal_pipeline.py', '_admitted_training.py',
        '_all_mechanism_transforms.py', '_portable_model_io.py', '_future_research.py', '_scenario_meta.py',
        'run_conformal_pipeline.py', 'validate_conformal_candidate.py',
        'freeze_conformal_case.py', 'evaluate_conformal_case_future.py',
        'build_conformal_group_meta_dataset.py'])]
    code_hashes = {str(path): sha(path.read_bytes()) for path in scripts}
    for group in groups:
        rows = conformal_group_cases(manifest, group)
        report_path = run_root / 'group_admissions' / group / 'v1/group_admissions.json'
        status = 'ADMISSION_REPORT_PRESENT' if report_path.exists() else 'PENDING_ADMISSION'
        print('GROUP:', group, 'FOLD: conformal_calibration CASES:', len(rows), status)
    print('FIXED REQUEST ORDER:', ', '.join(groups))
    print('TOTAL PLANNED CASES:', 37 * len(groups))
    print('FULL BENCHMARK EXECUTION: DISABLED; EXPLICIT CONFORMAL GROUPS ONLY')
    if not args.execute:
        print('PLAN ONLY. No candidate training, future-outcome reads, or artifacts written.')
        return
    attempt = args.batch_version or ('conformal-candidate-all20-v1' if args.all_groups else f'conformal-candidate-b{args.batch_number:02d}-v1')
    output = run_root / 'conformal_candidate_batches' / attempt
    require(not output.exists(), 'Batch attempt exists; preserve it. Use a fresh ID after diagnosing failures.')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'claim.json', {'purpose': 'fixed_explicit_conformal_candidate_batch', 'created_at_utc': utc_now(),
        **git_context(), 'group_ids_in_fixed_order': groups, 'n1_binding': n1_binding, 'all_conformal_admission_sha256': all_reports,
        'conformal_support': support, 'protocol_freeze_sha256': freeze_hash,
        'scenario_manifest_sha256': manifest_hash, 'source_sha256': code_hashes,
        'outcome_based_selection': False, 'automatic_retries': False, 'release_authorized': False})

    def stage(group, label, script, arguments):
        for path, expected in code_hashes.items():
            checked_bytes(path, expected)
        checked_bytes(frozen_dir / 'freeze.json', freeze_hash)
        for gid, expected in all_reports.items():
            checked_bytes(run_root / 'group_admissions' / gid / 'v1/group_admissions.json', expected)
        log = output / (sha(group.encode())[:12] + '-' + label + '.log')
        print('RUNNING:', group, label, flush=True)
        with log.open('x', encoding='utf-8') as file:
            result = subprocess.run([sys.executable, str(C4_ROOT / 'scripts' / script),
                '--data-root', str(root), '--group-id', group, *arguments],
                stdout=file, stderr=subprocess.STDOUT, text=True)
        require(result.returncode == 0, 'Stage failed; preserve artifacts and inspect ' + str(log))

    records = []
    current = None
    try:
        for current in groups:
            require(verify_n1(root) == n1_binding, 'N1 changed during conformal candidate execution.')
            admission_dir = run_root / 'group_admissions' / current / 'v1'
            if not (admission_dir / 'group_admissions.json').exists():
                raise ValueError('All admissions must already be complete; no repair generation is allowed here.')
            report, admission_hash, admitted = verify_admissions(root, manifest, current, freeze_hash)
            if not admitted:
                records.append({'group_id': current, 'admitted': 0, 'blocked': 37,
                    'admission_sha256': admission_hash, 'meta_rows': 0, 'status': 'ADMISSION_ONLY_ALL_BLOCKED',
                    'future_targets_imputed': False})
                print('ALL BLOCKED:', current, 'preserved; no candidate/future stages invoked.', flush=True)
                continue
            # This command verifies cached cases and preserves original stage artifacts.
            stage(current, 'pipeline', 'run_conformal_pipeline.py', ['--execute'])
            pipeline, pipeline_hash = checked_json(admission_dir.parent / 'conformal-pipeline-v1', 'group_pipeline')
            require(pipeline['group_id'] == current and pipeline['group_admissions_sha256'] == admission_hash and
                    pipeline['completed_scenarios'] == len(admitted), 'Group pipeline identity/count differs.')
            meta_dir = root / 'c4_artifacts/meta_datasets/adult' / VERSION / (current + '-conformal-v1')
            # Verify construction even on reuse; never overwrite a saved snapshot.
            stage(current, 'meta-verify', 'build_conformal_group_meta_dataset.py', [])
            if not meta_dir.exists():
                stage(current, 'meta-save', 'build_conformal_group_meta_dataset.py', ['--save'])
            metadata, meta_hash = verified_meta(root, current, admission_hash, pipeline_hash)
            require(metadata['rows'] == len(admitted), 'Meta-row count differs from admitted cases.')
            records.append({'group_id': current, 'admitted': len(admitted), 'blocked': 37-len(admitted),
                'admission_sha256': admission_hash, 'pipeline_sha256': pipeline_hash,
                'meta_metadata_sha256': meta_hash, 'meta_rows': metadata['rows'], 'status': 'COMPLETE'})
            print('COMPLETE:', current, 'META ROWS:', metadata['rows'], 'BLOCKED:', 37-len(admitted), flush=True)
        for path, expected in code_hashes.items():
            checked_bytes(path, expected)
        checked_bytes(frozen_dir / 'freeze.json', freeze_hash)
        require(verify_n1(root) == n1_binding, 'N1 changed before batch completion.')
        summary = {'purpose': 'fixed_conformal_candidate_batch_results', 'created_at_utc': utc_now(), **git_context(),
            'group_ids_in_fixed_order': groups, 'records': records, 'group_count': len(groups),
            'admission_cases': 37*len(groups), 'meta_rows': sum(r['meta_rows'] for r in records),
            'blocked_cases': sum(r['blocked'] for r in records), 'protocol_freeze_sha256': freeze_hash,
            'n1_binding': n1_binding, 'conformal_support': support, 'outcome_based_selection': False, 'N1_refitted': False, 'release_authorized': False,
            'case_log_sha256': {p.name: sha(p.read_bytes()) for p in output.glob('*.log')}}
        checksum = save_json(output / 'batch_results.json', summary)
        print('BATCH GROUPS COMPLETE:', len(groups), 'META ROWS:', summary['meta_rows'],
              'BLOCKED CASES RETAINED:', summary['blocked_cases'])
        print('REPORT:', output / 'batch_results.json')
        print('REPORT SHA256:', checksum)
        print('N1 REFITTED: False; N2 CALIBRATION / POLICY / OUTER-TEST / PRODUCTION RELEASE: NOT EXECUTED')
    except Exception as error:
        save_json(output / 'failure.json', {'created_at_utc': utc_now(), 'group_id': current,
            'exception_type': type(error).__name__, 'message': str(error),
            'completed_group_records': records, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
