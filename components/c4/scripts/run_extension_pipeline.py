"""Complete every admitted case in one group, preserving the completed first case."""
from _extension_group import EXTENSION_HASH, load_extension
import subprocess
import sys
from pathlib import Path

from _common import parser, resolve_data_root, checked_json, checked_bytes, require, save_json, sha, utc_now, git_context, C4_ROOT, name
from _extension_pipeline import select_admitted_cases, group_context
from _extension_pipeline import verify_freeze_dependencies


def completed_case(root, record):
    run = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / record['scenario_id']
    if not (run / 'future_research-v1/outcomes.json').exists(): return None
    snapshot, freeze_hash = checked_json(run / 'pre_release_freeze-v1', 'freeze')
    verify_freeze_dependencies(snapshot)
    outcomes, outcomes_hash = checked_json(run / 'future_research-v1', 'outcomes')
    require(outcomes['decision_freeze_sha256'] == freeze_hash and
            outcomes['scenario']['scenario_id'] == record['scenario_id'] and
            outcomes['scenario']['group_id'] == record['group_id'] and outcomes['scenario']['split_role'] == 'conformal_extension',
            'Completed future scenario links differ.')
    for filename, checksum in outcomes['files'].items():
        checked_bytes(run / 'future_research-v1' / filename, checksum)
    return {'scenario_id': record['scenario_id'], 'group_id': record['group_id'], 'split_role': 'conformal_extension',
            'decision_freeze_sha256': freeze_hash, 'outcomes_sha256': outcomes_hash,
            'admission_sha256': record['admission_sha256']}


def main():
    p = parser(__doc__)
    p.add_argument('--group-id', type=name, required=True)
    p.add_argument('--execute', action='store_true', help='Complete all admitted cases, with no outcome-based filtering.')
    p.add_argument('--pipeline-version', type=name, default='extension-pipeline-v1', help='Fresh attempt ID; preserve failed attempts.')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    GROUP = args.group_id
    group_dir = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2/group_admissions' / GROUP / 'v1'
    report, report_hash = checked_json(group_dir, 'group_admissions')
    frozen_dir = root / 'c4_artifacts/benchmark_protocols/adult/adult-simulation-pp1-v2'
    frozen, frozen_hash, protocol, manifest, manifest_hash, extension_hash = load_extension(root)
    require(report['freeze_sha256'] == frozen_hash and report.get('extension_freeze_sha256') == EXTENSION_HASH, 'Extension group links differ.')
    rows = select_admitted_cases(manifest, report, GROUP)
    require(rows, 'No admitted scenarios are available.')
    cached = {}
    for row in rows:
        group_context(root, row['scenario_id'])
        result = completed_case(root, row)
        if result is not None: cached[row['scenario_id']] = result
    print('GROUP:', GROUP, 'FOLD: conformal_extension')
    print('ADMITTED CASES:', len(rows), 'COMPLETED VERIFIED CASES:', len(cached), 'PENDING CASES:', len(rows)-len(cached))
    print('BLOCKED CASES RETAINED:', report['blocked_cases'])
    output = group_dir.parent / args.pipeline_version
    if (output / 'group_pipeline.json').exists():
        saved, saved_hash = checked_json(output, 'group_pipeline')
        require(saved['group_admissions_sha256'] == report_hash and
                saved['records'] == [cached[r['scenario_id']] for r in rows], 'Completed group pipeline differs.')
        print('GROUP PIPELINE ALREADY COMPLETE:', output / 'group_pipeline.json')
        print('REPORT SHA256:', saved_hash)
        return
    require(not output.exists(), 'Incomplete pipeline claim exists; preserve logs and failure evidence.')
    if not args.execute:
        for row in rows: print('REUSE' if row['scenario_id'] in cached else 'PENDING', row['scenario_id'])
        print('PLAN ONLY. Use --execute to complete all pending admitted cases in fixed order.')
        return
    output.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), *(C4_ROOT / 'scripts' / name for name in
        ['_extension_pipeline.py', '_admitted_training.py', 'validate_extension_candidate.py', 'freeze_extension_case.py',
         'evaluate_extension_case_future.py', '_future_research.py', '_all_mechanism_transforms.py', '_portable_model_io.py'])]
    code_hashes = {path.name: sha(path.read_bytes()) for path in code_paths}
    save_json(output / 'claim.json', {'purpose': 'all_admitted_extension_group_pipeline_claim',
        'created_at_utc': utc_now(), **git_context(), 'extension_freeze_sha256': EXTENSION_HASH, 'group_admissions_sha256': report_hash,
        'scenario_ids': [r['scenario_id'] for r in rows], 'reused_scenario_ids': list(cached),
        'source_sha256': code_hashes, 'log_filename_scheme': 'first_12_SHA256_hex_of_scenario_id-stage.log',
        'outcome_based_selection': False, 'release_authorized': False})
    current = None
    try:
        for row in rows:
            current = row['scenario_id']
            if current in cached:
                print('REUSED COMPLETE SCENARIO:', current, flush=True)
                continue
            run = root / 'c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2' / current
            steps = [('validate_extension_candidate.py', '--execute', 'scoped_candidate_validation-v1', 'validation'),
                     ('freeze_extension_case.py', '--freeze', 'pre_release_freeze-v1', 'freeze'),
                     ('evaluate_extension_case_future.py', '--execute', 'future_research-v1', 'outcomes')]
            for script, flag, directory, stem in steps:
                stage_dir = run / directory
                if (stage_dir / (stem + '.json')).exists():
                    checked_json(stage_dir, stem)
                    print('REUSE VERIFIED STAGE:', current, stem, flush=True)
                    continue
                require(not stage_dir.exists(), 'Incomplete scenario stage exists: ' + str(stage_dir))
                print('RUNNING:', current, stem, flush=True)
                log_path = output / (sha(current.encode('utf-8'))[:12] + '-' + stem + '.log')
                with log_path.open('x', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, str(C4_ROOT / 'scripts' / script),
                        '--data-root', str(root), '--scenario-id', current, flag],
                        stdout=log, stderr=subprocess.STDOUT, text=True)
                require(result.returncode == 0, 'Stage failed; inspect ' + str(log_path))
            cached[current] = completed_case(root, row)
            require(cached[current] is not None, 'Scenario did not produce verified outcomes.')
            print('COMPLETE:', current, flush=True)
        for path in code_paths: checked_bytes(path, code_hashes[path.name])
        checked_bytes(group_dir / 'group_admissions.json', report_hash)
        result = {'purpose': 'completed_admitted_extension_group_pipeline', 'created_at_utc': utc_now(), **git_context(),
            'group_id': GROUP, 'extension_freeze_sha256': EXTENSION_HASH, 'group_admissions_sha256': report_hash, 'completed_scenarios': len(rows),
            'reused_scenarios': len(rows)-sum(1 for p in output.glob('*-outcomes.log')),
            'records': [cached[r['scenario_id']] for r in rows], 'blocked_cases': report['blocked_cases'],
            'negative_targets_retained': True, 'outcome_based_selection': False, 'release_authorized': False,
            'N1_refitted': False, 'scope_note': 'One extension-calibration group; no N1 fitting, policy or outer-test cases executed.',
            'case_log_sha256': {path.name: sha(path.read_bytes()) for path in output.glob('*.log')}}
        checksum = save_json(output / 'group_pipeline.json', result)
        print('COMPLETED VERIFIED SCENARIOS:', len(rows))
        print('GROUP COUNT: 1; BLOCKED CASES RETAINED:', report['blocked_cases'])
        print('REPORT:', output / 'group_pipeline.json')
        print('REPORT SHA256:', checksum)
        print('RELEASE AUTHORIZED: False; N1 REFITTED: False; N2 CALIBRATION: PENDING')
    except Exception as error:
        save_json(output / 'failure.json', {'created_at_utc': utc_now(), 'scenario_id': current,
            'exception_type': type(error).__name__, 'message': str(error), 'release_authorized': False})
        raise


if __name__ == '__main__':
    main()
