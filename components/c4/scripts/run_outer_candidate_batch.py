"""Run all admitted outer cases with visible progress and a whole-fold future barrier.

Uses existing repairs. Candidate models are fitted; forecasting models, calibration
and release policies remain fixed. Completed outputs are verified and reused.
Incomplete stage claims are preserved and stop execution for investigation.
"""
import importlib.util
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from _common import (parser, resolve_data_root, require, checked_json, checked_bytes,
                     save_json, sha, utc_now, git_context, C4_ROOT, name)
from _outer_pipeline import (VERSION, inventory, runs, verified_phase, verify_execution_spec,
                             verify_freeze_dependencies, verify_pre_release_registry)


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, contents):
        for stream in self.streams:
            stream.write(contents)
            stream.flush()
        return len(contents)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def invoke(script, argv):
    path = C4_ROOT/'scripts'/script
    spec = importlib.util.spec_from_file_location('_outer_stage_'+path.stem, path)
    require(spec is not None and spec.loader is not None, 'Cannot load outer stage.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = sys.argv
    try:
        sys.argv = [str(path), *argv]
        module.main()
    finally:
        sys.argv = previous


def validated_case(root, row):
    directory = runs(root)/row['scenario_id']/'scoped_candidate_validation-v1'
    if not directory.exists():
        return None
    require((directory/'validation.json').is_file(),
            'Incomplete candidate claim; preserve it and inspect failure.json: '+str(directory))
    result, digest = checked_json(directory, 'validation')
    metadata, metadata_hash = checked_json(directory)
    require(result['scenario']['scenario_id'] == row['scenario_id'] and
            result['scenario']['split_role'] == 'outer_test' and
            result['candidate_metadata_sha256'] == metadata_hash and
            result['candidate_training_executed'] is True and result['future_evaluation_executed'] is False,
            'Candidate validation binding/state differs.')
    checked_bytes(directory/'candidate.ubj', metadata['model_sha256'])
    from _portable_model_io import load_candidate
    load_candidate(directory)
    evidence, evidence_hash = checked_json(Path(result['pre_release_directory']), 'evidence')
    require(evidence_hash == result['pre_release_evidence_sha256'], 'Pre-release evidence differs.')
    claim, _ = checked_json(directory, 'claim')
    for relative, expected in claim['source_sha256'].items():
        checked_bytes(C4_ROOT/relative, expected)
    return digest


def frozen_case(root, row):
    directory = runs(root)/row['scenario_id']/'pre_release_freeze-v1'
    if not directory.exists():
        return None
    require((directory/'freeze.json').is_file(), 'Incomplete pre-release claim; preserve: '+str(directory))
    snapshot, digest = checked_json(directory, 'freeze')
    verify_freeze_dependencies(snapshot)
    require(snapshot['scenario']['scenario_id'] == row['scenario_id'], 'Frozen scenario differs.')
    return digest


def completed_case(root, row):
    directory = runs(root)/row['scenario_id']/'future_research-v1'
    if not directory.exists():
        return None
    require((directory/'outcomes.json').is_file(), 'Incomplete future claim; preserve: '+str(directory))
    outcome, digest = checked_json(directory, 'outcomes')
    registry, registry_hash = verify_pre_release_registry(root)
    require(outcome['scenario']['scenario_id'] == row['scenario_id'] and
            outcome['scenario']['split_role'] == 'outer_test' and
            outcome['decision_freeze_sha256'] == registry['scenario_freeze_sha256'][row['scenario_id']] and
            outcome['outer_pre_release_registry_sha256'] == registry_hash and
            outcome['future_evaluation_executed'] is True and outcome['training_executed'] is False and
            outcome['controller_received_future_targets'] is False, 'Future result binding/state differs.')
    for filename, expected in outcome['files'].items():
        checked_bytes(directory/filename, expected)
    return digest


def main():
    p = parser(__doc__)
    p.add_argument('--all-groups', action='store_true', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--batch-version', type=name, default='outer-candidate-all16-v1')
    args = p.parse_args()
    root = resolve_data_root(args.data_root)
    frozen, execution_hash = verify_execution_spec(root, full=True)
    _, groups, reports, selected = inventory(root)
    print('OUTER: 16 GROUPS / 592 ADMISSIONS / 56 ADMITTED / 536 BLOCKED', flush=True)
    print('PHASES: candidate -> all-method decision freeze -> whole-fold registry -> future', flush=True)
    print('NO TVAE / N1-N2 REFITTING / POLICY RESELECTION / LIVE DEPLOYMENT', flush=True)
    if not args.execute:
        print('PLAN ONLY. Use --execute. No candidate training or future reads executed.')
        return
    output = root/'c4_artifacts/benchmark_runs/adult'/VERSION/'outer_candidate_batches'/args.batch_version
    require(not output.exists(), 'Batch version exists; preserve it. For a verified resume use a fresh --batch-version.')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'claim.json', {'purpose':'fixed_outer_candidate_batch', 'created_at_utc':utc_now(),
        **git_context(), 'outer_execution_freeze_sha256':execution_hash, 'groups':groups,
        'scenarios':[r['scenario_id'] for r in selected]})
    current, phase = None, 'starting'
    logs = {}

    def stage(script, argv, label):
        key = sha((str(current)+'/'+label).encode())[:16]+'-'+label+'.log'
        path = output/key
        with path.open('x', encoding='utf-8') as stream:
            with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                invoke(script, ['--data-root', str(root), *argv])
        logs[key] = sha(path.read_bytes())

    try:
        validations, freezes, outcomes = {}, {}, {}
        for phase, script, flag, checker, results in (
            ('candidate','validate_outer_candidate.py','--execute',validated_case,validations),
            ('freeze','freeze_outer_case.py','--freeze',frozen_case,freezes)):
            with verified_phase(root, phase):
                for number, row in enumerate(selected, 1):
                    current = row['scenario_id']
                    print(f'{phase.upper()} {number}/56: {current}', flush=True)
                    digest = checker(root, row)
                    if digest is None:
                        stage(script, ['--scenario-id', current, flag], phase)
                        digest = checker(root, row)
                        require(digest is not None, 'Stage did not produce verified completion.')
                    else:
                        print('REUSE VERIFIED:', current, flush=True)
                    results[current] = digest
                if phase == 'freeze':
                    current = 'all-56-registry'
                    stage('freeze_outer_pre_release_registry.py', ['--freeze'], 'registry')
        phase = 'future'
        with verified_phase(root, phase):
            verify_pre_release_registry(root)
            for number, row in enumerate(selected, 1):
                current = row['scenario_id']
                print(f'FUTURE {number}/56: {current}', flush=True)
                digest = completed_case(root, row)
                if digest is None:
                    stage('evaluate_outer_case_future.py', ['--scenario-id', current, '--execute'], 'future')
                    digest = completed_case(root, row)
                    require(digest is not None, 'Future evaluator did not complete.')
                outcomes[current] = digest
        _, ending_hash = verify_execution_spec(root, full=True)
        require(ending_hash == execution_hash, 'Execution checkpoint changed.')
        records = [{'scenario_id':r['scenario_id'], 'group_id':r['group_id'], 'status':'COMPLETE',
                    'validation_sha256':validations[r['scenario_id']], 'decision_freeze_sha256':freezes[r['scenario_id']],
                    'outcomes_sha256':outcomes[r['scenario_id']]} for r in selected]
        digest = save_json(output/'batch_results.json', {'purpose':'completed_frozen_outer_candidate_batch',
            'created_at_utc':utc_now(), **git_context(), 'outer_execution_freeze_sha256':execution_hash,
            'group_ids_in_fixed_order':groups, 'assigned_groups':16, 'admission_records':592,
            'completed_scenarios':56, 'blocked':536, 'records':records, 'log_sha256':logs,
            'N1_refitted':False, 'N2_refitted':False, 'policy_reselected':False, 'production_release_authorized':False})
        print('COMPLETE: 56 OUTER FUTURES; 536 BLOCKED RETAINED; 16 GROUPS', flush=True)
        print('REPORT:', output/'batch_results.json')
        print('REPORT SHA256:', digest)
        print('NEXT: audit_outer_snapshot.py (plan, then --save).')
    except Exception as error:
        save_json(output/'failure.json', {'scenario_id':current, 'phase':phase,
            'exception_type':type(error).__name__, 'message':str(error),
            'preserve_artifacts':True, 'production_release_authorized':False})
        raise


if __name__ == '__main__':
    main()
