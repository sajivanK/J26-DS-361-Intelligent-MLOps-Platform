"""Frozen outer execution scope; no fitting/calibration/policy selection here."""
from contextlib import contextmanager
import io
import json
from pathlib import Path

from _common import C4_ROOT, checked_json, checked_bytes, require, sha
from _outer_group import VERSION, outer_groups, outer_group_cases, load_manifest, current_bindings, source_hashes as admission_sources

AUDIT_HASH = 'c4003cc8331495a8411ab0f9440bd794c7d1da9da52351b7be5623f11de8560b'
OUTER_ADMISSION_HASH = 'd8815e1cc433daa45963ede0f9075e7d2ec4038dd2b9f6785cf38cd7ffa859b9'
ABLATION_RUN_HASH = 'ab1ef7b2cfebbe762a3037f0dca883f43ef33d168d23824cd829e95528073385'
PREPARED_HASH = '3b51935a3f5dbe3a359edabe113c872d433699e11bfa590f4ee747da3ff917eb'
N1_SPEC_HASH = '437ae92c313020697a35e8b1372442d327df22c9189aa6392afaeb2e354d98c3'
N1_RUN_HASH = 'c9bcedd6f74c47d856eead5a3dcfc5edb37605312318cebc695cf12fe0d05545'
POLICY_SPEC_HASH = '113e2c5cd4007301ecddc2fd8bf3a226c1b2e3f1acd108c49a3b2dc389b2d505'
SPEC_VERSION = 'outer-execution-spec-v1'
SOURCE_FILES = ('scripts/_outer_pipeline.py', 'scripts/freeze_outer_execution_protocol.py',
    'scripts/validate_outer_candidate.py', 'scripts/freeze_outer_case.py',
    'scripts/freeze_outer_pre_release_registry.py', 'scripts/evaluate_outer_case_future.py',
    'scripts/run_outer_candidate_batch.py', 'scripts/audit_outer_snapshot.py')
_ACTIVE = None


def runs(root):
    return Path(root) / 'c4_artifacts/benchmark_runs/adult' / VERSION


def specification_directory(root):
    return Path(root) / 'c4_artifacts/outer_protocols/adult' / VERSION / SPEC_VERSION


def registry_directory(root):
    return Path(root) / 'c4_artifacts/outer_pre_release/adult' / VERSION / 'outer-all56-decision-freeze-v1'


def source_hashes():
    return {p: sha((C4_ROOT / p).read_bytes()) for p in SOURCE_FILES}


def audit_checkpoint(root):
    directory = Path(root) / 'c4_artifacts/outer_admission_audits/adult' / VERSION / 'outer-all16-admission-audit-portable-v2'
    md, digest = checked_json(directory)
    require(digest == AUDIT_HASH and md['assigned_groups'] == 16 and md['admission_cases'] == 592 and
            md['admitted'] == 56 and md['blocked'] == 536 and md['nonempty_groups'] == 14 and
            md['all_blocked_groups'] == 2 and md['outer_future_evaluated'] is False,
            'Completed outer admission checkpoint differs.')
    for filename, expected in md['files'].items():
        checked_bytes(directory / filename, expected)
    return md


def inventory(root):
    """Select all admitted scenarios from the frozen manifest/audited reports only."""
    manifest, original_hash, manifest_hash = load_manifest(root)
    audit = audit_checkpoint(root)
    groups = outer_groups(manifest)
    require(audit['groups'] == groups, 'Audited outer group order differs.')
    reports, selected = {}, []
    for group in groups:
        report, digest = checked_json(runs(root) / 'group_admissions' / group / 'v1', 'group_admissions')
        require(digest == audit['group_admission_sha256'][group] and report['cases'] == 37 and
                report['group_id'] == group and report['freeze_sha256'] == original_hash and
                report['outer_admission_freeze_sha256'] == OUTER_ADMISSION_HASH,
                'Outer group admission binding differs.')
        require(report['n1_binding'] == audit['binding']['n1_binding'], 'Outer report N1 differs.')
        planned = outer_group_cases(manifest, group)
        by_id = {r['scenario_id']: r for r in report['records']}
        require(len(by_id) == len(report['records']) == 37 and set(by_id) == {r['scenario_id'] for r in planned},
                'Outer admission inventory differs.')
        chosen = []
        for row in planned:
            record = by_id[row['scenario_id']]
            fields = ('group_id','split_role','mechanism','magnitude','diagnosis_condition','repair_condition','is_no_drift_control')
            require(all(record[k] == row[k] for k in fields), 'Outer record differs from original manifest.')
            require(record['simulation_admission'] in {'READY_FOR_SCOPED_CANDIDATE_VALIDATION','BLOCKED_ESCALATE'},
                    'Unknown outer admission state.')
            if record['simulation_admission'] == 'READY_FOR_SCOPED_CANDIDATE_VALIDATION':
                require(record['screen_status'] == 'PASS', 'Admitted scenario lacks a passing screen.')
                chosen.append(record)
        require(len(chosen) == report['scoped_admissible_cases'] and report['blocked_cases'] == 37-len(chosen),
                'Outer admission counts differ.')
        reports[group] = {'report': report, 'sha256': digest}
        selected.extend(sorted(chosen, key=lambda r:r['scenario_id']))
    require(len(selected) == 56 and len({r['scenario_id'] for r in selected}) == 56,
            'All and only the 56 admitted outer scenarios must be executed.')
    return manifest, groups, reports, selected


def shared_binding(root):
    """Full development verification, permitted after admission/candidate stages.

    Do not call the older admission-only verifier here: it intentionally rejects
    candidate and future directories. Historical admission authorization remains
    unchanged and is read as a checkpoint, not broadened in place.
    """
    print('VERIFYING FROZEN DEVELOPMENT / AUDIT CHECKPOINTS...', flush=True)
    current = current_bindings(root)
    from _outer_group import protocol_directory
    admission, digest = checked_json(protocol_directory(root), 'freeze')
    require(digest == OUTER_ADMISSION_HASH and admission['binding'] == current and
            admission['source_sha256'] == admission_sources(), 'Original outer admission checkpoint differs.')
    audit = audit_checkpoint(root)
    from _outer_admission_verification import revision_directory, source_hashes as revision_sources
    revision, revision_hash = checked_json(revision_directory(root), 'freeze')
    require(revision_hash == audit['verification_scheduling_freeze_sha256'] and
            revision['original_binding'] == current and revision['source_sha256'] == revision_sources(),
            'Portable admission verification checkpoint differs.')
    models = Path(root) / 'c4_artifacts/comparison_models/adult' / VERSION / 'n1-feature-ablations-v1/fit-v1'
    md, model_hash = checked_json(models)
    require(model_hash == ABLATION_RUN_HASH and md['model_reload_prediction_equality'] is True,
            'A1-A3 frozen models differ.')
    for filename, expected in md['files'].items():
        checked_bytes(models / filename, expected)
    manifest, groups, reports, selected = inventory(root)
    print('DEVELOPMENT / AUDIT / 16 GROUPS / 56 ADMITTED CASES: PASS', flush=True)
    return {'development': current, 'outer_admission_audit_metadata_sha256': AUDIT_HASH,
            'outer_admission_freeze_sha256': OUTER_ADMISSION_HASH,
            'verification_scheduling_freeze_sha256': revision_hash,
            'ablation_run_metadata_sha256': model_hash,
            'n1_binding': current['n1_binding'], 'group_ids_in_fixed_order': groups,
            'admitted_scenario_ids_in_fixed_order': [r['scenario_id'] for r in selected],
            'group_admission_sha256': {g:reports[g]['sha256'] for g in groups}}


def verify_execution_spec(root, *, full=False):
    frozen, digest = checked_json(specification_directory(root), 'freeze')
    require(frozen['purpose'] == 'frozen_outer_candidate_decision_future_execution' and
            frozen['source_sha256'] == source_hashes() and frozen['assigned_groups'] == 16 and
            frozen['admission_records'] == 592 and frozen['admitted_scenarios'] == 56 and
            frozen['pre_release_registry_required_before_future'] is True and
            frozen['new_forecaster_or_policy_fitting_allowed'] is False and
            frozen['production_release_authorized'] is False, 'Outer execution definition/source differs.')
    if full:
        require(frozen['binding'] == shared_binding(root), 'Outer execution model/audit bindings changed.')
    else:
        # Small immutable bindings/code are checked on every case; complete shared
        # bytes are verified at phase boundaries, not repeatedly for every record.
        audit_checkpoint(root)
        _, original_hash, manifest_hash = load_manifest(root)
        require(original_hash == frozen['binding']['n1_binding']['protocol_freeze_sha256'] and
                manifest_hash == frozen['binding']['development']['scenario_manifest_sha256'],
                'Outer original manifest binding differs.')
    return frozen, digest


@contextmanager
def verified_phase(root, phase):
    global _ACTIVE
    require(_ACTIVE is None and phase in {'candidate','freeze','future','audit'}, 'Invalid/nested execution phase.')
    frozen, digest = verify_execution_spec(root, full=True)
    _ACTIVE = {'root': Path(root), 'phase': phase, 'specification': frozen, 'sha256': digest, 'models': None}
    try:
        yield _ACTIVE
        ending, ending_hash = verify_execution_spec(root, full=True)
        require(ending_hash == digest and ending == frozen, 'Execution specification changed during phase.')
    finally:
        _ACTIVE = None


def execution_binding(root, phase=None):
    if _ACTIVE is None:
        frozen, _ = verify_execution_spec(root, full=True)
    else:
        require(Path(root) == _ACTIVE['root'] and (phase is None or phase == _ACTIVE['phase']),
                'Execution phase/root differs.')
        frozen, digest = verify_execution_spec(root)
        require(digest == _ACTIVE['sha256'], 'Execution definition changed during phase.')
    return frozen['binding']


def group_context(root, scenario_id):
    binding = execution_binding(root)
    manifest, groups, reports, selected = inventory(root)
    matches = [r for r in selected if r['scenario_id'] == scenario_id]
    require(len(matches) == 1, 'Only an admitted original outer scenario may execute.')
    record = matches[0]
    admission, admission_hash = checked_json(runs(root) / scenario_id, 'admission')
    planned = next(r for r in manifest['scenarios'] if r['scenario_id'] == scenario_id)
    require(admission_hash == record['admission_sha256'] and admission['scenario'] == planned and
            admission['n1_binding'] == binding['n1_binding'] and
            admission['outer_admission_freeze_sha256'] == OUTER_ADMISSION_HASH,
            'Outer admission scenario/model binding differs.')
    from _outer_admission_verification import existing_record
    require(existing_record(root, planned, binding['n1_binding']['protocol_freeze_sha256'], binding['n1_binding']) == record,
            'Outer case admission/provenance failed recomputation.')
    report = reports[record['group_id']]
    return report['report'], report['sha256'], record


def verify_freeze_dependencies(snapshot):
    require(snapshot.get('complete_decision_evidence_frozen') is True and
            snapshot.get('future_evaluation_executed') is False and
            snapshot.get('scope') == 'one_admitted_outer_scenario_under_simulated_upstream_evidence' and
            snapshot['scenario']['split_role'] == 'outer_test', 'Wrong outer pre-future freeze scope.')
    require(snapshot['n1_binding']['specification_sha256'] == N1_SPEC_HASH and
            snapshot['n1_binding']['run_metadata_sha256'] == N1_RUN_HASH and
            snapshot['policy_specification_sha256'] == POLICY_SPEC_HASH, 'Frozen forecasting/policy identity differs.')
    dependencies = snapshot['dependencies']
    require(dependencies.get(snapshot['comparison_forecasts_path']) == snapshot['comparison_forecasts_sha256'] and
            dependencies.get(snapshot['n1_forecasts_path']) == snapshot['n1_forecasts_sha256'],
            'Missing frozen outer predictions/decisions.')
    n1_directory = Path(snapshot['n1_run_directory'])
    n1_metadata = json.loads(checked_bytes(n1_directory/'metadata.json', N1_RUN_HASH))
    required = {str(n1_directory/'metadata.json'):N1_RUN_HASH,
                str(n1_directory.parent/'specification.json'):N1_SPEC_HASH}
    required.update({str(n1_directory/key):value for key,value in n1_metadata['files'].items()})
    require(all(dependencies.get(path) == digest for path,digest in required.items()),
            'Outer freeze must pin all original N1 artifacts.')
    require(PREPARED_HASH in dependencies.values() and ABLATION_RUN_HASH in dependencies.values() and
            POLICY_SPEC_HASH in dependencies.values() and snapshot['outer_execution_freeze_sha256'] in dependencies.values(),
            'Outer freeze is missing comparison/policy/execution dependencies.')
    for path, digest in dependencies.items():
        checked_bytes(path, digest)


def model_runtime(root, remember):
    import joblib
    from _controller_artifacts import load_controller
    directory = Path(root) / 'c4_artifacts/comparison_calibrations/adult' / VERSION / 'comparison-calibration-v1/prepared-v1'
    md, md_hash = checked_json(directory)
    require(md_hash == PREPARED_HASH, 'Prepared comparison runtime differs.')
    runtime, runtime_hash = checked_json(directory, 'comparison_runtime')
    require(runtime_hash == md['files']['comparison_runtime.json'], 'Runtime differs from pinned comparison metadata.')
    remember(directory / 'metadata.json', md_hash)
    remember(directory / 'comparison_runtime.json', runtime_hash)
    controller, controller_binding, controller_hash = load_controller(root) if _ACTIVE is None or _ACTIVE['models'] is None else _ACTIVE['models']['controller']
    models = {'ORIGINAL': controller.models['lightgbm']}
    base = Path(root) / 'c4_artifacts/comparison_models/adult' / VERSION / 'n1-feature-ablations-v1/fit-v1'
    ablation, ablation_hash = checked_json(base)
    require(ablation_hash == ABLATION_RUN_HASH, 'Ablation model checkpoint differs.')
    remember(base / 'metadata.json', ablation_hash)
    for variant in ('A1','A2','A3'):
        path = base / (variant+'.joblib')
        contents = remember(path, ablation['files'][variant+'.joblib'])
        models[variant] = joblib.load(io.BytesIO(contents))
    if _ACTIVE is not None and _ACTIVE['models'] is None:
        _ACTIVE['models'] = {'controller': (controller, controller_binding, controller_hash)}
    return models, runtime, controller


def comparison_payload(root, scenario_id, feature, gates, remember):
    import numpy as np
    import pandas as pd
    from c4.forecasting.benchmark_feature_ablation import project_features
    from c4.evaluation.comparison_decisions import METHOD_IDS, decide_comparison
    models, runtime, controller = model_runtime(root, remember)
    frame = pd.DataFrame([feature], index=pd.Index([scenario_id], name='scenario_id'))
    forecasts = {}
    for variant, model in models.items():
        projected = frame if variant == 'ORIGINAL' else project_features(frame, variant, frame.columns)
        raw = model.predict(projected).iloc[0].to_dict()
        forecasts[variant] = {k:bool(v) if isinstance(v,(bool,np.bool_)) else float(v) for k,v in raw.items()}
    decisions = {}
    for method in METHOD_IDS:
        source = method if method in {'A1','A2','A3'} else 'ORIGINAL'
        decisions[method] = decide_comparison(method, feature, gates, forecasts[source],
            runtime['calibrations'], runtime['selected_policies'], runtime['unsupported_contexts'])
    actual = controller.decide(feature, gates, evidence_id=scenario_id)
    require(actual['decision'] == decisions['PROPOSED']['decision'], 'Primary controller/comparison decision differs.')
    require(decisions['A8'] == {**decisions['B8'], 'comparison':'A8'}, 'A8/B8 declared equivalence differs.')
    return {'purpose':'all_frozen_comparison_decisions_before_outer_future', 'scenario_id':scenario_id,
            'feature_columns':list(feature), 'decision_time_features':feature,
            'explicit_hard_gates':gates, 'forecasts':forecasts, 'decisions':decisions,
            'runtime':runtime, 'controller_output':actual,
            'future_targets_included':False, 'production_release_authorized':False}


def registry_definition(root, *, forbid_future):
    binding = execution_binding(root)
    manifest, groups, reports, selected = inventory(root)
    if forbid_future:
        require(not any((runs(root)/r['scenario_id']/'future_research-v1').exists()
                        for r in manifest['scenarios'] if r['split_role']=='outer_test'),
                'Outer future stage exists before the whole-fold decision registry.')
    links, comparison_links = {}, {}
    for row in selected:
        directory = runs(root) / row['scenario_id'] / 'pre_release_freeze-v1'
        snapshot, digest = checked_json(directory, 'freeze')
        verify_freeze_dependencies(snapshot)
        require(snapshot['scenario']['scenario_id'] == row['scenario_id'] and
                snapshot['outer_execution_freeze_sha256'] == checked_json(specification_directory(root),'freeze')[1],
                'Outer decision freeze specification differs.')
        payload, payload_hash = checked_json(directory, 'comparison_forecasts')
        from c4.evaluation.comparison_decisions import METHOD_IDS
        require(payload_hash == snapshot['comparison_forecasts_sha256'] and
                set(payload['decisions']) == set(METHOD_IDS) and payload['future_targets_included'] is False,
                'Incomplete all-method decision freeze.')
        links[row['scenario_id']] = digest
        comparison_links[row['scenario_id']] = payload_hash
    return {'purpose':'all_56_outer_decisions_frozen_before_any_outer_future',
            'outer_execution_freeze_sha256':checked_json(specification_directory(root),'freeze')[1],
            'group_ids_in_fixed_order':groups, 'admission_records':592, 'admitted_scenarios':56, 'blocked':536,
            'scenario_freeze_sha256':links, 'comparison_forecast_sha256':comparison_links,
            'group_admission_sha256':binding['group_admission_sha256'],
            'future_targets_included':False, 'policy_context_support_reestimated':False,
            'production_release_authorized':False}


def verify_pre_release_registry(root):
    saved, digest = checked_json(registry_directory(root), 'registry')
    # All 56 immutable markers are checked; each future case additionally checks
    # its complete own dependency set. Do not reread all 56 model dependencies per case.
    require(saved['definition']['outer_execution_freeze_sha256'] == checked_json(specification_directory(root),'freeze')[1],
            'Outer whole-fold registry execution binding differs.')
    definition = saved['definition']
    expected = execution_binding(root)['admitted_scenario_ids_in_fixed_order']
    require(set(definition['scenario_freeze_sha256']) == set(expected) and
            set(definition['comparison_forecast_sha256']) == set(expected) and
            definition['admitted_scenarios'] == 56 and definition['admission_records'] == 592 and
            definition['blocked'] == 536 and definition['group_ids_in_fixed_order'] == execution_binding(root)['group_ids_in_fixed_order'] and
            definition['policy_context_support_reestimated'] is False and definition['future_targets_included'] is False,
            'Whole-fold registry inventory differs.')
    for scenario in expected:
        directory = runs(root) / scenario / 'pre_release_freeze-v1'
        checked_bytes(directory/'freeze.json',definition['scenario_freeze_sha256'][scenario])
        checked_bytes(directory/'comparison_forecasts.json',definition['comparison_forecast_sha256'][scenario])
    return definition, digest


def require_no_outer_future(root):
    """A missing candidate or decision freeze cannot be created after futures start."""
    manifest, _, _ = load_manifest(root)
    require(not any((runs(root)/r['scenario_id']/'future_research-v1').exists()
                    for r in manifest['scenarios'] if r['split_role']=='outer_test'),
            'New outer candidates/decision freezes are forbidden after any outer future claim.')
