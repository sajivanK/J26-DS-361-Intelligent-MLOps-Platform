"""Outer admission authorization bound to completed, frozen release policies."""
from __future__ import annotations
import io
import json
from pathlib import Path

import pandas as pd
import _hybrid_policy_execution as base
import select_hybrid_fresh_policies as selection

d, e, v = base.d, base.e, base.v
bundle_at, ensure_documents = base.bundle_at, base.ensure_documents
C4 = Path(__file__).resolve().parents[1]
METHODS = [selection.u.q.PRIMARY, *selection.u.q.b.METHODS]


def outer_group_cases(manifest, group_id):
    groups = [g for g in manifest['groups'] if g['group_id'] == group_id]
    d.require(len(groups) == 1 and groups[0]['split_role'] == 'outer_test',
              'Only an assigned outer-test group is allowed.')
    originals = [r for r in manifest['scenarios'] if r['group_id'] == group_id]
    d.require(all(r['split_role'] == 'outer_test' for r in originals), 'Outer case role differs.')
    view = dict(groups=[{**groups[0], 'split_role':'conformal_calibration'}],
                scenarios=[{**r, 'split_role':'conformal_calibration'} for r in originals])
    ordered = base.base.fresh_group_cases(view, group_id)
    by_id = {r['scenario_id']:r for r in originals}
    return [by_id[r['scenario_id']] for r in ordered]


def expected_case_source(source):
    text = source.replace('fresh policy', 'fresh outer').replace('Fresh policy', 'Fresh outer')
    text = text.replace('Fresh outer-calibration', 'Fresh outer-test')
    text = text.replace('from _hybrid_policy_execution import policy_group_cases as training_group_cases, verify_execution_binding',
        'from _hybrid_outer_execution import outer_group_cases as training_group_cases, verify_execution_binding')
    return text.replace('fresh_policy', 'fresh_outer')


def expected_group_function(source):
    start = source.index('def run_group(')
    end = source.index('\n\ndef completed_group(', start)
    return (source[start:end].replace('f.policy_group_cases', 'f.outer_group_cases')
        .replace('policy-admission-executor-v1', 'outer-admission-executor-v1')
        .replace('policy_calibration', 'outer_test').replace('fresh_policy', 'fresh_outer')
        .replace('run_hybrid_policy_case_admission.py', 'run_hybrid_outer_case_admission.py'))


def verify_selection(root, c4=C4, *, full_inputs=False, replay=False):
    """Verify saved policy evidence; never train or regenerate future predictions."""
    root, c4 = Path(root), Path(c4)
    directory = bundle_at(root)/'policy-selection-v1'
    claim, claim_hash = d.checked_json(directory, 'claim')
    output = directory/'completion-v1'
    metadata, metadata_hash = d.checked_json(output)
    d.require(claim['purpose'] == 'fresh_four_method_policy_selection_v1'
        and metadata['purpose'] == 'verified_fresh_four_method_policy_selection'
        and metadata['claim_sha256'] == claim_hash
        and claim['cases'] == metadata['cases'] == 48
        and claim['assigned_groups'] == metadata['assigned_groups'] == 12
        and claim['nonempty_groups'] == metadata['nonempty_groups'] == 12
        and claim['blocked_cases_retained'] == metadata['blocked_cases_retained'] == 396
        and metadata['selection_frozen_for_final_test'] is True,
        'Completed 48-case, 12-group frozen policy selection is required.')
    for key in ('N1_refitted', 'N2_refitted', 'future_reevaluated', 'outer_test_evaluated', 'production_release_authorized'):
        d.require(metadata[key] is False, 'Policy completion scope changed: '+key)
    for key in ('N1_refitting_authorized', 'N2_refitting_authorized', 'future_reevaluation_authorized',
                'outer_evaluation_authorized', 'production_release_authorized', 'formal_conditional_risk_guarantee_established'):
        d.require(claim[key] is False, 'Policy selection claim scope changed: '+key)
    pins = {str(directory/'claim.json'):claim_hash, str(output/'metadata.json'):metadata_hash}
    for name, digest in metadata['files'].items():
        d.require(Path(name).name == name, 'Unexpected policy output filename.')
        d.checked(output/name, digest); pins[str(output/name)] = digest
    result, digest = d.checked_json(output, 'selection')
    d.require(metadata['files']['selection.json'] == digest and result['claim_sha256'] == claim_hash
        and result['purpose'] == 'fresh_frozen_empirical_global_and_adaptive_policy_selection'
        and result['model_bindings'] == claim['model_bindings']
        and result['N2_calibration_sha256'] == claim['N2_calibration_sha256']
        and result['definition'] == claim['definition'] and result['context_support'] == claim['context_support']
        and result['primary_method'] == claim['primary_method'] == METHODS[0]
        and claim['methods'] == METHODS and set(result['methods']) == set(METHODS)
        and result['evaluation'] == 'policy_calibration_selection_not_final_test'
        and result['production_release_authorized'] is False
        and result['formal_conditional_risk_guarantee_established'] is False,
        'Selected policies differ from the frozen selection claim.')
    policies = selection.verify_definition(claim['definition'])
    d.require(claim['candidates'] == policies, 'Frozen policy candidate grid differs.')
    for method in METHODS:
        d.require(set(result['methods'][method]) == {'global','adaptive'}, 'Both policy families required.')
        for family, entry in result['methods'][method].items():
            policy = entry['selected_policy']
            if entry['status'] == 'UNSUPPORTED_SELECTION_REVIEW_ONLY':
                d.require(policy == dict(policy_id='unsupported-review-only',kind='review_only'), 'Unsupported policy must remain review-only.')
            else:
                d.require(entry['status'] == 'EMPIRICALLY_SELECTED' and policy in policies[family], 'Unknown selected policy.')
            d.require(entry['formal_conditional_risk_guarantee_established'] is False and entry['release_authorized'] is False,
                      'Research selection must not authorize production release.')
    if full_inputs:
        for path, checksum in claim['inputs'].items(): d.checked(Path(path), checksum)
        bound = dict(claim['inputs'])
        definition, _ = selection.frozen_policy_rules(root, c4, bound, claim['model_bindings'])
        d.require(definition == claim['definition'], 'Original pre-outcome policy rules differ.')
    if replay:
        d.require(full_inputs, 'Policy replay requires full input verification.')
        def read(name):
            return pd.read_csv(io.BytesIO(d.checked(output/name, metadata['files'][name])), float_precision='round_trip')
        index, targets = read('case_index.csv'), read('targets.csv').set_index('scenario_id')
        d.require(len(index) == 48 and index.group_id.nunique() == 12, 'Policy row/group counts differ.')
        contexts = read('contexts.csv')
        support = selection.context_support(contexts.to_dict('records'))
        d.require(support == claim['context_support'], 'Frozen context support differs.')
        recalculated = selection.select_methods(read('candidate_signals.csv'), targets, index, claim['definition'], policies)
        d.require(recalculated[0] == result['methods'], 'Saved selected policies do not reproduce from bound policy signals and targets.')
    return claim, result, pins


def build_execution(root, c4=C4):
    root, c4 = Path(root), Path(c4)
    # Reuse the already-verified N2/protocol/input bindings, without writing or
    # executing policy cases. Its immutable science checks remain unchanged.
    prior, manifest = base.build_execution(root, c4)
    policy_claim, selected, policy_pins = verify_selection(root, c4, full_inputs=True, replay=True)
    d.require(selected['model_bindings'] == prior['N2_model_bindings'], 'Selected policy and fixed N2 model bindings differ.')
    n2_path = bundle_at(root)/'n2-calibration-v1/completion-v1/calibration.json'
    d.require(policy_claim['N2_calibration_sha256'] == prior['inputs'][str(n2_path)], 'Selected policy N2 artifact differs.')
    d.require(policy_claim['environment'] == prior['environment'], 'Policy/admission environment differs.')
    groups = [g['group_id'] for g in manifest['groups'] if g['split_role'] == 'outer_test']
    d.require(len(groups) == len(set(groups)) == 16, 'Exactly 16 assigned outer groups required.')
    for gid in groups: outer_group_cases(manifest,gid)
    executor_dir = bundle_at(root)/'outer-admission-executor-v1'
    if not executor_dir.exists():
        run_root = root/'c4_artifacts/benchmark_runs/adult'/v.VERSION
        for gid in groups:
            d.require(not (run_root/'group_admissions'/gid).exists(),
                      'Outer group artifacts predate this executor claim; preserve and inspect.')
            for row in outer_group_cases(manifest,gid):
                d.require(not (run_root/row['scenario_id']).exists(),
                          'Outer case artifacts predate this executor claim; preserve and inspect.')
    source = c4/'scripts'
    d.require((source/'run_hybrid_outer_case_admission.py').read_text(encoding='utf-8-sig') ==
        expected_case_source((source/'run_hybrid_policy_case_admission.py').read_text(encoding='utf-8-sig')),
        'Outer case science differs beyond the role adapter.')
    group_source = (source/'run_hybrid_outer_admissions.py').read_text(encoding='utf-8-sig')
    start,end = group_source.index('def run_group('),group_source.index('\n\ndef completed_group(')
    d.require(group_source[start:end] == expected_group_function((source/'run_hybrid_policy_admissions.py').read_text(encoding='utf-8-sig')),
        'Outer checkpoint/order implementation changed.')
    pins = {**prior['inputs'], **policy_pins}
    for name in ('_hybrid_outer_execution.py','run_hybrid_outer_case_admission.py','run_hybrid_outer_admissions.py'):
        path = source/name; pins[str(path)] = d.sha(path.read_bytes())
    for path,digest in pins.items(): d.checked(Path(path),digest)
    return dict(purpose='fresh_outer_admission_executor_v1',inputs=pins,environment=prior['environment'],
        allowed_role='outer_test',allowed_group_ids=groups,planned_groups=16,planned_cases=592,
        frozen_protocol_sha256=prior['frozen_protocol_sha256'],manifest_sha256=prior['manifest_sha256'],
        N2_model_bindings=prior['N2_model_bindings'],
        policy_selection_sha256=policy_pins[str(bundle_at(root)/'policy-selection-v1/completion-v1/selection.json')],
        policy_fixed_before_outer_admission=True,allowed_stages=['repair_generation','simulation_admission'],
        outer_admissions_authorized=True,candidate_training_authorized=False,future_evaluation_authorized=False,
        N1_refitting_authorized=False,N2_refitting_authorized=False,policy_fitting_authorized=False,
        outer_evaluation_authorized=False,production_release_authorized=False),manifest


def verify_execution_binding(root, group_id):
    from _hybrid_fresh_candidate import environment
    root = Path(root)
    claim,_ = d.checked_json(bundle_at(root)/'outer-admission-executor-v1','claim')
    d.require(claim['purpose'] == 'fresh_outer_admission_executor_v1' and claim['allowed_role'] == 'outer_test'
        and group_id in claim['allowed_group_ids'] and claim['planned_groups'] == 16 and claim['planned_cases'] == 592
        and claim['outer_admissions_authorized'] is True and claim['policy_fixed_before_outer_admission'] is True
        and claim['allowed_stages'] == ['repair_generation','simulation_admission'], 'Outer admission scope differs.')
    for key in ('candidate_training_authorized','future_evaluation_authorized','N1_refitting_authorized',
                'N2_refitting_authorized','policy_fitting_authorized','outer_evaluation_authorized','production_release_authorized'):
        d.require(claim[key] is False, 'Unauthorized outer stage: '+key)
    d.require(claim['environment'] == environment(), 'Outer admission environment changed.')
    for path,digest in claim['inputs'].items(): d.checked(Path(path),digest)
    _,selected,_ = verify_selection(root)
    selection_path = bundle_at(root)/'policy-selection-v1/completion-v1/selection.json'
    d.require(claim['policy_selection_sha256'] == claim['inputs'][str(selection_path)]
        and selected['model_bindings'] == claim['N2_model_bindings'], 'Outer policy/model bindings changed.')
    directory = root/'c4_artifacts/benchmark_protocols/adult'/v.VERSION
    frozen,freeze_hash = d.checked_json(directory,'freeze')
    manifest,manifest_hash = d.checked_json(directory,'scenario_manifest')
    d.require(freeze_hash == claim['frozen_protocol_sha256'] and
        manifest_hash == claim['manifest_sha256'] == frozen['scenario_manifest_sha256'], 'Outer frozen protocol/manifest changed.')
    outer_group_cases(manifest,group_id)
    return manifest
