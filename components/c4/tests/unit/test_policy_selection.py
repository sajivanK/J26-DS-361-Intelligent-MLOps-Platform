"""Frozen policy selection: group statistics, support, fallback and leakage checks."""
import ast
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest
from c4.risk.benchmark_n3 import BOUNDARIES, TARGETS
from c4.risk.benchmark_policy import candidate_policies, policy_decision, GATES
from c4.risk.benchmark_policy_selection import policy_statistics, select_policy, validate_index

RULE = {'minimum_auto_deploy_groups': 3, 'empirical_conditional_unsafe_ceiling': .05}


def fixture(groups=3, each=1):
    rows = [{'scenario_id': f'g{g}-s{s}', 'group_id': f'g{g}', 'split_role': 'policy_calibration'}
            for g in range(groups) for s in range(each)]
    targets = {r['scenario_id']: dict(BOUNDARIES) for r in rows}
    return rows, targets


def test_zero_deployment_risk_is_null_and_automation_zero():
    rows, targets = fixture()
    m = policy_statistics(dict.fromkeys(targets, 'HUMAN_REVIEW'), targets, rows)
    assert m['equal_group_conditional_unsafe_rate'] is None
    assert m['scenario_conditional_unsafe_rate'] is None
    assert m['equal_group_automation'] == 0


def test_equal_group_metrics_do_not_weight_large_groups_more():
    rows, targets = fixture(2, 1)
    for n in range(9):
        sid = f'extra{n}'
        rows.append({'scenario_id': sid, 'group_id': 'g1', 'split_role': 'policy_calibration'})
        targets[sid] = dict(BOUNDARIES)
    targets['g0-s0']['future_global_margin'] = -.01
    decisions = dict.fromkeys(targets, 'AUTO_DEPLOY')
    metrics = policy_statistics(decisions, targets, rows)
    assert metrics['equal_group_conditional_unsafe_rate'] == .5
    assert metrics['scenario_conditional_unsafe_rate'] == pytest.approx(1/11)
    decisions = dict.fromkeys(targets, 'HUMAN_REVIEW'); decisions['g0-s0'] = 'AUTO_DEPLOY'
    metrics = policy_statistics(decisions, targets, rows)
    assert metrics['equal_group_automation'] == .5
    assert metrics['scenario_automation'] == pytest.approx(1/11)


def test_unsafe_boundary_is_strict_and_checks_complement():
    rows, targets = fixture()
    decisions = dict.fromkeys(targets, 'AUTO_DEPLOY')
    assert policy_statistics(decisions, targets, rows)['unsafe_auto_deploy_count'] == 0
    targets['g0-s0']['future_nonaffected_margin'] = -.020001
    assert policy_statistics(decisions, targets, rows)['unsafe_auto_deploy_count'] == 1


def test_three_groups_required_even_with_many_scenarios():
    rows, targets = fixture(2, 10)
    policy = candidate_policies()['global'][0]
    result = select_policy([policy], {policy['policy_id']: dict.fromkeys(targets, 'AUTO_DEPLOY')}, targets, rows, RULE)
    assert result['status'] == 'UNSUPPORTED_SELECTION_REVIEW_ONLY'
    assert result['selected_policy']['kind'] == 'review_only'


def test_zero_deployment_cannot_be_selected_as_zero_risk():
    rows, targets = fixture()
    policy = candidate_policies()['global'][0]
    result = select_policy([policy], {policy['policy_id']: dict.fromkeys(targets, 'HUMAN_REVIEW')}, targets, rows, RULE)
    assert result['selected_candidate_metrics'] is None
    assert not result['candidates'][0]['eligible']


def test_risk_ceiling_is_inclusive():
    rows, targets = fixture(3, 20)
    targets['g0-s0']['future_global_margin'] = -.01
    targets['g1-s0']['future_global_margin'] = -.01
    targets['g2-s0']['future_global_margin'] = -.01
    policy = candidate_policies()['global'][0]
    result = select_policy([policy], {policy['policy_id']: dict.fromkeys(targets, 'AUTO_DEPLOY')}, targets, rows, RULE)
    assert result['status'] == 'EMPIRICALLY_SELECTED'
    targets['g0-s1']['future_global_margin'] = -.01
    result = select_policy([policy], {policy['policy_id']: dict.fromkeys(targets, 'AUTO_DEPLOY')}, targets, rows, RULE)
    assert result['status'] == 'UNSUPPORTED_SELECTION_REVIEW_ONLY'


def test_tie_prefers_larger_global_buffer():
    rows, targets = fixture()
    policies = candidate_policies()['global']
    result = select_policy(policies, {p['policy_id']: dict.fromkeys(targets, 'AUTO_DEPLOY') for p in policies}, targets, rows, RULE)
    assert result['selected_policy'] == policies[-1]


def test_tie_prefers_lexicographically_larger_adaptive_tuple():
    rows, targets = fixture()
    policies = candidate_policies()['adaptive']
    result = select_policy(policies, {p['policy_id']: dict.fromkeys(targets, 'AUTO_DEPLOY') for p in policies}, targets, rows, RULE)
    assert result['selected_policy'] == policies[-1]


def test_tie_prefers_lower_group_risk_before_larger_buffer():
    rows, targets = fixture(3, 20)
    targets['g0-s0']['future_global_margin'] = -.01
    policies = candidate_policies()['global'][:2]
    a = dict.fromkeys(targets, 'AUTO_DEPLOY'); a['g0-s0'] = 'HUMAN_REVIEW'
    b = dict.fromkeys(targets, 'AUTO_DEPLOY'); b['g0-s1'] = 'HUMAN_REVIEW'
    result = select_policy(policies, {policies[0]['policy_id']: a, policies[1]['policy_id']: b}, targets, rows, RULE)
    assert result['selected_policy'] == policies[0]


def test_objective_prefers_more_automation_among_eligible_candidates():
    rows, targets = fixture(3, 2)
    policies = candidate_policies()['global'][:2]
    a = dict.fromkeys(targets, 'AUTO_DEPLOY'); b = dict(a)
    b['g0-s0'] = 'HUMAN_REVIEW'
    result = select_policy(policies, {policies[0]['policy_id']: a, policies[1]['policy_id']: b}, targets, rows, RULE)
    assert result['selected_policy'] == policies[0]


@pytest.mark.parametrize('role', ['forecast_train', 'conformal_calibration', 'conformal_extension', 'outer_test'])
def test_nonpolicy_fold_rejected(role):
    rows, targets = fixture(); rows[0]['split_role'] = role
    with pytest.raises(ValueError, match='Only policy'):
        policy_statistics(dict.fromkeys(targets, 'AUTO_DEPLOY'), targets, rows)


def test_duplicate_scenario_rejected():
    rows, _ = fixture(); rows.append(rows[0])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_index(rows)


@pytest.mark.parametrize('value', [True, math.nan, math.inf, 1.01])
def test_invalid_future_margins_rejected(value):
    rows, targets = fixture(); targets['g0-s0'][TARGETS[0]] = value
    with pytest.raises(ValueError, match='Invalid future'):
        policy_statistics(dict.fromkeys(targets, 'HUMAN_REVIEW'), targets, rows)


def test_missing_target_scenario_rejected():
    rows, targets = fixture(); decisions = dict.fromkeys(targets, 'AUTO_DEPLOY'); targets.pop('g0-s0')
    with pytest.raises(ValueError, match='inventory differs'):
        policy_statistics(decisions, targets, rows)


def test_changed_frozen_support_rule_rejected():
    rows, targets = fixture(); policy = candidate_policies()['global'][0]
    with pytest.raises(ValueError, match='Frozen selection'):
        select_policy([policy], {policy['policy_id']: dict.fromkeys(targets, 'AUTO_DEPLOY')}, targets, rows,
                      {**RULE, 'minimum_auto_deploy_groups': 1})


def test_actual_policy_engine_suppresses_sparse_context_deployment():
    rows, targets = fixture(); policy = candidate_policies()['adaptive'][0]
    intervals = {t: [b+.02, b+.03] for t, b in BOUNDARIES.items()}
    evidence = {'c1_harm_width': .01, 'c2_diagnosis_stability': .9, 'c3_fidelity_score': .9, 'c3_utility_score': .01}
    decisions = {sid: policy_decision(intervals, evidence, dict.fromkeys(GATES, True), n2_supported=True,
        policy=policy, unsupported_contexts=('LOW',))['decision'] for sid in targets}
    assert set(decisions.values()) == {'HUMAN_REVIEW'}
    assert select_policy([policy], {policy['policy_id']: decisions}, targets, rows, RULE)['selected_policy']['kind'] == 'review_only'


def audit_combiner():
    path = Path(__file__).resolve().parents[2] / 'scripts/audit_policy_snapshot.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'combine_tables')
    ns = {'pd': pd, 'np': np, 'TARGET_COLUMNS': TARGETS, 'validate_index': validate_index,
          'require': lambda condition, message: None if condition else (_ for _ in ()).throw(ValueError(message))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns['combine_tables']


def table_fixture():
    ids = ['s1', 's2']
    return (pd.DataFrame(0., index=ids, columns=[f'feature{n}' for n in range(75)]),
            pd.DataFrame(0., index=ids, columns=list(TARGETS)),
            pd.DataFrame({'scenario_id': ids, 'group_id': ['g1', 'g1'], 'split_role': ['policy_calibration']*2}))


def test_audit_combines_valid_tables():
    f, t, i = table_fixture()
    combined = audit_combiner()([f], [t], [i])
    assert combined[0].shape == (2, 75)


def test_audit_rejects_cross_group_duplicate_scenarios():
    f, t, i = table_fixture()
    with pytest.raises(ValueError, match='Duplicate'):
        audit_combiner()([f, f], [t, t], [i, i])


def test_audit_rejects_misaligned_targets():
    f, t, i = table_fixture()
    with pytest.raises(ValueError, match='order differs'):
        audit_combiner()([f], [t.iloc[::-1]], [i])


def test_audit_rejects_nonfinite_feature():
    f, t, i = table_fixture(); f.iloc[0, 0] = math.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        audit_combiner()([f], [t], [i])



def test_calibration_script_saves_review_fallback_and_preserves_escalation(tmp_path):
    """Run the actual CLI on synthetic saved tables; these are test fixtures only."""
    import argparse
    import hashlib
    import json
    import runpy
    import sys
    from types import ModuleType
    from unittest.mock import patch
    base = Path(__file__).resolve().parents[2]
    version = 'adult-simulation-pp1-v2'
    common = ModuleType('_common')
    common.C4_ROOT = base
    common.sha = lambda data: hashlib.sha256(data).hexdigest()
    def require(condition, message):
        if not condition: raise ValueError(message)
    def checked_bytes(path, digest):
        data = Path(path).read_bytes(); require(common.sha(data) == digest, 'Checksum differs'); return data
    def save_json(path, payload):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(payload, sort_keys=True, allow_nan=False) + '\n').encode()
        with path.open('xb') as file: file.write(data)
        digest = common.sha(data); path.with_suffix('.sha256').write_text(digest)
        return digest
    def checked_json(directory, stem='metadata'):
        path = Path(directory) / (stem + '.json')
        digest = path.with_suffix('.sha256').read_text()
        return json.loads(checked_bytes(path, digest)), digest
    def parser(doc):
        p = argparse.ArgumentParser(description=doc); p.add_argument('--data-root'); return p
    common.require = require; common.checked_bytes = checked_bytes; common.checked_json = checked_json
    common.save_json = save_json; common.parser = parser; common.name = str
    common.resolve_data_root = lambda value: tmp_path
    common.git_context = lambda: {'working_tree_dirty': False, 'git_revision': 'synthetic-fixture'}
    common.utc_now = lambda: 'synthetic-fixture-only'
    pipeline = ModuleType('_policy_pipeline'); pipeline.VERSION = version; pipeline.POLICY_SPEC_HASH = 'policy-spec-fixture'
    specdir = tmp_path / 'spec'
    definition = {'methods': ['lightgbm', 'linear'], 'primary_method': 'lightgbm',
        'safety_boundaries': BOUNDARIES, 'selection': RULE}
    save_json(specdir / 'definition.json', definition)
    save_json(specdir / 'candidates.json', candidate_policies())
    frozen = {'n1_binding': {}, 'n2_metadata_sha256': 'n2-fixture', 'n3_metadata_sha256': 'n3-fixture'}
    pipeline.verify_policy_spec = lambda root: (frozen, specdir)
    pipeline.verify_context_support = lambda root: ({'unsupported_contexts': ['HIGH']}, 'support-fixture')
    directory = tmp_path / 'c4_artifacts/meta_datasets/adult' / version / 'policy-all12-audit-v1'
    directory.mkdir(parents=True)
    ix, targets = fixture(11, 4)
    target_frame = pd.DataFrame.from_dict(targets, orient='index'); target_frame.index.name = 'scenario_id'
    signal_rows = []
    for method in definition['methods']:
        for family, policies in candidate_policies().items():
            for policy in policies:
                for row in ix:
                    signal_rows.append({**row, 'method': method, 'family': family, 'policy_id': policy['policy_id'],
                        'decision': 'ESCALATE' if row['scenario_id'] == 'g0-s0' else 'HUMAN_REVIEW'})
    frames = {'targets.csv': (target_frame, True), 'scenario_index.csv': (pd.DataFrame(ix), False),
              'candidate_signals.csv': (pd.DataFrame(signal_rows), False)}
    hashes = {}
    for filename, (frame, include_index) in frames.items():
        data = frame.to_csv(index=include_index).encode(); (directory / filename).write_bytes(data)
        hashes[filename] = common.sha(data)
    save_json(directory / 'metadata.json', {'purpose': 'verified_combined_policy_snapshot', 'rows': 44,
        'fold': 'policy_calibration', 'nonempty_groups': 11, 'admission_cases': 444, 'blocked_cases': 400,
        'policy_specification_sha256': pipeline.POLICY_SPEC_HASH, 'policy_context_support_sha256': 'support-fixture',
        'source_code_sha256': common.sha((base / 'scripts/audit_policy_snapshot.py').read_bytes()),
        'source_links': {}, 'files': hashes})
    with patch.dict(sys.modules, {'_common': common, '_policy_pipeline': pipeline}), \
         patch.object(sys, 'argv', ['calibrate_policy_benchmark.py', '--save']):
        runpy.run_path(str(base / 'scripts/calibrate_policy_benchmark.py'), run_name='__main__')
    output = tmp_path / 'c4_artifacts/policies/adult' / version / 'policy-fit-v1'
    selection, _ = checked_json(output, 'selection')
    for families in selection['methods'].values():
        for result in families.values():
            assert result['status'] == 'UNSUPPORTED_SELECTION_REVIEW_ONLY'
            assert result['selected_or_fallback_metrics']['equal_group_conditional_unsafe_rate'] is None
            assert result['selected_or_fallback_metrics']['decision_counts'] == {
                'AUTO_DEPLOY': 0, 'HUMAN_REVIEW': 43, 'ESCALATE': 1}
    metadata, _ = checked_json(output)
    for filename, digest in metadata['files'].items(): checked_bytes(output / filename, digest)
    assert selection['release_authorized'] is False
