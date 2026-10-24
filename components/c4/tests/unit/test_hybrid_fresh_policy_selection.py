"""Synthetic technical checks, not empirical release-performance evidence."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

C4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(C4 / 'scripts'))
import select_hybrid_fresh_policies as s

METHODS = [s.u.q.PRIMARY, *s.u.q.b.METHODS]
RULES = json.loads((C4 / 'configs/policy_benchmark_v1.json').read_text())


@pytest.fixture
def tmp_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('c4selection')
    if os.name == 'nt':
        raw = str(path.resolve())
        if not raw.startswith('\\\\?\\'):
            raw = '\\\\?\\UNC\\'+raw[2:] if raw.startswith('\\\\') else '\\\\?\\'+raw
        path = Path(raw)
    yield path
    if os.name == 'nt' and path.exists():
        shutil.rmtree(path)


def doc(directory, stem, value):
    directory.mkdir(parents=True, exist_ok=True)
    data = s.e.encoded(value)
    (directory / (stem+'.json')).write_bytes(data)
    (directory / (stem+'.sha256')).write_text(s.d.sha(data)+'\n')
    return s.d.sha(data)


def batch(mode='safe'):
    rows, states = [], {m: {'auto_deploy_supported': True} for m in METHODS}
    features = dict(c1_harm_width=0., c2_diagnosis_stability=1., c3_fidelity_score=1., c3_utility_score=1.)
    for i in range(3):
        intervals = {t: [.02, .03] for t in s.d.TARGETS}
        if mode == 'uncertain': intervals[s.d.TARGETS[0]] = [-.01, .03]
        if mode == 'unsafe': intervals[s.d.TARGETS[0]] = [-.03, -.01]
        rows.append(dict(scenario_id=f's{i}', group_id=f'g{i}', split_role='policy_calibration',
            features=dict(features), gates=dict.fromkeys(s.GATES, True),
            intervals={m: copy.deepcopy(intervals) for m in METHODS}))
    index = pd.DataFrame([{k: r[k] for k in ('scenario_id','group_id','split_role')} for r in rows])
    targets = pd.DataFrame([{t: .01 for t in s.d.TARGETS} for _ in rows], index=pd.Index(index.scenario_id, name='scenario_id'))
    return rows, states, index, targets


def selected(mode='safe', unsupported=(), unsafe_targets=False):
    rows, states, index, targets = batch(mode)
    if unsafe_targets: targets.loc[:, s.d.TARGETS[0]] = -.01
    policies = s.verify_definition(RULES)
    signals, safety = s.replay_signals(rows, states, policies, unsupported)
    results = s.select_methods(signals, targets, index, RULES, policies)
    return rows, states, index, targets, signals, safety, results


def test_fixed_grid_and_monotonicity():
    p = s.verify_definition(RULES)
    assert len(p['global']) == 4 and len(p['adaptive']) == 20
    assert all(list(r['buffers'].values()) == sorted(r['buffers'].values()) for r in p['adaptive'])


@pytest.mark.parametrize('key,value', [('additional_buffer_grid_fraction',[0,.1]),('assigned_groups',11),('change_rules_after_policy_outcomes',True)])
def test_changed_design_rejected(key,value):
    rules = copy.deepcopy(RULES); rules[key] = value
    with pytest.raises(ValueError): s.verify_definition(rules)


def test_context_support_counts_groups_not_rows():
    rows = [dict(scenario_id=f's{i}',group_id='g',split_role='policy_calibration',context='MEDIUM') for i in range(8)]
    result = s.context_support(rows)
    assert result['distinct_admitted_groups_by_context']['MEDIUM'] == 1
    assert set(result['unsupported_contexts']) == set(s.CONTEXTS)
    assert result['registry_saved_before_future'] is False


@pytest.mark.parametrize('mode,decision', [('uncertain','HUMAN_REVIEW'),('unsafe','ESCALATE')])
def test_n3_precedence_and_review_fallback(mode,decision):
    *_, signals, safety, result = selected(mode)
    assert set(signals.decision) == {decision}
    for families in result[0].values():
        for entry in families.values():
            assert entry['status'] == 'UNSUPPORTED_SELECTION_REVIEW_ONLY'
            assert entry['selected_or_fallback_metrics']['decision_counts'][decision] == 3
            assert entry['selected_or_fallback_metrics']['equal_group_conditional_unsafe_rate'] is None


def test_positive_control_tie_break_and_no_guarantee():
    result = selected()[-1]
    for families in result[0].values():
        assert families['global']['selected_policy']['buffer'] == .005
        assert families['adaptive']['selected_policy']['buffers'] == dict.fromkeys(s.CONTEXTS,.005)
        assert families['global']['formal_conditional_risk_guarantee_established'] is False


def test_observed_unsafe_deployments_reject_all_candidates():
    result = selected(unsafe_targets=True)[-1]
    assert all(v['status'] == 'UNSUPPORTED_SELECTION_REVIEW_ONLY' for families in result[0].values() for v in families.values())


def test_sparse_context_affects_only_adaptive_safe_cases():
    *_, signals, safety, results = selected(unsupported=['LOW'])
    assert set(signals.loc[signals.family.eq('adaptive')].decision) == {'HUMAN_REVIEW'}
    assert set(signals.loc[signals.family.eq('global')].decision) == {'AUTO_DEPLOY'}


def test_gate_failure_always_escalates():
    rows, states, *_ = batch()
    rows[0]['gates']['privacy_check'] = False
    signals, _ = s.replay_signals(rows,states,s.candidate_policies(),['LOW'])
    assert set(signals.loc[signals.scenario_id.eq('s0')].decision) == {'ESCALATE'}


@pytest.mark.parametrize('mutation',['duplicate','missing_method','wrong_role','wrong_group','missing_signal'])
def test_signal_inventory_corruption_rejected(mutation):
    _,_,index,targets,signals,*_ = selected()
    if mutation == 'duplicate': signals = pd.concat([signals,signals.iloc[:1]])
    elif mutation == 'missing_method': signals = signals.loc[~signals.method.eq(METHODS[0])]
    elif mutation == 'wrong_role': signals.loc[0,'split_role'] = 'outer_test'
    elif mutation == 'wrong_group': signals.loc[0,'group_id'] = 'wrong'
    else: signals = signals.iloc[1:]
    with pytest.raises(ValueError): s.select_methods(signals,targets,index,RULES,s.candidate_policies())


def provenance(root,c4):
    inputs = {}
    for rel in ('configs/policy_benchmark_v1.json','src/c4/risk/benchmark_policy.py','src/c4/risk/benchmark_n3.py','src/c4/risk/benchmark_policy_selection.py'):
        path = c4 / rel; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((C4/rel).read_bytes()); inputs[str(path)] = s.d.sha(path.read_bytes())
    validation = dict(policy_rules=RULES,model_methods_override_policy_source=METHODS,inputs=inputs)
    digest = doc(s.f.bundle_at(root),'validation_protocol',validation)
    directory = s.f.bundle_at(root)/'admission-executor-v1'
    digest = doc(directory,'claim',dict(validation_protocol_sha256=digest))
    return {str(directory/'claim.json'):digest}, validation


def test_policy_definition_bound_through_prior_admission_claim(tmp_path):
    c4 = tmp_path/'source'; pins,_ = provenance(tmp_path,c4)
    definition,policies = s.frozen_policy_rules(tmp_path,c4,pins,dict.fromkeys(METHODS))
    assert definition == RULES and len(policies['adaptive']) == 20
    assert str(c4/'src/c4/risk/benchmark_policy_selection.py') in pins


@pytest.mark.parametrize('mutation',['missing_claim_pin','changed_config','changed_code','changed_protocol'])
def test_unbound_or_changed_prefuture_rules_rejected(tmp_path,mutation):
    c4 = tmp_path/'source'; pins,validation = provenance(tmp_path,c4)
    if mutation == 'missing_claim_pin': pins.clear()
    elif mutation == 'changed_config': (c4/'configs/policy_benchmark_v1.json').write_text('{}')
    elif mutation == 'changed_code': (c4/'src/c4/risk/benchmark_policy.py').write_text('#changed')
    else:
        validation['policy_rules'] = {}; doc(s.f.bundle_at(tmp_path),'validation_protocol',validation)
    with pytest.raises(ValueError): s.frozen_policy_rules(tmp_path,c4,pins,dict.fromkeys(METHODS))


def test_selection_save_reload_resume_and_tamper_rejection(tmp_path):
    _,_,index,targets,signals,safety,results = selected('uncertain')
    source = tmp_path/'source.txt'; source.write_text('frozen')
    claim = dict(inputs={str(source):s.d.sha(source.read_bytes())},primary_method=METHODS[0],
        model_bindings=dict.fromkeys(METHODS),N2_calibration_sha256='a'*64,definition=RULES,
        context_support={},cases=3,nonempty_groups=3,blocked_cases_retained=0)
    args = (tmp_path,claim,index,targets,pd.DataFrame(),signals,safety,results)
    output = s.save_results(*args)
    assert s.save_results(*args) == output
    saved,_ = s.d.checked_json(output,'selection')
    assert saved['production_release_authorized'] is False
    assert saved['methods'][METHODS[0]]['global']['selected_or_fallback_metrics']['equal_group_conditional_unsafe_rate'] is None
    (output/'candidate_signals.csv').write_text('changed')
    with pytest.raises(ValueError): s.save_results(*args)


def test_prepare_consumes_verified_batch_and_prefuture_features(tmp_path,monkeypatch):
    c4 = tmp_path/'source'; pins,_ = provenance(tmp_path,c4)
    script = c4/'scripts/select_hybrid_fresh_policies.py'
    script.parent.mkdir(parents=True,exist_ok=True); script.write_bytes(Path(s.__file__).read_bytes())
    rows,states,index,targets = batch('uncertain')
    models = dict.fromkeys(METHODS,{'test_binding':'fixed'})
    evaluator = dict(inputs=pins,model_bindings=models,N2_calibration_sha256='a'*64)
    directory = s.f.bundle_at(tmp_path)/'policy-future-evaluator-v1'
    evaluator_hash = doc(directory,'claim',evaluator)
    records, sources = [], {}
    for row in rows:
        sid = row['scenario_id']; run = s.u.run_at(tmp_path,row)
        feature_dir = run/'forecast_freeze-v1'; feature_dir.mkdir(parents=True)
        values = row['features']
        feature = dict(scenario_id=sid,columns=list(values),values=list(values.values()))
        data = s.e.encoded(feature); (feature_dir/'features.json').write_bytes(data)
        forecasts = {m:{'calibrated':{t+'__'+label:bound for t,(lower,upper) in row['intervals'][m].items()
            for label,bound in [('lower',lower),('upper',upper)]}} for m in METHODS}
        forecast_bytes = s.e.encoded(forecasts); (feature_dir/'forecasts.json').write_bytes(forecast_bytes)
        sources[sid] = dict(feature_sha256=s.d.sha(data),forecasts_sha256=s.d.sha(forecast_bytes))
        upstream = dict(c2_diagnosis=dict(diagnosis_stability=1.),c3_repair_package=dict(repair_package_status='APPROVED',privacy_status='PASS'))
        envelope_hash = doc(run/'pre_release_freeze-v1','upstream_simulation',dict(upstream=upstream))
        doc(run/'pre_release_freeze-v1','freeze',dict(upstream_simulation_sha256=envelope_hash,runtime_checks=dict(schema_compatible=True,rollback_ready=True)))
        doc(run/'external_future-v1','outcomes',dict(files={}))
        records.append(dict(scenario_id=sid,primary_targets=targets.loc[sid].to_dict()))
    output = directory/'completion-v1'; output.mkdir()
    files = {}
    for name,table,indexed in [('case_index.csv',index,False),('targets.csv',targets,True)]:
        data = table.to_csv(index=indexed,lineterminator='\n').encode(); (output/name).write_bytes(data); files[name] = s.d.sha(data)
    report = dict(purpose='fresh_policy_future_evaluation_complete_not_final_outer_results',
        evaluator_claim_sha256=evaluator_hash,cases=3,assigned_groups=12,nonempty_groups=3,
        N2_calibrated=True,policy_calibrated=False,training_executed=False,production_release_authorized=False,
        N2_calibration_sha256='a'*64,records=records,files=files,blocked_cases_retained=0)
    doc(output,'future_batch',report)
    monkeypatch.setattr(s.u.q.c,'environment',lambda: {'fixture':'technical_only'})
    monkeypatch.setattr(s.u,'prepare',lambda root,c4:(evaluator,rows,None,None))
    monkeypatch.setattr(s.u,'load_cohort',lambda claim:None)
    monkeypatch.setattr(s.u,'completed_case',lambda root,row,*args:next(r for r in records if r['scenario_id']==row['scenario_id']))
    monkeypatch.setattr(s.u,'source_record',lambda claim,row:sources[row['scenario_id']])
    monkeypatch.setattr(s.f,'verify_n2',lambda root:({},dict(model_bindings=models,methods=states),{}))
    claim,got_index,got_targets,inputs,got_states,contexts = s.prepare(tmp_path,c4)
    assert got_index.equals(index) and got_targets.equals(targets)
    assert got_states == states and len(contexts) == 3
    assert inputs[0]['gates'] == dict.fromkeys(s.GATES,True)
    assert inputs[0]['intervals'] == rows[0]['intervals']
    assert claim['outer_evaluation_authorized'] is False
    assert claim['context_support']['registry_saved_before_future'] is False
