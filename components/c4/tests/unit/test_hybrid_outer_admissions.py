"""Outer admission scope/checkpoint tests; no research generators are fitted."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

C4 = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(C4/'scripts'))
import _hybrid_outer_execution as f
import run_hybrid_outer_admissions as r
import select_hybrid_fresh_policies as s
import calibrate_hybrid_fresh_n2 as n


@pytest.fixture
def tmp_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('c4outer')
    if os.name == 'nt':
        raw = str(path.resolve())
        if not raw.startswith('\\\\?\\'):
            raw = '\\\\?\\UNC\\'+raw[2:] if raw.startswith('\\\\') else '\\\\?\\'+raw
        path = Path(raw)
    yield path
    if os.name == 'nt' and path.exists(): shutil.rmtree(path)


def doc(directory, stem, value):
    directory.mkdir(parents=True, exist_ok=True)
    data = f.e.encoded(value)
    (directory/(stem+'.json')).write_bytes(data)
    (directory/(stem+'.sha256')).write_text(f.d.sha(data)+'\n')
    return f.d.sha(data)



@pytest.fixture
def manifest():
    return f.v.create_design(json.loads((C4/'configs/benchmark_protocol.json').read_text()))


def test_exact_case_science_reuse():
    assert (C4/'scripts/run_hybrid_outer_case_admission.py').read_text(encoding='utf-8-sig') == f.expected_case_source((C4/'scripts/run_hybrid_policy_case_admission.py').read_text(encoding='utf-8-sig'))


def test_exact_checkpoint_reuse():
    source = (C4/'scripts/run_hybrid_outer_admissions.py').read_text(encoding='utf-8-sig')
    start,end = source.index('def run_group('),source.index('\n\ndef completed_group(')
    assert source[start:end] == f.expected_group_function((C4/'scripts/run_hybrid_policy_admissions.py').read_text(encoding='utf-8-sig'))


def test_all_sixteen_outer_groups_grid_pair_order_no_mutation(manifest):
    before = f.e.encoded(manifest)
    groups = [g for g in manifest['groups'] if g['split_role'] == 'outer_test']
    assert len(groups) == 16
    for group in groups:
        rows = f.outer_group_cases(manifest,group['group_id'])
        assert len(rows) == 37 and all(row['split_role'] == 'outer_test' for row in rows)
        positions = {row['scenario_id']:i for i,row in enumerate(rows)}
        for row in rows:
            if row['repair_condition'] == 'label_corrupted_repair':
                base = next(b for b in rows if b['shared_base_package_id'] == row['shared_base_package_id'] and b['repair_condition'] == 'targeted_repair')
                assert positions[base['scenario_id']] < positions[row['scenario_id']]
    assert f.e.encoded(manifest) == before


@pytest.mark.parametrize('role',['conformal_calibration','policy_calibration','unknown'])
def test_other_roles_rejected(manifest,role):
    gid = 'unknown' if role == 'unknown' else next(g['group_id'] for g in manifest['groups'] if g['split_role'] == role)
    with pytest.raises(ValueError): f.outer_group_cases(manifest,gid)


def make_selection(root,models,environment=None):
    rules = json.loads((C4/'configs/policy_benchmark_v1.json').read_text())
    inputs = {str(C4/rel):f.d.sha((C4/rel).read_bytes()) for rel in ('configs/policy_benchmark_v1.json',
        'src/c4/risk/benchmark_policy.py','src/c4/risk/benchmark_n3.py','src/c4/risk/benchmark_policy_selection.py')}
    validation_hash = doc(f.bundle_at(root),'validation_protocol',dict(policy_rules=rules,
        model_methods_override_policy_source=f.METHODS,inputs=inputs))
    original = f.bundle_at(root)/'admission-executor-v1'
    old,_ = f.d.checked_json(original,'claim')
    old['validation_protocol_sha256'] = validation_hash
    old_hash = doc(original,'claim',old)
    inputs[str(original/'claim.json')] = old_hash
    ids = [f's{i}' for i in range(48)]
    index = pd.DataFrame(dict(scenario_id=ids,group_id=[f'g{i//4}' for i in range(48)],split_role=['policy_calibration']*48))
    targets = pd.DataFrame({t:[.01]*48 for t in f.d.TARGETS},index=pd.Index(ids,name='scenario_id'))
    contexts = pd.DataFrame([dict(**row,quality=.6,context='MEDIUM',quality_is_calibrated_probability=False) for row in index.to_dict('records')])
    support = s.context_support(contexts.to_dict('records'))
    features = dict(c1_harm_width=.01,c2_diagnosis_stability=.6,c3_fidelity_score=.7,c3_utility_score=.1)
    rows = [dict(**row,features=features,gates=dict.fromkeys(s.GATES,True),intervals={m:{t:[-.01,.01] for t in f.d.TARGETS} for m in f.METHODS}) for row in index.to_dict('records')]
    policies = s.verify_definition(rules)
    signals,safety = s.replay_signals(rows,{m:dict(auto_deploy_supported=True) for m in f.METHODS},policies,support['unsupported_contexts'])
    result = s.select_methods(signals,targets,index,rules,policies)
    claim = dict(purpose='fresh_four_method_policy_selection_v1',inputs=inputs,environment=environment or {},
        cases=48,assigned_groups=12,nonempty_groups=12,blocked_cases_retained=396,
        primary_method=f.METHODS[0],methods=f.METHODS,model_bindings=models,
        N2_calibration_sha256=f.d.checked_json(f.bundle_at(root)/'n2-calibration-v1/completion-v1','calibration')[1],
        definition=rules,candidates=policies,context_support=support,
        N1_refitting_authorized=False,N2_refitting_authorized=False,future_reevaluation_authorized=False,
        outer_evaluation_authorized=False,production_release_authorized=False,formal_conditional_risk_guarantee_established=False)
    return s.save_results(root,claim,index,targets,contexts,signals,safety,result)


def make_n2(root, inputs=None):
    ids = [f"s{i}" for i in range(20)]
    index = pd.DataFrame(dict(scenario_id=ids, group_id=[f"g{i}" for i in range(20)], split_role=["conformal_calibration"]*20))
    targets = pd.DataFrame({t: np.arange(20)*.001 for t in f.d.TARGETS}, index=pd.Index(ids, name="scenario_id"))
    predicted = pd.DataFrame({t+"__"+label: np.full(20, val) for t in f.d.TARGETS
        for label, val in (("lower", -.01), ("median", 0.), ("upper", .01))}, index=targets.index)
    forecasts = {method: predicted.copy() for method in ("hybrid_ridge_lightgbm_lightgbm", *f.base.base.b.METHODS)}
    groups = pd.DataFrame(dict(group_id=[f"g{i}" for i in range(24)], admitted=[1]*20+[0]*4, blocked=[36]*20+[37]*4))
    protocol = json.loads((C4 / "configs/benchmark_protocol.json").read_text())
    claim = dict(purpose="fresh_four_method_N2_group_max_calibration_v1", environment={}, inputs=inputs or {},
        definition=protocol["N2"], model_bindings={m: {"fixture": True} for m in forecasts},
        assigned_groups=24, nonempty_groups=20, all_blocked_groups=4, cases=20, blocked_cases_retained=868,
        support_note="Numerical support only", limitation="Synthetic fixture", production_release_authorized=False)
    output = n.save_results(root, claim, index, targets, forecasts, groups, n.fit_methods(index, targets, forecasts))
    return output


def setup(root,manifest):
    protocol = json.loads((C4/'configs/benchmark_protocol.json').read_text())
    protocol['protocol_version'] = f.v.VERSION
    inputs = {}
    for path in (root/'uci_adult_data/reference.csv',root/'uci_adult_data/test.csv',
                 root/'c4_artifacts/champion/adult'/protocol['source_versions']['champion']/'metadata.json'):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'frozen-fixture-input')
        inputs[str(path)] = f.d.sha(path.read_bytes())
    model = root/'c4_artifacts/champion/adult'/protocol['source_versions']['champion']/'champion.ubj'
    model.write_bytes(b'native-fixture')
    split = root/'c4_artifacts/splits/adult'/protocol['source_versions']['split_manifest']/'manifest.json'
    split.parent.mkdir(parents=True); split.write_bytes(b'split-fixture')
    make_n2(root,inputs)
    directory = root/'c4_artifacts/benchmark_protocols/adult'/f.v.VERSION
    ph = doc(directory,'protocol',protocol); mh = doc(directory,'scenario_manifest',manifest)
    fh = doc(directory,'freeze',dict(protocol_sha256=ph,scenario_manifest_sha256=mh,
        champion_model_sha256=f.d.sha(model.read_bytes()),split_manifest_sha256=f.d.sha(split.read_bytes())))
    doc(f.bundle_at(root)/'admission-executor-v1','claim',dict(compatibility_hashes=dict(protocol=ph,scenario_manifest=mh,freeze=fh)))
    _,calibration,_ = f.base.verify_n2(root)
    output = make_selection(root,calibration['model_bindings'])
    return output,model


def test_selection_full_chain_replay_and_no_outer_claim(tmp_path,manifest):
    setup(tmp_path,manifest)
    claim,result,pins = f.verify_selection(tmp_path,C4,full_inputs=True,replay=True)
    assert claim['cases'] == 48 and len(result['methods']) == 4
    assert any(p.endswith('selection.json') for p in pins)
    assert not (f.bundle_at(tmp_path)/'outer-admission-executor-v1').exists()


@pytest.mark.parametrize('change',['missing','checksum','scope','method','rule','selected_policy','unbound_source'])
def test_invalid_selection_rejected(tmp_path,manifest,change):
    if change == 'missing':
        with pytest.raises(FileNotFoundError): f.verify_selection(tmp_path)
        return
    output,_ = setup(tmp_path,manifest)
    if change == 'checksum': (output/'selection.json').write_bytes(b'changed')
    elif change == 'scope':
        metadata,_ = f.d.checked_json(output); metadata['outer_test_evaluated'] = True; doc(output,'metadata',metadata)
    elif change == 'unbound_source':
        original = f.bundle_at(tmp_path)/'admission-executor-v1/claim.json'; original.write_text('{}')
    else:
        result,_ = f.d.checked_json(output,'selection')
        if change == 'method': result['methods'].pop(f.METHODS[0])
        elif change == 'rule': result['definition']['selection']['minimum_auto_deploy_groups'] = 1
        else: result['methods'][f.METHODS[0]]['global']['selected_policy'] = dict(policy_id='force',kind='global',buffer=0.)
        digest = doc(output,'selection',result)
        metadata,_ = f.d.checked_json(output); metadata['files']['selection.json'] = digest
        metadata['files']['selection.sha256'] = f.d.sha((output/'selection.sha256').read_bytes()); doc(output,'metadata',metadata)
    with pytest.raises(ValueError): f.verify_selection(tmp_path,C4,full_inputs=True,replay=True)


def test_outer_claim_child_scope_environment_and_policy_hash(tmp_path,manifest,monkeypatch):
    output,model = setup(tmp_path,manifest)
    import _hybrid_fresh_candidate
    monkeypatch.setattr(_hybrid_fresh_candidate,'environment',lambda:{})
    claim,actual = f.build_execution(tmp_path,C4)
    assert actual == manifest and claim['planned_groups'] == 16 and claim['planned_cases'] == 592
    assert claim['policy_fixed_before_outer_admission'] is True
    assert all(claim[k] is False for k in ('candidate_training_authorized','future_evaluation_authorized','N1_refitting_authorized','N2_refitting_authorized','policy_fitting_authorized'))
    f.ensure_documents(f.bundle_at(tmp_path)/'outer-admission-executor-v1',{'claim':claim})
    assert f.verify_execution_binding(tmp_path,claim['allowed_group_ids'][0]) == manifest
    for role in ('conformal_calibration','policy_calibration'):
        gid = next(g['group_id'] for g in manifest['groups'] if g['split_role'] == role)
        with pytest.raises(ValueError): f.verify_execution_binding(tmp_path,gid)
    (output/'selection.json').write_bytes(b'changed')
    with pytest.raises(ValueError): f.verify_execution_binding(tmp_path,claim['allowed_group_ids'][0])


def test_bounded_resume_no_replacement_and_all_blocked_retained(tmp_path,manifest,monkeypatch):
    gid = next(g['group_id'] for g in manifest['groups'] if g['split_role'] == 'outer_test')
    doc(tmp_path/'c4_artifacts/benchmark_protocols/adult'/f.v.VERSION,'freeze',{})
    doc(f.bundle_at(tmp_path)/'outer-admission-executor-v1','claim',{})
    monkeypatch.setattr(f,'verify_execution_binding',lambda *a:manifest)
    records,calls = {},[]
    monkeypatch.setattr(r,'existing_record',lambda root,row,digest:records.get(row['scenario_id']))
    def runner(command,**kwargs):
        assert command[1].endswith('run_hybrid_outer_case_admission.py')
        sid = command[command.index('--scenario-id')+1]; calls.append(sid)
        records[sid] = dict(scenario_id=sid,simulation_admission='BLOCKED_ESCALATE',screen_status='FAIL')
        return SimpleNamespace(returncode=0)
    r.run_group(tmp_path,C4,gid,manifest,1,runner)
    assert len(calls) == 1 and r.completed_group(tmp_path,gid,manifest) is None
    r.run_group(tmp_path,C4,gid,manifest,37,runner)
    assert len(calls) == len(set(calls)) == 37
    complete = r.completed_group(tmp_path,gid,manifest)
    assert complete['admitted'] == 0 and complete['blocked'] == 37
    r.run_group(tmp_path,C4,gid,manifest,37,lambda *a,**k:pytest.fail('Must resume'))


def test_failed_child_keeps_log_and_stops(tmp_path,manifest,monkeypatch):
    gid = next(g['group_id'] for g in manifest['groups'] if g['split_role'] == 'outer_test')
    doc(tmp_path/'c4_artifacts/benchmark_protocols/adult'/f.v.VERSION,'freeze',{})
    doc(f.bundle_at(tmp_path)/'outer-admission-executor-v1','claim',{})
    monkeypatch.setattr(r,'existing_record',lambda *a:None)
    with pytest.raises(ValueError,match='preserve'):
        r.run_group(tmp_path,C4,gid,manifest,37,lambda *a,**k:SimpleNamespace(returncode=1))
    assert len(list((tmp_path/'c4_artifacts/benchmark_runs/adult'/f.v.VERSION/'group_admissions'/gid/'v1').glob('*.log'))) == 1


@pytest.mark.parametrize('complete',[False,True])
def test_full_sixteen_group_accounting(tmp_path,manifest,monkeypatch,complete):
    gids = [g['group_id'] for g in manifest['groups'] if g['split_role'] == 'outer_test']
    def record(root,gid,source):
        if not complete and gid == gids[-1]: return None
        return dict(group_id=gid,group_admissions_sha256='fixture',admitted=0,blocked=37)
    monkeypatch.setattr(r,'completed_group',record)
    r.save_completion(tmp_path,dict(allowed_group_ids=gids),manifest)
    output = f.bundle_at(tmp_path)/'outer-admission-executor-v1/completion-v1'
    if not complete:
        assert not output.exists(); return
    summary,_ = f.d.checked_json(output,'outer_admissions')
    assert summary['assigned_groups'] == len(summary['records']) == 16 and summary['blocked_cases'] == 592
    assert summary['admitted_cases'] == summary['nonempty_groups'] == 0
    assert summary['outer_admissions_executed'] is True and summary['outer_future_evaluation_executed'] is False


def test_preexisting_outer_artifacts_without_executor_rejected(tmp_path,manifest,monkeypatch):
    setup(tmp_path,manifest)
    import _hybrid_fresh_candidate
    monkeypatch.setattr(_hybrid_fresh_candidate,'environment',lambda:{})
    row = next(row for row in manifest['scenarios'] if row['split_role'] == 'outer_test')
    (tmp_path/'c4_artifacts/benchmark_runs/adult'/f.v.VERSION/row['scenario_id']).mkdir(parents=True)
    with pytest.raises(ValueError,match='predate'):
        f.build_execution(tmp_path,C4)
    assert not (f.bundle_at(tmp_path)/'outer-admission-executor-v1').exists()


@pytest.mark.parametrize('flag',['future_evaluation_authorized','candidate_training_authorized','policy_fitting_authorized','N2_refitting_authorized'])
def test_child_rejects_unauthorized_training_or_future_stage(tmp_path,manifest,monkeypatch,flag):
    setup(tmp_path,manifest)
    import _hybrid_fresh_candidate
    monkeypatch.setattr(_hybrid_fresh_candidate,'environment',lambda:{})
    claim,_ = f.build_execution(tmp_path,C4)
    claim[flag] = True
    doc(f.bundle_at(tmp_path)/'outer-admission-executor-v1','claim',claim)
    with pytest.raises(ValueError,match='Unauthorized'):
        f.verify_execution_binding(tmp_path,claim['allowed_group_ids'][0])
