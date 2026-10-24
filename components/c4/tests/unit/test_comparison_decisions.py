"""Baseline/ablation branches never receive observed future outcomes."""
import copy
import pandas as pd
import pytest
from c4.risk.benchmark_n3 import TARGETS,BOUNDARIES
from c4.risk.benchmark_policy import GATES,policy_decision
from c4.risk.benchmark_policy_selection import policy_statistics
from c4.evaluation.comparison_decisions import METHOD_IDS,decide_comparison,policy_for_active_targets
from c4.evaluation.comparison_calibration import fit_comparison_calibration


def evidence():return {'c1_harm_width':.01,'c2_diagnosis_stability':.9,'c3_fidelity_score':.9,'c3_utility_score':.01,
    'c4_global_candidate_minus_champion_accuracy':.01,'c4_affected_slice_candidate_minus_champion_accuracy':.01}


def fixture():
    raw={t+'__'+label:b+value for t,b in BOUNDARIES.items() for label,value in [('lower',.015),('median',.02),('upper',.025)]}
    frame=pd.DataFrame([raw],index=['s'])
    base={'comparison':'ORIGINAL','mode':'joint','active_targets':list(TARGETS),'joint_alpha':.05,
          'nonempty_groups':37,'rank':37,'fallback':False,'auto_deploy_supported':True,'adjustments':dict.fromkeys(TARGETS,.001)}
    calibrations={'ORIGINAL':base}
    policies={'PROPOSED':{'policy_id':'measured-review','kind':'review_only'},'B8':{'policy_id':'measured-review','kind':'review_only'}}
    for comparison in ('A1','A2','A3','A4','A5','B5'):
        cal=copy.deepcopy(base);cal['comparison']=comparison
        if comparison=='A4':cal['active_targets']=[TARGETS[0],TARGETS[2]];cal['adjustments']={t:.001 for t in cal['active_targets']}
        if comparison=='A5':cal['mode']='point_joint'
        if comparison=='B5':cal.update(mode='bonferroni',rank=38,fallback=True,auto_deploy_supported=False,adjustments=dict.fromkeys(TARGETS,None))
        calibrations[comparison]=cal
        policies[comparison]={'policy_id':comparison+'-test','kind':'global','buffer':.005}
    policies['A6']={'policy_id':'A6-test','kind':'global','buffer':.005}
    return raw,calibrations,policies


def run(comparison,features=None,gates=None,**kwargs):
    raw,calibrations,policies=fixture()
    return decide_comparison(comparison,evidence() if features is None else features,
        dict.fromkeys(GATES,True) if gates is None else gates,raw,calibrations,policies,**kwargs)


@pytest.mark.parametrize('comparison',METHOD_IDS)
def test_every_method_retains_mandatory_gate_failure(comparison):
    gates=dict.fromkeys(GATES,True);gates['privacy_check']=False
    result=run(comparison,gates=gates)
    assert result['decision']=='ESCALATE' and result['release_authorized'] is False


def test_B1_uses_pre_release_global_margin():
    assert run('B1')['decision']=='AUTO_DEPLOY'
    features=evidence();features['c4_global_candidate_minus_champion_accuracy']=-.001
    assert run('B1',features)['decision']=='ESCALATE'


def test_B2_also_requires_diagnosed_slice_margin():
    features=evidence();features['c4_affected_slice_candidate_minus_champion_accuracy']=-.01
    assert run('B1',features)['decision']=='AUTO_DEPLOY'
    assert run('B2',features)['decision']=='ESCALATE'


def test_B3_uses_frozen_heuristic_threshold():
    assert run('B3')['decision']=='AUTO_DEPLOY'
    features=evidence();features.update(c1_harm_width=1.,c2_diagnosis_stability=0.,c3_fidelity_score=0.,c3_utility_score=-1.)
    assert run('B3',features)['decision']=='HUMAN_REVIEW'


def test_B4_uses_medians_not_calibrated_interval_policy():
    assert run('B4')['decision']=='AUTO_DEPLOY'
    raw,cal,policies=fixture();raw[TARGETS[0]+'__median']=-.001
    assert decide_comparison('B4',evidence(),dict.fromkeys(GATES,True),raw,cal,policies)['decision']=='ESCALATE'


@pytest.mark.parametrize('comparison',['B5','A6'])
def test_Bonferroni_comparison_cannot_deploy(comparison):
    assert run(comparison)['decision']=='HUMAN_REVIEW'


def test_constant_baselines_after_gates():
    assert run('B6')['decision']=='AUTO_DEPLOY'
    assert run('B7')['decision']=='HUMAN_REVIEW'


@pytest.mark.parametrize('comparison',['PROPOSED','B8','A8'])
def test_measured_review_only_policies_remain_disabled(comparison):
    result=run(comparison)
    assert result['decision']=='HUMAN_REVIEW' and result['reason']=='policy_selection_unsupported'


def test_A7_uses_fixed_zero_buffer_without_selection_fallback():
    assert run('A7')['decision']=='AUTO_DEPLOY'
    assert run('A7',unsupported_contexts=['LOW'])['decision']=='HUMAN_REVIEW'


def test_A4_decision_can_ignore_slice_but_future_unsafe_oracle_cannot():
    raw,cal,policies=fixture()
    raw[TARGETS[1]+'__lower']=-.04;raw[TARGETS[1]+'__median']=-.035;raw[TARGETS[1]+'__upper']=-.03
    result=decide_comparison('A4',evidence(),dict.fromkeys(GATES,True),raw,cal,policies)
    assert result['decision']=='AUTO_DEPLOY'
    ix=[{'scenario_id':'s','group_id':'g','split_role':'policy_calibration'}]
    actual={'s':{TARGETS[0]:.01,TARGETS[1]:-.01,TARGETS[2]:0.}}
    metrics=policy_statistics({'s':result['decision']},actual,ix)
    assert metrics['unsafe_auto_deploy_count']==1


def test_A4_positive_buffer_applies_only_to_retained_targets():
    bounds={TARGETS[0]:[.01,.02],TARGETS[2]:[-.01,0.]}
    p={'kind':'global','policy_id':'test','buffer':.005}
    assert policy_for_active_targets(bounds,evidence(),dict.fromkeys(GATES,True),supported=True,policy=p)['decision']=='AUTO_DEPLOY'
    p['buffer']=.015
    assert policy_for_active_targets(bounds,evidence(),dict.fromkeys(GATES,True),supported=True,policy=p)['decision']=='HUMAN_REVIEW'


def test_three_target_path_is_identical_to_existing_policy_engine():
    raw,cal,policies=fixture();bounds={t:[raw[t+'__lower'],raw[t+'__upper']] for t in TARGETS}
    gates=dict.fromkeys(GATES,True);policy=policies['A1']
    assert policy_for_active_targets(bounds,evidence(),gates,supported=True,policy=policy)==policy_decision(
        bounds,evidence(),gates,n2_supported=True,policy=policy)


def test_future_targets_are_not_a_decision_function_argument():
    raw,cal,policies=fixture()
    with pytest.raises(TypeError):
        decide_comparison('A1',evidence(),dict.fromkeys(GATES,True),raw,cal,policies,targets={TARGETS[0]:.1})


def test_preparation_cli_saves_all_runtime_paths_on_synthetic_development_data(tmp_path):
    """Actual preparation CLI with synthetic input bindings; never research evidence."""
    import argparse
    import hashlib
    import io
    import json
    import runpy
    import sys
    from pathlib import Path
    from types import ModuleType
    from unittest.mock import patch
    from c4.risk.benchmark_policy_selection import select_policy
    common=ModuleType('_common')
    common.sha=lambda data:hashlib.sha256(data).hexdigest()
    def save_json(path,payload):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        data=(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode()
        with path.open('xb') as file:file.write(data)
        digest=common.sha(data);path.with_suffix('.sha256').write_text(digest);return digest
    def checked_bytes(path,digest):
        data=Path(path).read_bytes()
        if common.sha(data)!=digest:raise ValueError('Checksum differs')
        return data
    def checked_json(directory,stem='metadata'):
        path=Path(directory)/(stem+'.json');digest=path.with_suffix('.sha256').read_text()
        return json.loads(checked_bytes(path,digest)),digest
    def require(condition,message):
        if not condition:raise ValueError(message)
    def parser(doc):
        p=argparse.ArgumentParser(description=doc);p.add_argument('--data-root');return p
    common.parser=parser;common.resolve_data_root=lambda value:tmp_path;common.require=require;common.name=str
    common.checked_json=checked_json;common.checked_bytes=checked_bytes;common.save_json=save_json
    common.utc_now=lambda:'synthetic-test-only';common.git_context=lambda:{'working_tree_dirty':False,'git_revision':'synthetic-fixture'}
    module=ModuleType('_comparison_inputs');module.VERSION='adult-simulation-pp1-v2';module.SPEC_VERSION='comparison-calibration-v1'
    raw,calibrations,policies=fixture()
    original=copy.deepcopy(calibrations['ORIGINAL']);original['adjustments']=dict.fromkeys(TARGETS,0.)
    binding={'original_calibration':original,'original_selected_policies':{'PROPOSED':policies['PROPOSED'],'B8':policies['B8']},
        'unsupported_contexts':[],'policy_selection_rule':{'minimum_auto_deploy_groups':3,'empirical_conditional_unsafe_ceiling':.05},
        'evaluation_specification_sha256':'synthetic-spec','ablation_run_metadata_sha256':'synthetic-models',
        'controller_freeze_sha256':'synthetic-controller','source_sha256':{}}
    def table(rows,groups,role):
        ids=[f'{role}-s{i}' for i in range(rows)]
        features=pd.DataFrame([evidence() for _ in ids],index=ids)
        targets=pd.DataFrame([{t:b+.02 for t,b in BOUNDARIES.items()} for _ in ids],index=ids)
        ix=pd.DataFrame({'scenario_id':ids,'group_id':[f'{role}-g{i%groups}' for i in range(rows)],
                         'split_role':['policy_calibration' if role=='policy' else 'conformal_calibration']*rows})
        fixed=pd.DataFrame([raw for _ in ids],index=ids)
        return {'features':features,'targets':targets,'index':ix,'original_forecasts':fixed}
    tables={'calibration':table(150,37,'calibration'),'policy':table(44,11,'policy')}
    gates={sid:dict.fromkeys(GATES,True) for sid in tables['policy']['features'].index}
    module.load_inputs=lambda root:({},tables,gates,binding)
    module.fixed_predictions=lambda models,features:{m:pd.DataFrame([raw for _ in features.index],index=features.index)
                                                   for m in ('ORIGINAL','A1','A2','A3')}
    base=tmp_path/'c4_artifacts/comparison_calibrations/adult'/module.VERSION/module.SPEC_VERSION
    save_json(base/'specification.json',{'binding':binding})
    script=Path(__file__).resolve().parents[2]/'scripts/prepare_comparison_artifacts.py'
    with patch.dict(sys.modules,{'_common':common,'_comparison_inputs':module}), \
         patch.object(sys,'argv',['prepare_comparison_artifacts.py','--prepare']):
        runpy.run_path(str(script),run_name='__main__')
    output=base/'prepared-v1';runtime,_=checked_json(output,'comparison_runtime');md,_=checked_json(output)
    assert runtime['method_ids']==list(METHOD_IDS)
    assert runtime['calibrations']['B5']['fallback'] is True
    assert runtime['selected_policies']['A1']['kind']=='adaptive'
    assert runtime['selected_policies']['B5']['kind']=='review_only'
    assert runtime['selected_policies']['PROPOSED']['kind']=='review_only'
    decisions=pd.read_csv(output/'development_decisions.csv')
    assert len(decisions)==17*44
    assert decisions.loc[decisions.comparison.eq('PROPOSED')].decision.eq('HUMAN_REVIEW').all()
    assert md['outer_test_evaluated'] is False and md['release_authorized'] is False
    for filename,digest in md['files'].items():checked_bytes(output/filename,digest)
