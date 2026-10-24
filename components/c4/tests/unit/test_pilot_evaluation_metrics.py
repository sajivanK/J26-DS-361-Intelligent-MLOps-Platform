"""Known-value metrics, target boundaries, coverage populations and score ties."""
import math
import numpy as np
import pandas as pd
import pytest
from c4.risk.benchmark_n3 import TARGETS,BOUNDARIES
from c4.evaluation.pilot_evaluation_metrics import (governance_report,group_counts,metrics_from_counts,
    forecast_report,interval_report,diagnostic_risk_coverage)


def fixture():
    ids=['a','b','c','d']
    y=pd.DataFrame([dict(BOUNDARIES) for _ in ids],index=ids)
    y.loc['a',TARGETS[0]]=-.01;y.loc['b',TARGETS[1]]=-.01
    ix=pd.DataFrame({'scenario_id':ids,'group_id':['g1','g2','g2','g2'],'split_role':['policy_calibration']*4})
    return y,ix,['g1','g2','empty']


def test_equal_group_and_scenario_risks_have_distinct_denominators():
    y,ix,groups=fixture();d=dict.fromkeys(y.index,'AUTO_DEPLOY')
    r=governance_report(y,ix,d,groups)
    assert r['equal_group_conditional_unsafe_rate']==pytest.approx((1+1/3)/2)
    assert r['scenario_weighted']['conditional_unsafe_rate']==.5
    assert r['equal_group_unsafe_auto_frequency']==pytest.approx((1+1/3)/2)
    assert r['counts']['unsafe_auto']==2 and r['counts']['n']==4
    assert r['assigned_groups']==3 and r['groups_without_target_rows']==1


def test_review_only_has_zero_automation_but_undefined_conditional_risk():
    y,ix,groups=fixture();r=governance_report(y,ix,dict.fromkeys(y.index,'HUMAN_REVIEW'),groups)
    assert r['equal_group_automation']==r['equal_group_unsafe_auto_frequency']==0
    assert r['equal_group_conditional_unsafe_rate'] is None
    assert r['scenario_weighted']['conditional_unsafe_rate'] is None
    assert r['human_review_burden']==1 and r['expected_deployment_cost']==1
    assert r['ratio_group_support']['equal_group_conditional_unsafe_rate']==0


def test_safe_automation_coverage_differs_from_safe_yield():
    y,ix,groups=fixture();r=governance_report(y,ix,dict.fromkeys(y.index,'AUTO_DEPLOY'),groups)
    assert r['safe_automation_coverage']==1
    assert r['safe_automation_yield']==pytest.approx(1/3)


def test_cost_uses_unsafe_deployments_review_and_safe_escalation():
    y,ix,groups=fixture();d={'a':'AUTO_DEPLOY','b':'ESCALATE','c':'HUMAN_REVIEW','d':'ESCALATE'}
    r=governance_report(y,ix,d,groups)
    assert r['expected_deployment_cost']==pytest.approx((10+(0+1+2)/3)/2)
    assert r['correct_escalation_recall']==.5
    assert r['escalation_precision']==.5
    assert r['cost_sensitivity']['1.0']==pytest.approx(1.)


def test_boundary_equality_is_safe():
    y,ix,groups=fixture();y.loc[:,:]=[BOUNDARIES[t] for t in TARGETS]
    r=governance_report(y,ix,dict.fromkeys(y.index,'AUTO_DEPLOY'),groups)
    assert r['counts']['unsafe_auto']==0 and r['equal_group_conditional_unsafe_rate']==0


def test_empty_assigned_group_does_not_receive_labels():
    y,ix,groups=fixture();r=governance_report(y.iloc[:0],ix.iloc[:0],{},groups)
    assert r['counts']['n']==0 and r['nonempty_groups']==0
    assert r['equal_group_automation'] is None and r['expected_deployment_cost'] is None


def test_mixed_folds_rejected():
    y,ix,groups=fixture();ix.loc[0,'split_role']='outer_test'
    with pytest.raises(ValueError,match='never mix folds'):
        governance_report(y,ix,dict.fromkeys(y.index,'HUMAN_REVIEW'),groups)


def test_training_fold_not_evaluated_as_heldout():
    y,ix,groups=fixture();ix['split_role']='forecast_train'
    with pytest.raises(ValueError,match='Evaluate policy or outer'):
        governance_report(y,ix,dict.fromkeys(y.index,'HUMAN_REVIEW'),groups)


def test_duplicate_assigned_group_rejected():
    y,ix,groups=fixture()
    with pytest.raises(ValueError,match='Unique assigned'):
        group_counts(y,ix,dict.fromkeys(y.index,'HUMAN_REVIEW'),groups+['g1'])


def forecasts(index):
    return pd.DataFrame({t+'__'+name:np.full(len(index),value) for t in TARGETS
        for name,value in [('lower',-.1),('median',0.),('upper',.1)]},index=index)


def test_forecast_mae_rmse_pinball_known_values():
    y,ix,groups=fixture();y.loc[:,:]=0.;y.loc['a',:]=.1
    r=forecast_report(y,forecasts(y.index),ix,groups)[TARGETS[0]]
    assert r['scenario_weighted']['mae']==pytest.approx(.025)
    assert r['scenario_weighted']['rmse']==pytest.approx(.05)
    assert r['scenario_weighted']['pinball']['0.5']==pytest.approx(.0125)
    assert r['equal_group_weighted']['mae']==pytest.approx(.05)


def test_constant_target_R2_is_null_even_for_perfect_prediction():
    y,ix,groups=fixture();y.loc[:,:]=0.
    r=forecast_report(y,forecasts(y.index),ix,groups)
    assert all(v['scenario_weighted']['r2'] is None and v['equal_group_weighted']['r2'] is None for v in r.values())


def test_interval_score_and_normalized_width():
    y,ix,groups=fixture();y.loc[:,:]=0.;y.loc['a',TARGETS[0]]=.2
    f=forecasts(y.index);r=interval_report(y,f,ix,groups,TARGETS,native_alpha=.05/3)
    m=r['targets'][TARGETS[0]]['scenario_weighted']
    assert m['coverage']==.75 and m['mean_width_fraction']==pytest.approx(.2)
    assert m['mean_width_normalized_range_2']==pytest.approx(.1)
    assert m['interval_score_alpha_005']==pytest.approx(1.2)
    assert m['interval_score_native_bonferroni']==pytest.approx(3.2)
    assert r['group_all_variant_simultaneous_coverage']==.5


def test_A4_two_target_joint_coverage_is_explicitly_not_three_target():
    y,ix,groups=fixture();y.loc[:,:]=0.;y.loc['a',TARGETS[1]]=.9
    r=interval_report(y,forecasts(y.index),ix,groups,(TARGETS[0],TARGETS[2]))
    assert r['scenario_simultaneous_coverage']==1 and r['three_target_comparable'] is False


def test_reversed_interval_rejected():
    y,ix,groups=fixture();f=forecasts(y.index);f[TARGETS[0]+'__lower']=.5
    with pytest.raises(ValueError,match='Invalid bounded'):
        interval_report(y,f,ix,groups,TARGETS)


def test_risk_curve_adds_tied_scores_as_one_block():
    y,ix,groups=fixture()
    curve=diagnostic_risk_coverage([1.,1.,0.,0.],y,ix,groups,[True]*4)
    assert len(curve['curve'])==2
    assert curve['curve'][0]['auto_scenarios']==2
    # g1 all deployed, g2 one of three; then both groups fully deployed.
    assert curve['curve'][0]['equal_group_coverage']==pytest.approx(2/3)
    assert curve['curve'][0]['conditional_unsafe_rate']==1
    assert curve['aurc']==pytest.approx(2/3+(1/3)*(2/3))


def test_constant_ranking_and_no_score_have_null_AURC():
    y,ix,groups=fixture()
    assert diagnostic_risk_coverage([1.]*4,y,ix,groups,[True]*4)['aurc'] is None
    assert diagnostic_risk_coverage(None,y,ix,groups,[True]*4)['aurc'] is None


def test_failed_gate_rows_cannot_be_forced_into_diagnostic_deployment():
    y,ix,groups=fixture()
    curve=diagnostic_risk_coverage([4.,3.,2.,1.],y,ix,groups,[False,True,True,True])
    assert curve['curve'][-1]['auto_scenarios']==3
    assert curve['curve'][-1]['equal_group_coverage']==.5


def test_development_report_cli_saves_complete_separate_metric_outputs(tmp_path):
    """Execute actual reporting/serialization on synthetic data, not research observations."""
    import argparse
    import copy
    import hashlib
    import json
    import runpy
    import sys
    from pathlib import Path
    from types import ModuleType
    from unittest.mock import patch
    from c4.evaluation.comparison_decisions import METHOD_IDS,decide_comparison
    from c4.risk.benchmark_policy import GATES
    common=ModuleType('_common');common.sha=lambda b:hashlib.sha256(b).hexdigest()
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
    def parser(doc):
        p=argparse.ArgumentParser(description=doc);p.add_argument('--data-root');return p
    common.parser=parser;common.resolve_data_root=lambda value:tmp_path;common.name=str
    common.require=lambda condition,message:None if condition else (_ for _ in ()).throw(ValueError(message))
    common.save_json=save_json;common.checked_json=checked_json;common.checked_bytes=checked_bytes
    common.utc_now=lambda:'synthetic-only';common.git_context=lambda:{'working_tree_dirty':False,'git_revision':'synthetic-only'}
    ids=[f's{i}' for i in range(44)];groups=[f'g{i}' for i in range(12)]
    ix=pd.DataFrame({'scenario_id':ids,'group_id':[groups[i%11] for i in range(44)],'split_role':['policy_calibration']*44})
    y=pd.DataFrame([{t:b+(.01 if n%3 else -.01) for t,b in BOUNDARIES.items()} for n in range(44)],index=ids)
    evidence={'c1_harm_width':.01,'c2_diagnosis_stability':.9,'c3_fidelity_score':.9,'c3_utility_score':.01,
              'c4_global_candidate_minus_champion_accuracy':.01,'c4_affected_slice_candidate_minus_champion_accuracy':.01}
    x=pd.DataFrame([evidence for _ in ids],index=ids)
    f=pd.DataFrame([{t+'__'+label:b+value+(n%5)*.001 for t,b in BOUNDARIES.items()
                    for label,value in [('lower',.015),('median',.02),('upper',.025)]} for n in range(44)],index=ids)
    gates={sid:dict.fromkeys(GATES,True) for sid in ids}
    base={'comparison':'ORIGINAL','mode':'joint','active_targets':list(TARGETS),'joint_alpha':.05,'nonempty_groups':37,
          'rank':37,'fallback':False,'auto_deploy_supported':True,'adjustments':dict.fromkeys(TARGETS,.001)}
    calibrations={'ORIGINAL':base};policies={m:{'policy_id':'synthetic-review','kind':'review_only'} for m in ['PROPOSED','B8','A1','A2','A3','A4','A5','B5','A6']}
    for cid in ['A1','A2','A3','A4','A5','B5']:
        cal=copy.deepcopy(base);cal['comparison']=cid
        if cid=='A4':cal['active_targets']=[TARGETS[0],TARGETS[2]];cal['adjustments']={t:.001 for t in cal['active_targets']}
        if cid=='A5':cal['mode']='point_joint'
        if cid=='B5':cal.update(mode='bonferroni',rank=38,fallback=True,auto_deploy_supported=False,adjustments=dict.fromkeys(TARGETS,None))
        calibrations[cid]=cal
    runtime={'calibrations':calibrations,'selected_policies':policies,'unsupported_contexts':[]}
    rows=[]
    for method in METHOD_IDS:
        for sid in ids:
            r=decide_comparison(method,x.loc[sid].to_dict(),gates[sid],f.loc[sid].to_dict(),calibrations,policies)
            rows.append({'comparison':method,'scenario_id':sid,'decision':r['decision']})
    data={'features':x,'targets':y,'index':ix,'groups':groups,'runtime':runtime,'gates':gates,'admission_cases':444,'blocked_cases':400,
        'decisions':pd.DataFrame(rows),'forecasts':{m:f for m in ('ORIGINAL','A1','A2','A3')},
        'manifest':{'groups':[{'group_id':g,'namespace':42} for g in groups],
          'scenarios':[{'scenario_id':sid,'mechanism':'interaction','magnitude':.2,'diagnosis_condition':'exact_slice',
                        'repair_condition':'targeted_repair'} for sid in ids]}}
    source=ModuleType('_evaluation_engine_inputs');source.VERSION='adult-simulation-pp1-v2'
    source.load_inputs=lambda root:(data,{'synthetic_binding':True})
    path=Path(__file__).resolve().parents[2]/'scripts/evaluate_development_comparisons.py'
    with patch.dict(sys.modules,{'_common':common,'_evaluation_engine_inputs':source}), \
         patch.object(sys,'argv',['evaluate_development_comparisons.py','--save']):
        runpy.run_path(str(path),run_name='__main__')
    output=tmp_path/'c4_artifacts/evaluations/adult'/source.VERSION/'development-engine-check-v1'
    md,_=checked_json(output);r,_=checked_json(output,'governance');statistics,_=checked_json(output,'statistics')
    assert md['independent_evaluation'] is False and md['outer_test_evaluated'] is False
    assert r['results']['PROPOSED']['equal_group_conditional_unsafe_rate'] is None
    assert r['results']['B6']['counts']['unsafe_auto']>0
    assert statistics['results']['methods']['PROPOSED']['equal_group_conditional_unsafe_rate']['valid_replicates']==0
    for filename,digest in md['files'].items():checked_bytes(output/filename,digest)


def test_raw_crossings_and_raw_pinball_are_retained():
    y,ix,groups=fixture();f=forecasts(y.index)
    for target in TARGETS:
        for suffix in ('lower','median','upper'):
            f[target+'__raw_'+suffix]=f[target+'__'+suffix]
    f.loc['a',TARGETS[0]+'__raw_lower']=.3
    r=forecast_report(y,f,ix,groups)[TARGETS[0]]
    assert r['raw_quantile_crossing_rows']==1
    assert r['scenario_weighted']['raw_pinball']['0.05']!=r['scenario_weighted']['pinball']['0.05']
