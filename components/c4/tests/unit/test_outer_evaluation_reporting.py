"""Synthetic known-value checks; these fixtures are not empirical research evidence."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from c4.risk.benchmark_n3 import TARGETS,BOUNDARIES
from c4.risk.benchmark_policy import GATES
from c4.evaluation.comparison_decisions import METHOD_IDS,decide_comparison

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'


def load_modules():
    common=ModuleType('_common');common.C4_ROOT=SCRIPTS.parent
    def require(ok,message):
        if not ok:raise ValueError(message)
    common.require=require;common.sha=lambda b:hashlib.sha256(b).hexdigest()
    def checked_bytes(path,digest):
        content=Path(path).read_bytes();require(common.sha(content)==digest,'Checksum mismatch');return content
    common.checked_bytes=checked_bytes
    def save(path,doc):
        content=(json.dumps(doc,sort_keys=True,allow_nan=False)+'\n').encode();path=Path(path)
        with path.open('xb') as stream:stream.write(content)
        digest=common.sha(content);path.with_suffix('.sha256').write_text(digest);return digest
    common.save_json=save
    def checked_json(path,stem='metadata'):
        path=Path(path);digest=(path/(stem+'.sha256')).read_text().strip()
        return json.loads(checked_bytes(path/(stem+'.json'),digest)),digest
    common.checked_json=checked_json
    for name in ('parser','resolve_data_root','name','utc_now','git_context'):setattr(common,name,lambda *a,**kw:None)
    outer=ModuleType('_outer_pipeline');outer.VERSION='adult-simulation-pp1-v2'
    for name in ('inventory','runs','verified_phase','verify_freeze_dependencies','verify_pre_release_registry'):
        setattr(outer,name,lambda *a,**kw:None)
    def load(name,file):
        spec=importlib.util.spec_from_file_location(name,SCRIPTS/file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    with patch.dict(sys.modules,{'_common':common,'_outer_pipeline':outer}):
        inputs=load('isolated_outer_eval_inputs','_outer_evaluation_inputs.py')
        with patch.dict(sys.modules,{'_outer_evaluation_inputs':inputs}):
            evaluation=load('isolated_outer_reports','evaluate_outer_comparisons.py')
            plots=load('isolated_outer_plots','export_outer_result_plots.py')
    return inputs,evaluation,plots


def fixture():
    sizes=[2,6,2,6,4,2,4,4,4,4,2,0,2,8,0,6];groups=[f'g{i}' for i in range(16)]
    ids=[];group_ids=[]
    for group,n in zip(groups,sizes):
        ids += [f'{group}-s{i:02}' for i in range(n)];group_ids += [group]*n
    feature={'c1_harm_width':.01,'c2_diagnosis_stability':.9,'c3_fidelity_score':.9,'c3_utility_score':.01,
             'c4_global_candidate_minus_champion_accuracy':.01,'c4_affected_slice_candidate_minus_champion_accuracy':.01}
    feature.update({f'padding_{i}':0. for i in range(69)})
    features=pd.DataFrame([feature]*56,index=pd.Index(ids,name='scenario_id'))
    gates=dict.fromkeys(GATES,True)
    index=pd.DataFrame({'scenario_id':ids,'group_id':group_ids,'split_role':'outer_test',
                        'explicit_hard_gates_json':json.dumps(gates)})
    targets=pd.DataFrame([[.02,.02,0.]]*56,index=features.index,columns=list(TARGETS))
    targets.iloc[::2,0]=-.005
    raw={t+'__'+label:b+v for t,b in BOUNDARIES.items() for label,v in [('lower',.015),('median',.02),('upper',.025)]}
    forecasts=pd.DataFrame([raw]*56,index=features.index)
    base={'comparison':'ORIGINAL','mode':'joint','active_targets':list(TARGETS),'joint_alpha':.05,
          'nonempty_groups':37,'rank':37,'fallback':False,'auto_deploy_supported':True,'adjustments':dict.fromkeys(TARGETS,.001)}
    calibrations={'ORIGINAL':base};policies={m:{'policy_id':m+'-review','kind':'review_only'} for m in ('PROPOSED','B8','A1','A2','A3','A4','A5','A6','B5')}
    for method in ('A1','A2','A3','A4','A5','B5'):
        cal=copy.deepcopy(base);cal['comparison']=method
        if method=='A4':cal['active_targets']=[TARGETS[0],TARGETS[2]];cal['adjustments']={t:.001 for t in cal['active_targets']}
        if method=='A5':cal['mode']='point_joint'
        if method=='B5':cal.update(mode='bonferroni',rank=38,fallback=True,auto_deploy_supported=False,adjustments=dict.fromkeys(TARGETS,None))
        calibrations[method]=cal
    runtime={'calibrations':calibrations,'selected_policies':policies,'unsupported_contexts':[]}
    decisions=[]
    for sid,gid in zip(ids,group_ids):
        for method in METHOD_IDS:
            result=decide_comparison(method,feature,gates,raw,calibrations,policies)
            decisions.append({'scenario_id':sid,'group_id':gid,'comparison':method,'decision':result['decision'],'reason':result['reason']})
    admissions=[]
    for gid,n in zip(groups,sizes):
        for i in range(37):
            admissions.append({'scenario_id':f'{gid}-s{i:02}','group_id':gid,'split_role':'outer_test',
                'verified_target_row_available':i<n,'simulation_admission':'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if i<n else 'BLOCKED_ESCALATE'})
    manifest={'groups':[{'group_id':gid,'namespace':42} for gid in groups],
              'scenarios':[{'scenario_id':sid,'mechanism':'interaction','magnitude':.2,
                            'diagnosis_condition':'exact_slice','repair_condition':'targeted_repair'} for sid in ids]}
    return {'features':features,'targets':targets,'index':index,'groups':groups,'runtime':runtime,
        'decisions':pd.DataFrame(decisions),'gates':dict.fromkeys(ids,gates),'manifest':manifest,
        'forecasts':{v:forecasts.copy() for v in ('ORIGINAL','A1','A2','A3','linear')},
        'admissions':pd.DataFrame(admissions),'group_summary':pd.DataFrame([
            {'group_id':gid,'target_rows':n,'blocked':37-n,'all_blocked':n==0} for gid,n in zip(groups,sizes)])}


@pytest.fixture(scope='module')
def computed():
    inputs,evaluation,plots=load_modules();data=fixture();inputs.validate_inventory(data)
    return inputs,evaluation,plots,data,evaluation.compute_reports(data)


def test_full_outer_inventory_accepts_empty_groups_without_imputing_targets():
    inputs,_,_=load_modules();inputs.validate_inventory(fixture())


@pytest.mark.parametrize('mutation',['mixed_fold','duplicate_decision','blocked_target','missing_empty_group','group_mismatch'])
def test_inventory_rejects_leakage_or_missing_accounting(mutation):
    inputs,_,_=load_modules();data=fixture()
    if mutation=='mixed_fold':data['index'].loc[0,'split_role']='policy_calibration'
    elif mutation=='duplicate_decision':data['decisions'].iloc[1]=data['decisions'].iloc[0]
    elif mutation=='blocked_target':data['admissions'].loc[36,'verified_target_row_available']=True
    elif mutation=='missing_empty_group':data['groups'].remove('g11')
    else:data['decisions'].loc[0,'group_id']='g15'
    with pytest.raises(ValueError):inputs.validate_inventory(data)


def test_outer_results_are_labelled_heldout_pilot_not_development(computed):
    _,_,_,_,reports=computed
    assert reports['outer_test_evaluated'] is True and 'held_out_outer' in reports['evaluation']
    assert len(reports['governance'])==17


def test_review_only_conditional_risk_and_bootstrap_remain_undefined(computed):
    _,_,_,_,reports=computed;r=reports['governance']['PROPOSED']
    assert r['counts']['review']==56 and r['counts']['auto']==0
    assert r['equal_group_conditional_unsafe_rate'] is None
    result=reports['statistics']['methods']['PROPOSED']['equal_group_conditional_unsafe_rate']
    assert result['percentile_95_ci'] is None and result['undefined_replicates']==2000


def test_empty_assigned_groups_remain_in_shared_bootstrap_and_b8_a8_match(computed):
    _,_,_,_,reports=computed
    assert len(reports['statistics']['assigned_group_ids'])==16
    assert reports['governance']['PROPOSED']['groups_without_target_rows']==2
    assert reports['governance']['B8']==reports['governance']['A8']
    assert reports['statistics']['paired_draws_shared_across_methods'] is True


def test_two_target_and_full_bound_fallback_semantics_retained(computed):
    _,_,_,_,reports=computed
    assert reports['interval']['A4']['three_target_comparable'] is False
    for name in ('B5','A6'):
        assert reports['interval'][name]['fallback'] is True
        assert reports['interval'][name]['targets'][TARGETS[0]]['equal_group_weighted']['mean_width_fraction']==2.
    assert 'linear' in reports['forecast']


def test_frozen_decision_tampering_is_rejected():
    _,evaluation,_=load_modules();data=fixture()
    data['decisions'].loc[data['decisions'].comparison.eq('PROPOSED'),'decision']='AUTO_DEPLOY'
    with pytest.raises(ValueError,match='frozen executable'):evaluation.compute_reports(data)


def test_report_save_uses_binary_markdown_and_preserves_input_frames(tmp_path,computed):
    _,evaluation,_,data,reports=computed
    before=data['targets'].copy(deep=True);output=tmp_path/'report'
    binding={'recovered_snapshot_metadata_sha256':'test-synthetic'}
    digest=evaluation.save_reports(output,reports,data,binding,{'git_revision':'test','working_tree_dirty':False})
    assert hashlib.sha256((output/'metadata.json').read_bytes()).hexdigest()==digest
    md=json.loads((output/'metadata.json').read_bytes())
    assert md['outer_test_evaluated'] is True and md['independent_dataset_evaluated'] is False
    note=(output/'C4_OUTER_EVALUATION_EVIDENCE.md').read_bytes()
    assert b'\r' not in note and b'undefined' in note
    for filename,expected in md['files'].items():assert hashlib.sha256((output/filename).read_bytes()).hexdigest()==expected
    pd.testing.assert_frame_equal(data['targets'],before)
    with pytest.raises(ValueError,match='exists'):evaluation.save_reports(output,reports,data,binding,{})


def test_plots_handle_undefined_risk_and_generate_separate_figures(computed):
    pytest.importorskip('matplotlib')
    _,_,plots,_,reports=computed
    figures=plots.render_figures(reports['governance'],reports['forecast'],reports['interval'],reports['diagnostic_risk_coverage'])
    assert len(figures)==6
    assert any('Undefined' in t.get_text() for ax in figures['outer_conditional_risk'].axes for t in ax.texts)
    import matplotlib.pyplot as plt
    for figure in figures.values():plt.close(figure)
