"""Behavioral checks for outer isolation, admission inventory and future barrier.

No Adult data, GPU, TVAE fitting or saved research models are required.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from types import ModuleType
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts'
COUNTS = [2,6,2,6,4,2,4,4,4,4,2,0,2,8,0,6]
TARGETS = ('future_global_margin','future_slice_margin','future_nonaffected_margin')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def common_stub():
    module = ModuleType('_common')
    module.C4_ROOT = SCRIPTS.parent
    module.require = require
    module.sha = lambda b:hashlib.sha256(b).hexdigest()
    module.checked_json = lambda *a:({},'')
    module.checked_bytes = lambda *a:b''
    return module


def load_helper():
    group = ModuleType('_outer_group')
    group.VERSION = 'adult-simulation-pp1-v2'
    group.outer_groups = lambda m:[g['group_id'] for g in m['groups']]
    group.outer_group_cases = lambda m,g:[r for r in m['scenarios'] if r['group_id']==g]
    group.load_manifest = lambda root:({},'','')
    group.current_bindings = lambda root:{}
    group.source_hashes = lambda:{}
    spec = importlib.util.spec_from_file_location('isolated_outer_execution', SCRIPTS/'_outer_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'_common':common_stub(), '_outer_group':group}):
        spec.loader.exec_module(module)
    return module


def inventory_fixture(helper):
    groups = [f'g{i}' for i in range(16)]
    rows, reports = [], {}
    for gid, count in zip(groups, COUNTS):
        records = []
        for number in range(37):
            row = {'scenario_id':f'{gid}-s{number:02}', 'group_id':gid, 'split_role':'outer_test',
                   'mechanism':'interaction', 'magnitude':.2, 'diagnosis_condition':'exact_slice',
                   'repair_condition':'targeted_repair', 'is_no_drift_control':False}
            rows.append(row)
            records.append({**row, 'screen_status':'PASS' if number<count else 'FAIL',
                'simulation_admission':'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if number<count else 'BLOCKED_ESCALATE'})
        reports[gid] = {'group_id':gid,'cases':37,'records':records,'freeze_sha256':'protocol',
            'outer_admission_freeze_sha256':helper.OUTER_ADMISSION_HASH,
            'n1_binding':{'model':'fixed'},'scoped_admissible_cases':count,'blocked_cases':37-count}
    manifest = {'groups':[{'group_id':g} for g in groups], 'scenarios':rows}
    helper.load_manifest = lambda root:(manifest,'protocol','manifest')
    helper.audit_checkpoint = lambda root:{'groups':groups,'group_admission_sha256':dict.fromkeys(groups,'report'),
                                         'binding':{'n1_binding':{'model':'fixed'}}}
    helper.checked_json = lambda directory,stem='metadata':(reports[Path(directory).parent.name],'report')
    return manifest, reports


def test_exact_56_selection_preserves_16_groups_including_empty():
    h=load_helper();manifest,reports=inventory_fixture(h)
    _,groups,saved,selected=h.inventory(Path('/data'))
    assert len(groups)==16 and len(selected)==56 and len(saved)==16
    assert {r['group_id'] for r in selected}==set(groups)-{'g11','g14'}
    assert all(r['simulation_admission']=='READY_FOR_SCOPED_CANDIDATE_VALIDATION' for r in selected)


@pytest.mark.parametrize('mutation', ['duplicate','wrong_fold','failed_screen','wrong_counts','wrong_n1','unknown_status'])
def test_inventory_rejects_corrupt_or_unapproved_records(mutation):
    h=load_helper();manifest,reports=inventory_fixture(h);r=reports['g0']
    if mutation=='duplicate':r['records'][1]=copy.deepcopy(r['records'][0])
    elif mutation=='wrong_fold':r['records'][0]['split_role']='forecast_train'
    elif mutation=='failed_screen':r['records'][0]['screen_status']='FAIL'
    elif mutation=='wrong_counts':r['scoped_admissible_cases']=3
    elif mutation=='wrong_n1':r['n1_binding']={'model':'changed'}
    else:r['records'][0]['simulation_admission']='AUTO_DEPLOY'
    with pytest.raises(ValueError):h.inventory(Path('/data'))


def test_inventory_never_opens_future_outcomes():
    h=load_helper();inventory_fixture(h);original=h.checked_json;paths=[]
    def checked(path,stem='metadata'):
        paths.append((str(path),stem));return original(path,stem)
    h.checked_json=checked;h.inventory(Path('/data'))
    assert len(paths)==16 and all(stem=='group_admissions' for _,stem in paths)


def test_future_claim_blocks_new_candidate_or_freeze_even_without_outcomes(tmp_path):
    h=load_helper();manifest,_=inventory_fixture(h)
    directory=h.runs(tmp_path)/manifest['scenarios'][0]['scenario_id']/'future_research-v1'
    directory.mkdir(parents=True)
    with pytest.raises(ValueError,match='forbidden'):h.require_no_outer_future(tmp_path)


def test_no_future_claim_allows_first_execution(tmp_path):
    h=load_helper();inventory_fixture(h);h.require_no_outer_future(tmp_path)


def test_phase_runs_full_shared_checks_at_entry_and_exit(tmp_path):
    h=load_helper();calls=[]
    frozen={'binding':{'model':'fixed'}}
    def verify(root,*,full=False):calls.append(full);return frozen,'hash'
    h.verify_execution_spec=verify
    with h.verified_phase(tmp_path,'candidate'):
        assert h.execution_binding(tmp_path,'candidate')=={'model':'fixed'}
        h.execution_binding(tmp_path,'candidate')
    assert calls==[True,False,False,True] and h._ACTIVE is None


def test_phase_rejects_wrong_phase_and_clears_session_after_error(tmp_path):
    h=load_helper();h.verify_execution_spec=lambda *a,**kw:({'binding':{}},'hash')
    with pytest.raises(ValueError,match='phase/root'):
        with h.verified_phase(tmp_path,'candidate'):h.execution_binding(tmp_path,'future')
    assert h._ACTIVE is None


def test_phase_detects_checkpoint_change_before_returning(tmp_path):
    h=load_helper();calls=iter(['before','after'])
    h.verify_execution_spec=lambda *a,**kw:({'binding':{}},next(calls))
    with pytest.raises(ValueError,match='changed during phase'):
        with h.verified_phase(tmp_path,'freeze'):pass
    assert h._ACTIVE is None


def test_nested_phase_is_rejected(tmp_path):
    h=load_helper();h.verify_execution_spec=lambda *a,**kw:({'binding':{}},'hash')
    with h.verified_phase(tmp_path,'audit'):
        with pytest.raises(ValueError,match='nested'):
            with h.verified_phase(tmp_path,'future'):pass


@pytest.mark.parametrize('field,value', [('complete_decision_evidence_frozen',False),('future_evaluation_executed',True),
    ('scope','one_admitted_policy_scenario_under_simulated_upstream_evidence')])
def test_wrong_prefuture_scope_is_rejected(field,value):
    h=load_helper();snapshot={'complete_decision_evidence_frozen':True,'future_evaluation_executed':False,
        'scope':'one_admitted_outer_scenario_under_simulated_upstream_evidence','scenario':{'split_role':'outer_test'}}
    snapshot[field]=value
    with pytest.raises(ValueError,match='scope'):h.verify_freeze_dependencies(snapshot)


def snapshot_fixture(h, path_type=Path):
    n1=path_type('/n1/fit')
    deps={'/comparisons.json':'comparisons','/forecasts.json':'forecasts',str(n1/'metadata.json'):h.N1_RUN_HASH,
        str(n1.parent/'specification.json'):h.N1_SPEC_HASH,str(n1/'lightgbm.joblib'):'lgb',
        '/prepared/metadata.json':h.PREPARED_HASH,'/ablations/metadata.json':h.ABLATION_RUN_HASH,
        '/policy/freeze.json':h.POLICY_SPEC_HASH,'/execution/freeze.json':'execution'}
    h.checked_bytes=lambda *args:json.dumps({'files':{'lightgbm.joblib':'lgb'}}).encode()
    return {'complete_decision_evidence_frozen':True,'future_evaluation_executed':False,
        'scope':'one_admitted_outer_scenario_under_simulated_upstream_evidence','scenario':{'split_role':'outer_test'},
        'n1_binding':{'specification_sha256':h.N1_SPEC_HASH,'run_metadata_sha256':h.N1_RUN_HASH},
        'policy_specification_sha256':h.POLICY_SPEC_HASH,'comparison_forecasts_path':'/comparisons.json',
        'comparison_forecasts_sha256':'comparisons','n1_forecasts_path':'/forecasts.json','n1_forecasts_sha256':'forecasts',
        'n1_run_directory':str(n1),'dependencies':deps,'outer_execution_freeze_sha256':'execution'}


def test_complete_freeze_checks_all_dependency_bytes():
    h=load_helper();snapshot=snapshot_fixture(h);calls=[];original=h.checked_bytes
    def checked(path,digest):calls.append(str(path));return original(path,digest)
    h.checked_bytes=checked;h.verify_freeze_dependencies(snapshot)
    assert set(snapshot['dependencies'])<=set(calls)


@pytest.mark.parametrize('key',[str(Path('/n1/fit')/'lightgbm.joblib'),'/prepared/metadata.json','/ablations/metadata.json','/policy/freeze.json','/execution/freeze.json'])
def test_missing_model_or_comparison_dependency_rejected(key):
    h=load_helper();snapshot=snapshot_fixture(h);snapshot['dependencies'].pop(key)
    with pytest.raises(ValueError):h.verify_freeze_dependencies(snapshot)


def test_registry_cannot_be_created_after_one_outer_future_claim(tmp_path):
    h=load_helper();manifest,reports=inventory_fixture(h)
    h.execution_binding=lambda *a,**kw:{}
    h.inventory=lambda root:(manifest,[g['group_id'] for g in manifest['groups']],reports,[])
    (h.runs(tmp_path)/manifest['scenarios'][0]['scenario_id']/'future_research-v1').mkdir(parents=True)
    with pytest.raises(ValueError,match='before the whole-fold'):h.registry_definition(tmp_path,forbid_future=True)


def test_registry_verifier_rejects_missing_one_admitted_case(tmp_path):
    h=load_helper();expected=[f's{i}' for i in range(56)]
    h.execution_binding=lambda *a,**kw:{'admitted_scenario_ids_in_fixed_order':expected,'group_ids_in_fixed_order':['g']}
    definition={'outer_execution_freeze_sha256':'execution','scenario_freeze_sha256':dict.fromkeys(expected[:-1],'x'),
        'comparison_forecast_sha256':dict.fromkeys(expected,'y'),'admitted_scenarios':56}
    h.checked_json=lambda directory,stem='metadata':({'definition':definition},'registry') if stem=='registry' else ({},'execution')
    with pytest.raises(ValueError,match='inventory'):h.verify_pre_release_registry(tmp_path)


def extracted_recompute():
    tree=ast.parse((SCRIPTS/'audit_outer_snapshot.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='recompute_targets')
    namespace={'np':np,'pd':pd,'require':require,'TARGET_COLUMNS':TARGETS}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'audit_outer_snapshot.py','exec'),namespace)
    return namespace['recompute_targets']


def prediction_fixture():
    predicted=pd.DataFrame({'source_row_id':[10,11,12,13],'target':[0,0,1,1],
                           'candidate_prediction':[0,0,1,0],'champion_prediction':[1,0,0,1]})
    membership=pd.DataFrame({'source_row_id':[10,11,12,13],'true_slice_member':[True,True,False,False]})
    targets=dict(zip(TARGETS,[.25,.5,0.]))
    return predicted,membership,[10,11,12,13],targets


def test_paired_targets_recomputed_for_three_populations():extracted_recompute()(*prediction_fixture())


@pytest.mark.parametrize('mutation',['target','row_order','empty_slice','nonbinary'])
def test_paired_target_audit_rejects_invalid_evidence(mutation):
    p,m,ids,t=prediction_fixture()
    if mutation=='target':t[TARGETS[1]]=0.
    elif mutation=='row_order':m=m.iloc[::-1]
    elif mutation=='empty_slice':m['true_slice_member']=False
    else:p.loc[0,'candidate_prediction']=2
    with pytest.raises(ValueError):extracted_recompute()(p,m,ids,t)


@pytest.mark.parametrize('filename,flag', [('validate_outer_candidate.py','--execute'),('freeze_outer_case.py','--freeze'),
                                        ('evaluate_outer_case_future.py','--execute')])
def test_case_cli_declares_scenario_and_action_arguments(filename,flag):
    tree=ast.parse((SCRIPTS/filename).read_text())
    flags=[a.value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr=='add_argument' for a in n.args if isinstance(a,ast.Constant)]
    assert '--scenario-id' in flags and flag in flags


def test_future_registry_guard_precedes_prediction_and_output_claim():
    tree=ast.parse((SCRIPTS/'evaluate_outer_case_future.py').read_text())
    calls={}
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):calls.setdefault(n.func.id,[]).append(n.lineno)
    assert max(calls['verify_pre_release_registry'])<min(calls['future_comparisons'])
    assert max(calls['verify_pre_release_registry'])<min(calls['save_json'])


def test_all_new_execution_files_are_pinned_and_exist():
    h=load_helper();assert len(h.SOURCE_FILES)==8
    assert all((SCRIPTS.parent/f).is_file() for f in h.SOURCE_FILES)


def load_batch(tmp_path):
    import argparse
    common=common_stub()
    common.parser=lambda description:argparse.ArgumentParser(description=description)
    # Include the standard --data-root argument in the shared parser.
    def parser(description):
        result=argparse.ArgumentParser(description=description)
        result.add_argument('--data-root',default=None)
        return result
    common.parser=parser;common.resolve_data_root=lambda value:tmp_path
    common.utc_now=lambda:'test-time';common.git_context=lambda:{'git_revision':'test','working_tree_dirty':False}
    common.name=str
    def save(path,document):
        path.parent.mkdir(parents=True,exist_ok=True)
        contents=json.dumps(document,sort_keys=True).encode()
        with path.open('xb') as stream:stream.write(contents)
        return common.sha(contents)
    common.save_json=save
    h=load_helper()
    h.inventory=lambda root:({},['g0','g1'],{},[{'scenario_id':f's{i:02}','group_id':'g0' if i<28 else 'g1'} for i in range(56)])
    h.verify_execution_spec=lambda *a,**kw:({},'execution')
    @contextmanager
    def phase(root,label):yield {'phase':label}
    h.verified_phase=phase
    spec=importlib.util.spec_from_file_location('isolated_outer_batch',SCRIPTS/'run_outer_candidate_batch.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'_common':common,'_outer_pipeline':h}):spec.loader.exec_module(module)
    return module


def test_runner_freezes_all_56_before_opening_first_future(tmp_path,monkeypatch):
    batch=load_batch(tmp_path);completed={k:set() for k in ('candidate','freeze','future')};events=[]
    registry=[]
    for name,phase in [('validated_case','candidate'),('frozen_case','freeze'),('completed_case','future')]:
        monkeypatch.setattr(batch,name,lambda root,row,p=phase:'verified' if row['scenario_id'] in completed[p] else None)
    def invoke(script,argv):
        if script=='freeze_outer_pre_release_registry.py':
            assert len(completed['freeze'])==56 and not completed['future'];registry.append(True);return
        sid=argv[argv.index('--scenario-id')+1]
        phase='candidate' if script.startswith('validate_') else 'freeze' if script.startswith('freeze_') else 'future'
        if phase=='future':assert registry and len(completed['freeze'])==56
        completed[phase].add(sid);events.append((phase,sid))
    batch.invoke=invoke;batch.verify_pre_release_registry=lambda root:({},'registry')
    monkeypatch.setattr(sys,'argv',['runner','--all-groups','--execute'])
    batch.main()
    assert [p for p,_ in events]==['candidate']*56+['freeze']*56+['future']*56
    report=tmp_path/'c4_artifacts/benchmark_runs/adult'/batch.VERSION/'outer_candidate_batches/outer-candidate-all16-v1/batch_results.json'
    assert len(json.loads(report.read_text())['records'])==56


def test_runner_stops_on_freeze_failure_without_opening_any_future(tmp_path,monkeypatch):
    batch=load_batch(tmp_path);frozen=set();events=[]
    batch.validated_case=lambda *a:'already-trained'
    batch.frozen_case=lambda root,row:'frozen' if row['scenario_id'] in frozen else None
    def invoke(script,argv):
        events.append(script)
        sid=argv[argv.index('--scenario-id')+1]
        if sid=='s01':raise ValueError('simulated freeze failure')
        frozen.add(sid)
    batch.invoke=invoke
    monkeypatch.setattr(sys,'argv',['runner','--all-groups','--execute'])
    with pytest.raises(ValueError,match='simulated freeze'):batch.main()
    assert events==['freeze_outer_case.py']*2
    output=tmp_path/'c4_artifacts/benchmark_runs/adult'/batch.VERSION/'outer_candidate_batches/outer-candidate-all16-v1'
    assert (output/'failure.json').is_file() and not (output/'batch_results.json').exists()
    assert json.loads((output/'failure.json').read_text())['phase']=='freeze'


@pytest.mark.parametrize('stage,checker', [('scoped_candidate_validation-v1','validated_case'),
    ('pre_release_freeze-v1','frozen_case'),('future_research-v1','completed_case')])
def test_partial_stage_claim_is_preserved_and_rejected(tmp_path,stage,checker):
    batch=load_batch(tmp_path);directory=batch.runs(tmp_path)/'s' / stage;directory.mkdir(parents=True)
    with pytest.raises(ValueError,match='Incomplete'):getattr(batch,checker)(tmp_path,{'scenario_id':'s'})
    assert directory.exists()


def test_batch_plan_never_invokes_stage_or_opens_outcome(tmp_path,monkeypatch):
    batch=load_batch(tmp_path)
    batch.invoke=lambda *args:pytest.fail('plan must not execute a stage')
    batch.completed_case=lambda *args:pytest.fail('plan must not open an outcome')
    monkeypatch.setattr(sys,'argv',['runner','--all-groups']);batch.main()
    assert not (tmp_path/'c4_artifacts/benchmark_runs').exists()


@pytest.mark.parametrize('path_type', [Path, PureWindowsPath])
def test_missing_model_dependency_is_rejected_with_native_and_windows_paths(path_type):
    h=load_helper()
    snapshot=snapshot_fixture(h, path_type)
    h.Path=path_type
    model_key=str(path_type(snapshot['n1_run_directory'])/'lightgbm.joblib')
    assert model_key in snapshot['dependencies']
    snapshot['dependencies'].pop(model_key)
    with pytest.raises(ValueError, match='pin all original N1 artifacts'):
        h.verify_freeze_dependencies(snapshot)
