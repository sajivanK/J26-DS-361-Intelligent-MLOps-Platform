"""Policy-fold admission, freeze and finite-sample support boundaries."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


def helper():
    common = ModuleType('_common')
    def require(condition, message):
        if not condition: raise ValueError(message)
    common.require = require
    common.checked_json = lambda *a: ({}, '')
    common.checked_bytes = lambda *a: b''
    group = ModuleType('_policy_group')
    group.VERSION = 'adult-simulation-pp1-v2'
    group.N1_SPEC_HASH = 'spec'
    group.N1_RUN_HASH = 'run'
    group.verify_n1 = lambda root: {}
    def cases(manifest, gid):
        rows = manifest['scenarios']
        require(len(rows)==37 and all(r['group_id']==gid and r['split_role']=='policy_calibration' for r in rows),
                'Wrong policy group')
        return rows
    group.policy_group_cases = cases
    path = Path(__file__).resolve().parents[2] / 'scripts/_policy_pipeline.py'
    spec = importlib.util.spec_from_file_location('isolated_policy_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'_common':common, '_policy_group':group}):
        spec.loader.exec_module(module)
    return module


def report_fixture():
    rows=[]
    for i in range(37):
        rows.append({'scenario_id': f's{i}', 'group_id':'cal', 'split_role':'policy_calibration',
            'mechanism':'interaction', 'magnitude':.2, 'diagnosis_condition':'exact_slice',
            'repair_condition':'targeted_repair', 'is_no_drift_control':False})
    records=[{**r, 'simulation_admission':'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if i<2 else 'BLOCKED_ESCALATE',
              'screen_status':'PASS' if i<2 else 'FAIL'} for i,r in enumerate(rows)]
    return {'scenarios':rows}, {'group_id':'cal','cases':37,'records':records,
        'scoped_admissible_cases':2,'blocked_cases':35,
        'n1_binding':{'specification_sha256':'spec','run_metadata_sha256':'run'}}


def test_blocked_scenarios_never_receive_candidate_or_target_rows():
    manifest,report=report_fixture()
    selected=helper().select_admitted_cases(manifest,report,'cal')
    assert [r['scenario_id'] for r in selected]==['s0','s1']


def test_duplicate_admission_record_is_rejected():
    manifest,report=report_fixture()
    report['records'][1]=copy.deepcopy(report['records'][0])
    with pytest.raises(ValueError,match='Duplicate or missing'):
        helper().select_admitted_cases(manifest,report,'cal')


def test_admitted_status_cannot_bypass_a_failed_screen():
    manifest,report=report_fixture()
    report['records'][0]['screen_status']='FAIL'
    with pytest.raises(ValueError,match='passing screen'):
        helper().select_admitted_cases(manifest,report,'cal')


def test_other_n1_model_version_is_rejected():
    manifest,report=report_fixture()
    report['n1_binding']['run_metadata_sha256']='different'
    with pytest.raises(ValueError,match='different N1'):
        helper().select_admitted_cases(manifest,report,'cal')


def test_all_blocked_group_has_no_imputed_targets():
    manifest,report=report_fixture()
    for r in report['records']:
        r.update(simulation_admission='BLOCKED_ESCALATE',screen_status='FAIL')
    report.update(scoped_admissible_cases=0,blocked_cases=37)
    assert helper().select_admitted_cases(manifest,report,'cal')==[]


@pytest.mark.parametrize('field,value', [('complete_decision_evidence_frozen',False),('future_evaluation_executed',True)])
def test_future_evaluation_requires_original_complete_prefuture_freeze(field,value):
    snapshot={'complete_decision_evidence_frozen':True,'future_evaluation_executed':False}
    snapshot[field]=value
    with pytest.raises(ValueError,match='original pre-future freeze'):
        helper().verify_freeze_dependencies(snapshot)


def test_training_freeze_is_not_a_policy_freeze():
    snapshot={'complete_decision_evidence_frozen':True,'future_evaluation_executed':False,
              'scope':'one_admitted_forecast_training_scenario_under_simulated_upstream_evidence'}
    with pytest.raises(ValueError,match='Wrong policy freeze'):
        helper().verify_freeze_dependencies(snapshot)


def test_policy_freeze_must_pin_model_artifacts_before_future():
    h=helper()
    model_dir=Path('/models/n1-fit-v1')
    metadata={'files':{'lightgbm.joblib':'modelhash','linear.joblib':'linearhash'}}
    h.checked_bytes=lambda *args:json.dumps(metadata).encode()
    snapshot={'complete_decision_evidence_frozen':True,'future_evaluation_executed':False,
        'scope':'one_admitted_policy_scenario_under_simulated_upstream_evidence',
        'scenario':{'split_role':'policy_calibration'},'n1_binding':{'specification_sha256':'spec','run_metadata_sha256':'run'},
        'policy_specification_sha256':h.POLICY_SPEC_HASH,'policy_forecasts_path':'/policy.json','policy_forecasts_sha256':'policy-forecast',
        'n1_run_directory':str(model_dir),'dependencies':{'/policy.json':'policy-forecast','/policy/freeze.json':h.POLICY_SPEC_HASH,str(model_dir/'metadata.json'):'run',
            str(model_dir.parent/'specification.json'):'spec'}}
    with pytest.raises(ValueError,match='pin all original N1 artifacts'):
        h.verify_freeze_dependencies(snapshot)


def test_context_support_counts_groups_not_scenarios():
    rows=[{'scenario_id':f's{i}','group_id':'g1','context':'LOW','split_role':'policy_calibration'} for i in range(10)]
    support=helper().context_support_from_rows(rows)
    assert support['distinct_admitted_groups_by_context']['LOW']==1
    assert 'LOW' in support['unsupported_contexts']


def test_context_support_three_group_boundary():
    rows=[{'scenario_id':f's{i}','group_id':f'g{i}','context':'MEDIUM','split_role':'policy_calibration'} for i in range(3)]
    support=helper().context_support_from_rows(rows)
    assert 'MEDIUM' not in support['unsupported_contexts']
    assert support['unsupported_contexts']==['LOW','HIGH']


def test_context_support_rejects_other_folds():
    rows=[{'scenario_id':'s','group_id':'g','context':'LOW','split_role':'outer_test'}]
    with pytest.raises(ValueError):helper().context_support_from_rows(rows)


def test_context_support_rejects_duplicate_rows():
    row={'scenario_id':'s','group_id':'g','context':'LOW','split_role':'policy_calibration'}
    with pytest.raises(ValueError):helper().context_support_from_rows([row,row])


def test_policy_context_freeze_is_required_before_future_execution():
    import ast
    path=Path(__file__).resolve().parents[2]/'scripts/evaluate_policy_case_future.py'
    tree=ast.parse(path.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    support_calls=[n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                   and n.func.id=='verify_context_support']
    prediction_calls=[n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                      and n.func.id=='future_comparisons']
    assert support_calls and prediction_calls and max(support_calls)<min(prediction_calls)


def test_whole_fold_prerelease_phase_precedes_future_phase():
    import ast
    path=Path(__file__).resolve().parents[2]/'scripts/run_policy_candidate_batch.py'
    tree=ast.parse(path.read_text())
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.Try))
    phase1=next(n for n in block.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='selected')
    context=next(n for n in block.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                 and isinstance(n.value.func,ast.Name) and n.value.func.id=='stage')
    phase2=next(n for n in block.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='groups')
    assert ast.literal_eval(context.value.args[2])=='freeze_policy_context_support.py'
    assert phase1.lineno<context.lineno<phase2.lineno
