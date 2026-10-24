"""Extension-fold admission, freeze and finite-sample support boundaries."""
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
    group = ModuleType('_extension_group')
    group.VERSION = 'adult-simulation-pp1-v2'
    group.N1_SPEC_HASH = 'spec'
    group.N1_RUN_HASH = 'run'
    group.EXTENSION_HASH = 'extension'
    group.MANIFEST_HASH = 'manifest'
    group.load_extension = lambda *args: None
    group.verify_n1 = lambda root: {}
    def cases(manifest, gid):
        rows = manifest['scenarios']
        require(len(rows)==37 and all(r['group_id']==gid and r['split_role']=='conformal_extension' for r in rows),
                'Wrong extension group')
        return rows
    group.extension_group_cases = cases
    path = Path(__file__).resolve().parents[2] / 'scripts/_extension_pipeline.py'
    spec = importlib.util.spec_from_file_location('isolated_extension_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    original = ModuleType('_conformal_group')
    original.N1_SPEC_HASH = 'spec'; original.N1_RUN_HASH = 'run'
    with patch.dict(sys.modules, {'_common':common, '_extension_group':group, '_conformal_group':original}):
        spec.loader.exec_module(module)
    return module


def report_fixture():
    rows=[]
    for i in range(37):
        rows.append({'scenario_id': f's{i}', 'group_id':'cal', 'split_role':'conformal_extension',
            'mechanism':'interaction', 'magnitude':.2, 'diagnosis_condition':'exact_slice',
            'repair_condition':'targeted_repair', 'is_no_drift_control':False})
    records=[{**r, 'simulation_admission':'READY_FOR_SCOPED_CANDIDATE_VALIDATION' if i<2 else 'BLOCKED_ESCALATE',
              'screen_status':'PASS' if i<2 else 'FAIL'} for i,r in enumerate(rows)]
    return {'scenarios':rows}, {'group_id':'cal','cases':37,'records':records,
        'scoped_admissible_cases':2,'blocked_cases':35,'extension_freeze_sha256':'extension',
        'n1_binding':{'specification_sha256':'spec','run_metadata_sha256':'run'}}


@pytest.mark.parametrize('groups,rank,available', [(0,1,False),(17,18,False),(18,19,False),(19,19,True),(20,20,True)])
def test_exact_frozen_95_percent_rank_boundary(groups,rank,available):
    support=helper().extension_support(groups)
    assert support['quantile_rank']==rank
    assert support['finite_adjustment_available'] is available
    assert support['insufficient_support_fallback']=='bounds_minus1_plus1_and_disable_AUTO_DEPLOY'


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


def test_training_freeze_is_not_a_extension_freeze():
    snapshot={'complete_decision_evidence_frozen':True,'future_evaluation_executed':False,
              'scope':'one_admitted_forecast_training_scenario_under_simulated_upstream_evidence'}
    with pytest.raises(ValueError,match='Wrong extension freeze'):
        helper().verify_freeze_dependencies(snapshot)


def test_extension_freeze_must_pin_model_artifacts_before_future():
    h=helper()
    model_dir=Path('/models/n1-fit-v1')
    metadata={'files':{'lightgbm.joblib':'modelhash','linear.joblib':'linearhash'}}
    h.checked_bytes=lambda *args:json.dumps(metadata).encode()
    snapshot={'complete_decision_evidence_frozen':True,'future_evaluation_executed':False,
        'scope':'one_admitted_extension_scenario_under_simulated_upstream_evidence',
        'scenario':{'split_role':'conformal_extension'},'n1_binding':{'specification_sha256':'spec','run_metadata_sha256':'run'},
        'extension_freeze_sha256':'extension',
        'n1_run_directory':str(model_dir),'dependencies':{'/extension/freeze.json':'extension',str(model_dir/'metadata.json'):'run',
            str(model_dir.parent/'specification.json'):'spec'}}
    with pytest.raises(ValueError,match='pin all original N1 artifacts'):
        h.verify_freeze_dependencies(snapshot)


def test_original_admission_cannot_be_used_as_extension_admission():
    manifest,report=report_fixture()
    report['extension_freeze_sha256']='different'
    with pytest.raises(ValueError,match='Extension admission report'):
        helper().select_admitted_cases(manifest,report,'cal')


def test_new_freeze_cannot_omit_extension_binding():
    snapshot={'complete_decision_evidence_frozen':True,'future_evaluation_executed':False,
        'scope':'one_admitted_extension_scenario_under_simulated_upstream_evidence',
        'scenario':{'split_role':'conformal_extension'},
        'n1_binding':{'specification_sha256':'spec','run_metadata_sha256':'run'}}
    with pytest.raises(ValueError,match='Wrong extension evidence freeze'):
        helper().verify_freeze_dependencies(snapshot)


def test_stage_order_freezes_before_future_execution():
    import ast
    root=Path(__file__).resolve().parents[2]
    tree=ast.parse((root/'scripts/run_extension_pipeline.py').read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=='steps' for t in n.targets))
    steps=ast.literal_eval(assignment.value)
    assert [step[0] for step in steps]==['validate_extension_candidate.py','freeze_extension_case.py','evaluate_extension_case_future.py']


def test_candidate_batch_never_refits_generator_or_n1():
    import ast
    root=Path(__file__).resolve().parents[2]
    tree=ast.parse((root/'scripts/run_extension_candidate_batch.py').read_text())
    scripts=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.endswith('.py')]
    # Admission script paths appear only as dependency hashes, not as stage invocations.
    invoked=[ast.literal_eval(n.args[2]) for n in ast.walk(tree) if isinstance(n,ast.Call) and
             isinstance(n.func,ast.Name) and n.func.id=='stage']
    assert invoked==['run_extension_pipeline.py','build_extension_group_meta_dataset.py','build_extension_group_meta_dataset.py']
    assert 'run_n1_benchmark.py' not in scripts
