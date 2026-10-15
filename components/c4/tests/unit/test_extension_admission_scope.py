"""Only predefined extension groups may execute; frozen original folds stay excluded."""
import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parents[2]


def helper():
    common=ModuleType('_common');common.C4_ROOT=Path('/unused')
    def require(condition,message):
        if not condition:raise ValueError(message)
    common.require=require;common.checked_bytes=lambda *a:b'';common.checked_json=lambda *a:({},'wrong')
    conformal=ModuleType('_conformal_group');conformal.verify_n1=lambda *a:{'protocol_freeze_sha256':'expected'}
    training=ModuleType('_training_group');training.corrupt_labels=lambda *a:None
    builder=ModuleType('_calibration_extension_manifest');builder.build_extension=lambda *a:{}
    spec=importlib.util.spec_from_file_location('extension_scope_under_test',ROOT/'scripts/_extension_group.py')
    h=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'_common':common,'_conformal_group':conformal,'_training_group':training,'_calibration_extension_manifest':builder}):
        spec.loader.exec_module(h)
    return h


def manifest():
    groups=[{'group_id':f'adult-calext-v2-ns{ns}-r{i:03d}','split_role':'conformal_extension'}
            for ns in (42,123,2026,3407) for i in (20,21,22,23,24)]
    rows=[]
    for g in groups:
        gid=g['group_id']
        for m in ('marginal_multi','interaction','combined'):
            for s in (.2,.5,.8):
                for d in ('exact_slice','broadened_slice'):
                    for q in ('targeted_repair','label_corrupted_repair'):
                        base=f'{gid}:{m}:{s}:{d}'
                        rows.append(dict(group_id=gid,scenario_id=base+':'+q,split_role='conformal_extension',
                            mechanism=m,magnitude=s,diagnosis_condition=d,repair_condition=q,
                            shared_base_package_id=base,is_no_drift_control=False))
        rows.append(dict(group_id=gid,scenario_id=gid+'-control',split_role='conformal_extension',
                         mechanism='none',magnitude=0,diagnosis_condition='exact_slice',repair_condition='targeted_repair',
                         shared_base_package_id=gid+':control',is_no_drift_control=True))
    return {'groups':groups,'scenarios':rows}


def test_five_fixed_batches_cover_twenty_groups_once():
    h,m=helper(),manifest()
    flat=[gid for b in range(1,6) for gid in h.batch_groups(m,b)]
    assert len(flat)==len(set(flat))==20
    assert flat==[g['group_id'] for g in m['groups']]


@pytest.mark.parametrize('gid',['adult-ns42-r003','adult-ns123-r003','adult-ns42-r001','adult-ns42-r002'])
def test_original_groups_rejected(gid):
    with pytest.raises(ValueError):helper().extension_group_cases(manifest(),gid)


def test_cross_fold_scenario_rejected():
    h,m=helper(),manifest();m['scenarios'][0]['split_role']='outer_test'
    with pytest.raises(ValueError):h.extension_group_cases(m,m['groups'][0]['group_id'])


def test_reordered_groups_rejected():
    h,m=helper(),manifest();m['groups'].reverse()
    with pytest.raises(ValueError):h.extension_groups(m)


def test_base_precedes_corrupted_pair():
    h,m=helper(),manifest();seen=set()
    for r in h.extension_group_cases(m,m['groups'][0]['group_id']):
        if r['repair_condition']=='targeted_repair':seen.add(r['shared_base_package_id'])
        else:assert r['shared_base_package_id'] in seen


def test_factorial_mutation_rejected():
    h,m=helper(),manifest();m['scenarios'][0]['magnitude']=.9
    with pytest.raises(ValueError):h.extension_group_cases(m,m['groups'][0]['group_id'])


@pytest.mark.parametrize('number',[0,6,True])
def test_invalid_batch_rejected(number):
    with pytest.raises(ValueError):helper().batch_groups(manifest(),number)


def test_wrong_checkpoint_stops_before_execution():
    with pytest.raises(ValueError):helper().load_extension(Path('/unused'))


def test_scripts_never_invoke_candidate_future_or_calibration_runners():
    names=[]
    for p in (ROOT/'scripts').glob('run_extension*.py'):
        tree=ast.parse(p.read_text())
        names.extend(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.endswith('.py'))
    assert not any('candidate' in n or 'future' in n or 'calibrate' in n or 'pipeline' in n for n in names)
