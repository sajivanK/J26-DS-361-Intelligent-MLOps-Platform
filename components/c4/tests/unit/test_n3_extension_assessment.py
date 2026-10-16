"""Apply the same boundaries to finite and fallback intervals without targets."""
import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from c4.risk.benchmark_n3 import TARGETS, BOUNDARIES


def helper():
    common=ModuleType('_common')
    def require(condition,message):
        if not condition:raise ValueError(message)
    common.require=require;common.C4_ROOT=Path('/unused')
    for key in ('checked_bytes','checked_json','git_context','parser','resolve_data_root','save_json','sha','utc_now'):
        setattr(common,key,lambda *a,**k:None)
    group=ModuleType('_conformal_group');group.VERSION='adult-simulation-pp1-v2';group.verify_n1=lambda *a:None
    path=Path(__file__).resolve().parents[2]/'scripts/assess_n3_extension.py'
    spec=importlib.util.spec_from_file_location('n3_extension_under_test',path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'_common':common,'_conformal_group':group}):spec.loader.exec_module(module)
    return module


def frame():
    f=pd.DataFrame(index=pd.Index(['s0'],name='scenario_id'))
    for t,b in BOUNDARIES.items():
        f[t+'__lower']=b;f[t+'__median']=b+.01;f[t+'__upper']=b+.02
    return f


def test_all_safe_is_signal_not_authorization():
    records,counts,d=helper().assess_interval_table(frame(),supported=True)
    assert counts['AUTO_DEPLOY']==1 and records[0]['release_authorized'] is False
    assert all(v['SAFE']==1 for v in d.values())


def test_uncertain_and_unsafe_aggregate_separately():
    f=pd.concat([frame().rename(index={'s0':x}) for x in ('safe','uncertain','unsafe')])
    f.loc['uncertain',TARGETS[0]+'__lower']=-.01
    f.loc['unsafe',[TARGETS[1]+'__lower',TARGETS[1]+'__median',TARGETS[1]+'__upper']]=[-.03,-.02,-.01]
    _,counts,_=helper().assess_interval_table(f,supported=True)
    assert counts=={'scenarios':3,'AUTO_DEPLOY':1,'HUMAN_REVIEW':1,'ESCALATE':1}


def test_fallback_remains_review():
    f=frame()
    for t in TARGETS:f[t+'__lower']=-1.;f[t+'__median']=0.;f[t+'__upper']=1.
    _,counts,_=helper().assess_interval_table(f,supported=False)
    assert counts['HUMAN_REVIEW']==1


@pytest.mark.parametrize('value',[np.nan,np.inf,-1.1])
def test_invalid_interval_values_fail(value):
    f=frame();f.iloc[0,0]=value
    with pytest.raises(ValueError):helper().assess_interval_table(f,supported=True)


def test_median_outside_bounds_fails():
    f=frame();f[TARGETS[0]+'__median']=.9
    with pytest.raises(ValueError):helper().assess_interval_table(f,supported=True)


def test_duplicate_ids_fail():
    f=pd.concat([frame(),frame()])
    with pytest.raises(ValueError):helper().assess_interval_table(f,supported=True)


def test_future_label_columns_fail():
    f=frame();f['observed_future_margin']=0.
    with pytest.raises(ValueError):helper().assess_interval_table(f,supported=True)


def test_empty_table_fails():
    with pytest.raises(ValueError):helper().assess_interval_table(frame().iloc[:0],supported=True)


def test_support_is_explicit_boolean():
    with pytest.raises(ValueError):helper().assess_interval_table(frame(),supported='True')


def test_script_does_not_open_targets_or_fit_models():
    path=Path(__file__).resolve().parents[2]/'scripts/assess_n3_extension.py'
    tree=ast.parse(path.read_text())
    strings={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    assert not strings&{'targets.csv','outcomes.json','future_predictions.csv'}
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&{'fit','predict','predict_proba','sample'}
