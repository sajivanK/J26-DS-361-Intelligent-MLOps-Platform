"""Pool groups, never individual variants; preserve original N2 behavior."""
import json
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import pytest
from c4.conformal.benchmark_n2 import TARGETS, JointCalibration, apply_joint_calibration, fit_joint_calibration, calibration_diagnostics
from c4.conformal.benchmark_n2_extension import fit_extended_calibration


def tables(groups=37):
    ids=[f's{i}' for i in range(groups*2)]
    y=pd.DataFrame(0.,index=ids,columns=list(TARGETS))
    f=pd.DataFrame(index=ids)
    for t in TARGETS:
        for label,value in [('lower',-.1),('median',0.),('upper',.1)]:f[t+'__'+label]=value
    i=pd.DataFrame({'scenario_id':ids,'group_id':[f'g{n//2:02d}' for n in range(len(ids))],
        'split_role':['conformal_calibration' if n//2<17 else 'conformal_extension' for n in range(len(ids))]})
    return y,f,i


def test_pooling_has_thirty_seven_group_scores_and_maximum_rank():
    y,f,i=tables();y.loc['s0',TARGETS[0]]=.2;y.loc['s73',TARGETS[1]]=.6
    c,s,g=fit_extended_calibration(y,f,i)
    assert len(s)==74 and len(g)==37 and c.rank==37 and not c.fallback
    assert c.adjustment==pytest.approx(.5)
    assert c.adjustment==g.group_score.max()


def test_group_variants_are_not_independent_calibration_units():
    y,f,i=tables();y.loc['s0',TARGETS[0]]=.6;y.loc['s1',TARGETS[1]]=.8
    c,s,g=fit_extended_calibration(y,f,i)
    assert g.loc['g00','scenario_count']==2
    assert g.loc['g00','group_score']==pytest.approx(.7)
    assert c.nonempty_groups==37


@pytest.mark.parametrize('role',['forecast_train','policy_calibration','outer_test'])
def test_disallowed_folds_rejected(role):
    y,f,i=tables();i.loc[0,'split_role']=role
    with pytest.raises(ValueError,match='Only original and extension'):fit_extended_calibration(y,f,i)


def test_group_cannot_cross_original_and_extension_roles():
    y,f,i=tables();i.loc[0,'split_role']='conformal_extension'
    with pytest.raises(ValueError,match='multiple folds'):fit_extended_calibration(y,f,i)


def test_original_n2_still_rejects_extension_rows():
    y,f,i=tables()
    with pytest.raises(ValueError,match='Only conformal_calibration'):fit_joint_calibration(y,f,i)


def test_small_support_still_falls_back():
    y,f,i=tables(17);c,_,_=fit_extended_calibration(y,f,i)
    assert c.fallback and c.adjustment is None
    bounds=apply_joint_calibration(c,f)
    assert bounds[TARGETS[0]+'__lower'].eq(-1).all()


def test_common_adjustment_and_json_roundtrip():
    y,f,i=tables();y.loc['s0',TARGETS[0]]=.4
    c,_,_=fit_extended_calibration(y,f,i)
    restored=JointCalibration.from_dict(json.loads(json.dumps(c.to_dict())))
    a=apply_joint_calibration(c,f);b=apply_joint_calibration(restored,f)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    for t in TARGETS:assert np.allclose(a[t+'__upper'],.4)


def test_misaligned_targets_rejected():
    y,f,i=tables()
    with pytest.raises(ValueError):fit_extended_calibration(y.iloc[::-1],f,i)


def test_nonfinite_targets_rejected():
    y,f,i=tables();y.iloc[0,0]=np.nan
    with pytest.raises(ValueError):fit_extended_calibration(y,f,i)


def test_calibration_diagnostics_never_claim_independent_coverage():
    y,f,i=tables();c,_,_=fit_extended_calibration(y,f,i)
    d=calibration_diagnostics(y,apply_joint_calibration(c,f),i)
    assert d['independent_coverage_validation'] is False


def test_entry_points_do_not_train_or_predict_models():
    scripts=Path(__file__).resolve().parents[2]/'scripts'
    for filename in ('audit_extended_calibration.py','calibrate_n2_extension.py'):
        tree=ast.parse((scripts/filename).read_text())
        attrs={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        assert not attrs&{'fit','predict','predict_proba','sample','load_model'}


def combiner():
    path=Path(__file__).resolve().parents[2]/'scripts/audit_extended_calibration.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='combine_tables')
    def require(condition,message):
        if not condition:raise ValueError(message)
    namespace={'np':np,'pd':pd,'TARGET_COLUMNS':TARGETS,'require':require}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['combine_tables']


def test_combined_snapshot_preserves_both_original_roles():
    features=[pd.DataFrame(0.,index=[sid],columns=[f'x{j}' for j in range(75)]) for sid in ('a','b')]
    targets=[pd.DataFrame(0.,index=f.index,columns=TARGETS) for f in features]
    indices=[pd.DataFrame({'scenario_id':[sid],'group_id':[sid], 'split_role':[role]})
             for sid,role in [('a','conformal_calibration'),('b','conformal_extension')]]
    f,t,i=combiner()(features,targets,indices)
    assert f.index.tolist()==t.index.tolist()==['a','b']
    assert i.split_role.tolist()==['conformal_calibration','conformal_extension']


def test_combined_snapshot_rejects_policy_rows():
    f=pd.DataFrame(0.,index=['a'],columns=[f'x{j}' for j in range(75)])
    t=pd.DataFrame(0.,index=['a'],columns=TARGETS)
    i=pd.DataFrame({'scenario_id':['a'],'group_id':['g'],'split_role':['policy_calibration']})
    with pytest.raises(ValueError):combiner()([f],[t],[i])
