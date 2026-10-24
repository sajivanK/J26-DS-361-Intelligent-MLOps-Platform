"""Exact group ranks, omitted-slice semantics, point intervals and fallback integrity."""
import copy
import numpy as np
import pandas as pd
import pytest
from c4.conformal.benchmark_n2 import TARGETS
from c4.conformal.benchmark_n2_extension import fit_extended_calibration
from c4.evaluation.comparison_calibration import (rank_for,fit_comparison_calibration,apply_comparison_calibration,interval_diagnostics)


def tables(groups=37,each=2):
    ids=[f's{i}' for i in range(groups*each)]
    targets=pd.DataFrame(0.,index=ids,columns=list(TARGETS))
    forecasts=pd.DataFrame({t+'__'+label:np.full(len(ids),value) for t in TARGETS
        for label,value in [('lower',-.1),('median',0.),('upper',.1)]},index=ids)
    index=pd.DataFrame({'scenario_id':ids,'group_id':[f'g{i//each}' for i in range(len(ids))],
        'split_role':['conformal_calibration' if i//each<17 else 'conformal_extension' for i in range(len(ids))]})
    return targets,forecasts,index


@pytest.mark.parametrize('groups,bonferroni,rank',[(17,False,18),(19,False,19),(37,False,37),(37,True,38),(58,True,59),(59,True,59)])
def test_exact_rank(groups,bonferroni,rank):assert rank_for(groups,bonferroni)==rank


def test_joint_variant_matches_original_group_max_implementation():
    y,f,i=tables();y.iloc[0,0]=.3;y.iloc[-1,1]=.6
    original,_,groups=fit_extended_calibration(y,f,i)
    new,scores,newgroups=fit_comparison_calibration(y,f,i,'A1')
    assert new['rank']==original.rank and new['nonempty_groups']==original.nonempty_groups
    assert all(v==original.adjustment for v in new['adjustments'].values())
    assert newgroups.joint_score.tolist()==groups.group_score.tolist()
    assert len(scores)==74 and len(newgroups)==37


def test_A4_omits_slice_from_score_without_modifying_original_labels():
    y,f,i=tables();y.iloc[0,1]=.9;before=y.copy()
    calibration,_,_=fit_comparison_calibration(y,f,i,'A4')
    assert calibration['active_targets']==['future_global_margin','future_nonaffected_margin']
    assert set(calibration['adjustments'].values())=={0.}
    result=apply_comparison_calibration(calibration,f)
    assert not any(c.startswith('future_slice_margin') for c in result)
    pd.testing.assert_frame_equal(y,before)


def test_A5_calibrates_absolute_point_residual_not_original_band_error():
    y,f,i=tables();y.iloc[0,0]=.25
    point,_,_=fit_comparison_calibration(y,f,i,'A5')
    joint,_,_=fit_comparison_calibration(y,f,i,'A1')
    assert point['adjustments'][TARGETS[0]]==.25
    assert joint['adjustments'][TARGETS[0]]==pytest.approx(.15)
    result=apply_comparison_calibration(point,f)
    assert result[TARGETS[0]+'__lower'].eq(-.25).all()
    assert result[TARGETS[0]+'__upper'].eq(.25).all()


def test_B5_keeps_full_bounds_without_rank_clipping():
    y,f,i=tables();calibration,_,_=fit_comparison_calibration(y,f,i,'B5')
    assert calibration['rank']==38 and calibration['fallback'] and not calibration['auto_deploy_supported']
    assert all(q is None for q in calibration['adjustments'].values())
    adjusted=apply_comparison_calibration(calibration,f)
    assert all(adjusted[t+'__lower'].eq(-1).all() and adjusted[t+'__upper'].eq(1).all() for t in TARGETS)


def test_B5_59_groups_can_have_distinct_per_target_adjustments():
    y,f,i=tables(59);y.iloc[0,0]=.3;y.iloc[0,1]=.4
    cal,_,_=fit_comparison_calibration(y,f,i,'B5')
    assert cal['rank']==59 and not cal['fallback']
    assert cal['adjustments'][TARGETS[0]]==pytest.approx(.2)
    assert cal['adjustments'][TARGETS[1]]==pytest.approx(.3)
    assert cal['adjustments'][TARGETS[2]]==0


def test_dependent_variants_do_not_inflate_calibration_sample_size():
    y,f,i=tables(19,10);cal,_,groups=fit_comparison_calibration(y,f,i,'A2')
    assert len(y)==190 and cal['nonempty_groups']==len(groups)==19 and cal['rank']==19


@pytest.mark.parametrize('role',['forecast_train','policy_calibration','outer_test'])
def test_other_folds_cannot_calibrate(role):
    y,f,i=tables();i.loc[0,'split_role']=role
    with pytest.raises(ValueError,match='Conformal folds only'):
        fit_comparison_calibration(y,f,i,'A1')


def test_same_group_cannot_span_two_calibration_roles():
    y,f,i=tables();i.loc[0,'split_role']='conformal_extension'
    with pytest.raises(ValueError,match='Group identity/fold'):
        fit_comparison_calibration(y,f,i,'A1')


def test_scenario_order_must_match():
    y,f,i=tables()
    with pytest.raises(ValueError,match='order must align'):
        fit_comparison_calibration(y,f.iloc[::-1],i,'A1')


@pytest.mark.parametrize('field,value',[('rank',37),('fallback',False),('auto_deploy_supported',True)])
def test_cannot_bypass_B5_insufficient_support(field,value):
    y,f,i=tables();cal,_,_=fit_comparison_calibration(y,f,i,'B5');cal[field]=value
    with pytest.raises(ValueError):apply_comparison_calibration(cal,f)


def test_two_target_mode_cannot_be_assigned_to_other_variants():
    y,f,i=tables();cal,_,_=fit_comparison_calibration(y,f,i,'A4');cal['comparison']='A1'
    with pytest.raises(ValueError,match='mode or active targets'):
        apply_comparison_calibration(cal,f)


def test_joint_adjustment_cannot_vary_between_targets():
    y,f,i=tables();cal,_,_=fit_comparison_calibration(y,f,i,'A1');cal['adjustments'][TARGETS[0]]=.1
    with pytest.raises(ValueError,match='common across targets'):
        apply_comparison_calibration(cal,f)


def test_diagnostics_are_explicitly_in_sample():
    y,f,i=tables();cal,_,_=fit_comparison_calibration(y,f,i,'B5')
    d=interval_diagnostics(y,apply_comparison_calibration(cal,f),i,TARGETS)
    assert d['evaluation']=='in_sample_calibration_diagnostic_not_independent_validation'
    assert d['group_all_variant_simultaneous_coverage']==1.
    assert all(v['mean_width_fraction']==2. for v in d['targets'].values())
