"""Numerical and leakage-boundary checks for frozen joint group conformal logic."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from c4.conformal.benchmark_n2 import (TARGETS, JointCalibration, apply_joint_calibration,
    calibration_diagnostics, fit_joint_calibration)


def tables(groups=19):
    ids=[f's{i}' for i in range(groups*2)]
    y=pd.DataFrame(0.,index=ids,columns=list(TARGETS))
    f=pd.DataFrame(index=ids)
    for target in TARGETS:
        for label,value in [('lower',-.1),('median',0.),('upper',.1)]:
            f[target+'__'+label]=value
    index=pd.DataFrame({'scenario_id':ids,'group_id':[f'g{i//2:02d}' for i in range(len(ids))],
                        'split_role':'conformal_calibration'})
    return y,f,index


def test_joint_score_then_group_max_are_not_averaged():
    y,f,i=tables()
    y.loc['s0',TARGETS[0]]=.3
    y.loc['s1',TARGETS[1]]=.6
    y.loc['s1',TARGETS[2]]=.5
    cal,scores,groups=fit_joint_calibration(y,f,i)
    assert scores.loc['s0','joint_score']==pytest.approx(.2)
    assert scores.loc['s1','joint_score']==pytest.approx(.5)
    assert groups.loc['g00','group_score']==pytest.approx(.5)
    assert groups.loc['g00','scenario_count']==2
    assert cal.rank==19 and cal.adjustment==pytest.approx(.5)


def test_nonconformity_handles_lower_tail_and_inside_interval():
    y,f,i=tables()
    y.loc['s0',TARGETS[0]]=-.4
    _,scores,_=fit_joint_calibration(y,f,i)
    assert scores.loc['s0',TARGETS[0]+'__nonconformity']==pytest.approx(.3)
    assert scores.loc['s1','joint_score']==0.


@pytest.mark.parametrize('count,rank,fallback',[(17,18,True),(18,19,True),(19,19,False),(20,20,False)])
def test_finite_sample_boundary_uses_nonempty_groups(count,rank,fallback):
    y,f,i=tables(count)
    cal,_,grouped=fit_joint_calibration(y,f,i)
    assert len(grouped)==count and cal.rank==rank and cal.fallback is fallback
    assert cal.adjustment is None if fallback else cal.adjustment==0.


def test_small_sample_fallback_is_uninformative_even_with_zero_scores():
    y,f,i=tables(17)
    cal,_,_=fit_joint_calibration(y,f,i)
    intervals=apply_joint_calibration(cal,f)
    assert cal.auto_deploy_supported is False
    for target in TARGETS:
        assert intervals[target+'__lower'].eq(-1.).all()
        assert intervals[target+'__upper'].eq(1.).all()
    diagnostics=calibration_diagnostics(y,intervals,i)
    assert diagnostics['scenario_joint_coverage']==1.
    assert diagnostics['independent_coverage_validation'] is False
    assert all(r['mean_width_fraction']==2. for r in diagnostics['targets'].values())


def test_one_common_adjustment_expands_all_dimensions():
    y,f,i=tables()
    y.loc['s0',TARGETS[0]]=.4
    cal,_,_=fit_joint_calibration(y,f,i)
    intervals=apply_joint_calibration(cal,f)
    for target in TARGETS:
        assert np.allclose(intervals[target+'__lower'],-.4)
        assert np.allclose(intervals[target+'__upper'],.4)


@pytest.mark.parametrize('role',['forecast_train','policy_calibration','outer_test'])
def test_nonconformal_rows_cannot_calibrate(role):
    y,f,i=tables()
    i.loc[0,'split_role']=role
    with pytest.raises(ValueError,match='Only conformal_calibration'):
        fit_joint_calibration(y,f,i)


def test_target_alignment_cannot_be_silently_reordered():
    y,f,i=tables()
    with pytest.raises(ValueError,match='identical scenario order'):
        fit_joint_calibration(y.iloc[::-1],f,i)


def test_unordered_quantiles_are_rejected():
    y,f,i=tables()
    f.loc['s0',TARGETS[0]+'__lower']=.2
    with pytest.raises(ValueError,match='ordered'):
        fit_joint_calibration(y,f,i)


def test_nonfinite_targets_are_rejected():
    y,f,i=tables()
    y.iloc[0,0]=np.nan
    with pytest.raises(ValueError,match='finite accuracy-margin'):
        fit_joint_calibration(y,f,i)


def test_json_reload_preserves_label_free_inference():
    y,f,i=tables(17)
    cal,_,_=fit_joint_calibration(y,f,i)
    restored=JointCalibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    pd.testing.assert_frame_equal(apply_joint_calibration(cal,f),apply_joint_calibration(restored,f),check_exact=True)


def test_saved_rank_cannot_be_changed_to_avoid_fallback():
    y,f,i=tables(17)
    cal,_,_=fit_joint_calibration(y,f,i)
    with pytest.raises(ValueError,match='rank differs'):
        apply_joint_calibration(replace(cal,rank=17,fallback=False,adjustment=0.,informative_intervals=True,auto_deploy_supported=True),f)
