"""Frozen comparison intervals: joint CQR, two-target joint, point-residual and Bonferroni."""
from fractions import Fraction
from numbers import Real
import numpy as np
import pandas as pd
from c4.conformal.benchmark_n2 import TARGETS, validate_forecasts

CALIBRATION_IDS = ('A1','A2','A3','A4','A5','B5')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank_for(groups, bonferroni=False):
    require(type(groups) is int and groups >= 0, 'Invalid group count.')
    level = 1-Fraction(1,60 if bonferroni else 20)
    value = (groups+1)*level
    return (value.numerator+value.denominator-1)//value.denominator


def fit_comparison_calibration(targets, forecasts, index, comparison):
    require(comparison in CALIBRATION_IDS, 'Unknown interval comparison.')
    validate_forecasts(forecasts)
    require(targets.columns.tolist()==list(TARGETS) and targets.index.equals(forecasts.index)
            and index.scenario_id.tolist()==targets.index.tolist() and index.scenario_id.is_unique,
            'Targets, forecasts and scenario order must align.')
    require(index.split_role.isin(['conformal_calibration','conformal_extension']).all(), 'Conformal folds only.')
    require(index.group_id.map(lambda g:isinstance(g,str) and bool(g)).all()
            and index.groupby('group_id').split_role.nunique().eq(1).all(), 'Group identity/fold differs.')
    y=targets.to_numpy(dtype=float)
    require(np.isfinite(y).all() and (np.abs(y)<=1).all(), 'Invalid accuracy-margin targets.')
    active=tuple(t for t in TARGETS if comparison!='A4' or t!='future_slice_margin')
    mode='bonferroni' if comparison=='B5' else 'point_joint' if comparison=='A5' else 'joint'
    scores=pd.DataFrame(index=targets.index)
    for target in active:
        actual=targets[target].to_numpy(dtype=float)
        if mode=='point_joint':
            score=np.abs(actual-forecasts[target+'__median'].to_numpy(dtype=float))
        else:
            score=np.maximum.reduce([np.zeros(len(actual)),forecasts[target+'__lower'].to_numpy(dtype=float)-actual,
                                     actual-forecasts[target+'__upper'].to_numpy(dtype=float)])
        scores[target+'__nonconformity']=score
    scores['group_id']=index.group_id.to_numpy()
    group_scores=scores.groupby('group_id',sort=True).max()
    if mode!='bonferroni':
        scores['joint_score']=scores[[t+'__nonconformity' for t in active]].max(axis=1)
        group_scores['joint_score']=scores.groupby('group_id').joint_score.max()
    n=len(group_scores);rank=rank_for(n,mode=='bonferroni');fallback=rank>n
    adjustments={}
    for t in active:
        column=t+'__nonconformity' if mode=='bonferroni' else 'joint_score'
        adjustments[t]=None if fallback else float(np.sort(group_scores[column].to_numpy())[rank-1])
    group_scores['scenario_count']=scores.groupby('group_id').size()
    calibration={'comparison':comparison,'mode':mode,'active_targets':list(active),'joint_alpha':.05,
        'nonempty_groups':n,'rank':rank,'fallback':fallback,'auto_deploy_supported':not fallback,
        'adjustments':adjustments,'unit':'one max score per nonempty replication group',
        'production_release_authorized':False}
    return calibration,scores,group_scores


def apply_comparison_calibration(calibration,forecasts):
    validate_forecasts(forecasts)
    mode=calibration['mode'];active=tuple(calibration['active_targets']);n=calibration['nonempty_groups']
    comparison=calibration['comparison']
    require(comparison in (*CALIBRATION_IDS,'ORIGINAL'),'Unknown interval comparison.')
    expected_mode='bonferroni' if comparison=='B5' else 'point_joint' if comparison=='A5' else 'joint'
    expected_active=tuple(t for t in TARGETS if comparison!='A4' or t!='future_slice_margin')
    require(mode==expected_mode and active==expected_active,'Comparison mode or active targets differ.')
    require(mode in {'joint','point_joint','bonferroni'} and active in {TARGETS,('future_global_margin','future_nonaffected_margin')},
            'Unknown interval definition.')
    require(mode!='bonferroni' or active==TARGETS, 'Bonferroni comparison requires three targets.')
    require(calibration['joint_alpha']==.05 and calibration['rank']==rank_for(n,mode=='bonferroni'), 'Frozen rank differs.')
    fallback=calibration['rank']>n
    require(type(calibration['fallback']) is bool and calibration['fallback']==fallback and
            calibration['auto_deploy_supported'] is (not fallback), 'Calibration support flags differ.')
    require(set(calibration['adjustments'])==set(active),'Adjustment targets differ.')
    if mode!='bonferroni':
        require(len(set(calibration['adjustments'].values()))==1,'Joint adjustment must be common across targets.')
    result=pd.DataFrame(index=forecasts.index)
    for target in active:
        q=calibration['adjustments'][target]
        if fallback:
            require(q is None,'Fallback must not invent a finite adjustment.')
            lower=np.full(len(forecasts),-1.);upper=np.full(len(forecasts),1.)
        else:
            require(isinstance(q,Real) and not isinstance(q,bool) and np.isfinite(q) and q>=0,'Invalid adjustment.')
            lo=forecasts[target+'__median' if mode=='point_joint' else target+'__lower'].to_numpy(dtype=float)
            hi=forecasts[target+'__median' if mode=='point_joint' else target+'__upper'].to_numpy(dtype=float)
            lower=np.maximum(-1.,lo-q);upper=np.minimum(1.,hi+q)
        result[target+'__lower']=lower;result[target+'__upper']=upper
        result[target+'__median']=forecasts[target+'__median']
    return result


def interval_diagnostics(targets,intervals,index,active_targets):
    require(targets.index.equals(intervals.index) and index.scenario_id.tolist()==targets.index.tolist(),'Diagnostic order differs.')
    result={};covered=[]
    for target in active_targets:
        lo=intervals[target+'__lower'].to_numpy();hi=intervals[target+'__upper'].to_numpy();y=targets[target].to_numpy()
        c=(y>=lo)&(y<=hi);covered.append(c)
        result[target]={'scenario_coverage':float(c.mean()),'mean_width_fraction':float(np.mean(hi-lo))}
    joint=np.logical_and.reduce(covered)
    grouped=pd.DataFrame({'group_id':index.group_id.to_numpy(),'covered':joint}).groupby('group_id').covered.all()
    return {'evaluation':'in_sample_calibration_diagnostic_not_independent_validation','active_targets':list(active_targets),
        'targets':result,'scenario_simultaneous_coverage':float(joint.mean()),'group_all_variant_simultaneous_coverage':float(grouped.mean())}
