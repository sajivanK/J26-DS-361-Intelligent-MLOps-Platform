"""Frozen PP1 evaluation metrics; outcomes are used for evaluation, never decisions."""
import numpy as np
import pandas as pd
from c4.risk.benchmark_n3 import TARGETS,BOUNDARIES
from c4.conformal.benchmark_n2 import validate_forecasts

COUNT_COLUMNS=('n','auto','unsafe','unsafe_auto','safe_auto','review','escalate','unsafe_escalate','safe_escalate')
RATIO_NAMES=('equal_group_unsafe_auto_frequency','equal_group_automation','equal_group_conditional_unsafe_rate',
             'safe_automation_coverage','safe_automation_yield','human_review_burden','correct_escalation_recall','escalation_precision')
PRIMARY_METRICS=RATIO_NAMES[:3]


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_outcomes(targets,index,assigned_groups):
    require(targets.columns.tolist()==list(TARGETS) and targets.index.is_unique and index.scenario_id.is_unique
            and targets.index.tolist()==index.scenario_id.tolist(),'Unique aligned scenario identities and three targets required.')
    require(len(assigned_groups)>0 and len(assigned_groups)==len(set(assigned_groups))
            and all(isinstance(g,str) and g for g in assigned_groups),'Unique assigned replication groups required.')
    require(index.group_id.map(lambda g:g in assigned_groups).all(),'Observed group outside assigned inventory.')
    require(index.split_role.isin(['policy_calibration','outer_test']).all() and index.split_role.nunique()<=1,
            'Evaluate policy or outer fold separately; never mix folds.')
    require(all(pd.api.types.is_numeric_dtype(targets[t]) and not pd.api.types.is_bool_dtype(targets[t]) for t in TARGETS),'Numeric non-Boolean margins required.')
    y=targets.to_numpy(dtype=float)
    require(np.isfinite(y).all() and (np.abs(y)<=1).all(),'Future margins must be finite fractions in [-1,1].')


def group_counts(targets,index,decisions,assigned_groups):
    validate_outcomes(targets,index,assigned_groups)
    require(set(decisions)==set(targets.index),'Decision inventory differs.')
    d=np.array([decisions[sid] for sid in targets.index])
    require(np.isin(d,['AUTO_DEPLOY','HUMAN_REVIEW','ESCALATE']).all(),'Unknown decision.')
    unsafe=(targets.to_numpy(dtype=float)<np.array([BOUNDARIES[t] for t in TARGETS])).any(axis=1)
    rows=[]
    for gid in assigned_groups:
        mask=index.group_id.eq(gid).to_numpy();a=(d=='AUTO_DEPLOY')&mask;r=(d=='HUMAN_REVIEW')&mask;e=(d=='ESCALATE')&mask
        rows.append([int(mask.sum()),int(a.sum()),int((unsafe&mask).sum()),int((unsafe&a).sum()),int((~unsafe&a).sum()),
                     int(r.sum()),int(e.sum()),int((unsafe&e).sum()),int((~unsafe&e).sum())])
    return np.asarray(rows,dtype=float)


def ratio_pairs(counts):
    c=np.asarray(counts,dtype=float)
    require(c.ndim>=2 and c.shape[-1]==len(COUNT_COLUMNS) and np.isfinite(c).all() and (c>=0).all(),'Invalid group counts.')
    n,a,u,ua,sa,r,e,ue,se=np.moveaxis(c,-1,0)
    return ((ua,n),(a,n),(ua,a),(sa,n-u),(sa,n),(r,n),(ue,u),(ue,e))


def nan_group_mean(numerator,denominator):
    values=np.full_like(numerator,np.nan,dtype=float)
    np.divide(numerator,denominator,out=values,where=denominator>0)
    valid=np.isfinite(values);count=valid.sum(axis=-1)
    total=np.where(valid,values,0.).sum(axis=-1)
    out=np.full(np.shape(total),np.nan,dtype=float)
    np.divide(total,count,out=out,where=count>0)
    return out


def metrics_from_counts(counts,unsafe_cost=10.):
    require(isinstance(unsafe_cost,(int,float)) and not isinstance(unsafe_cost,bool) and np.isfinite(unsafe_cost) and unsafe_cost>=0,'Invalid illustrative cost.')
    result={name:nan_group_mean(num,den) for name,(num,den) in zip(RATIO_NAMES,ratio_pairs(counts))}
    c=np.asarray(counts,dtype=float)
    result['expected_deployment_cost']=nan_group_mean(unsafe_cost*c[...,3]+c[...,5]+2*c[...,8],c[...,0])
    return result


def scalar(value):
    return float(value) if np.isfinite(value) else None


def governance_report(targets,index,decisions,assigned_groups):
    counts=group_counts(targets,index,decisions,assigned_groups)
    report={name:scalar(value) for name,value in metrics_from_counts(counts).items()}
    totals=counts.sum(axis=0)
    report['counts']={name:int(value) for name,value in zip(COUNT_COLUMNS,totals)}
    report['assigned_groups']=len(assigned_groups);report['nonempty_groups']=int((counts[:,0]>0).sum())
    report['groups_without_target_rows']=int((counts[:,0]==0).sum())
    report['ratio_group_support']={name:int((den>0).sum()) for name,(_,den) in zip(RATIO_NAMES,ratio_pairs(counts))}
    report['scenario_weighted']={name.replace('equal_group_',''):(float(num[0]/den[0]) if den[0]>0 else None)
        for name,(num,den) in zip(RATIO_NAMES,ratio_pairs(totals[None,:]))}
    report['cost_sensitivity']={str(cost):scalar(metrics_from_counts(counts,cost)['expected_deployment_cost']) for cost in (1.,5.,10.,20.,100.)}
    report['per_group']=[{'group_id':gid,**{k:int(v) for k,v in zip(COUNT_COLUMNS,row)}} for gid,row in zip(assigned_groups,counts)]
    return report


def weights_for(index):
    sizes=index.group_id.value_counts()
    return np.array([1./sizes[g] for g in index.group_id],dtype=float)


def forecast_report(targets,forecasts,index,assigned_groups):
    validate_outcomes(targets,index,assigned_groups);validate_forecasts(forecasts)
    require(targets.index.equals(forecasts.index),'Forecast order differs.')
    result={}
    for target in TARGETS:
        y=targets[target].to_numpy(dtype=float);median=forecasts[target+'__median'].to_numpy(dtype=float)
        reports={}
        for label,w in [('scenario_weighted',np.ones(len(y))),('equal_group_weighted',weights_for(index))]:
            mse=float(np.average((y-median)**2,weights=w));center=float(np.average(y,weights=w))
            variance=float(np.average((y-center)**2,weights=w))
            scores={'mae':float(np.average(np.abs(y-median),weights=w)),'rmse':float(np.sqrt(mse)),
                'r2':None if np.ptp(y)==0 or variance==0 else float(1.-mse/variance),'pinball':{}}
            for q,suffix in [(.05,'lower'),(.5,'median'),(.95,'upper')]:
                residual=y-forecasts[target+'__'+suffix].to_numpy(dtype=float)
                scores['pinball'][str(q)]=float(np.average(np.maximum(q*residual,(q-1)*residual),weights=w))
            raw_columns=[target+'__raw_'+suffix for suffix in ('lower','median','upper')]
            if all(c in forecasts for c in raw_columns):
                raw=forecasts[raw_columns].to_numpy(dtype=float)
                require(np.isfinite(raw).all(),'Raw forecasts must be finite.')
                scores['raw_pinball']={}
                for n,q in enumerate((.05,.5,.95)):
                    residual=y-raw[:,n]
                    scores['raw_pinball'][str(q)]=float(np.average(np.maximum(q*residual,(q-1)*residual),weights=w))
            reports[label]=scores
        raw_columns=[target+'__raw_'+suffix for suffix in ('lower','median','upper')]
        reports['raw_quantile_crossing_rows']=int((np.diff(forecasts[raw_columns].to_numpy(dtype=float),axis=1)<0).any(axis=1).sum()) if all(c in forecasts for c in raw_columns) else None
        result[target]=reports
    return result


def interval_report(targets,intervals,index,assigned_groups,active_targets,native_alpha=None):
    validate_outcomes(targets,index,assigned_groups)
    require(targets.index.equals(intervals.index) and len(targets)>0,'Interval rows must align and be nonempty.')
    require(tuple(active_targets) in {TARGETS,('future_global_margin','future_nonaffected_margin')},'Unknown active target set.')
    if native_alpha is not None:require(native_alpha==.05/3,'Only declared Bonferroni native alpha allowed.')
    w=weights_for(index);covered=[];result={}
    for target in active_targets:
        y=targets[target].to_numpy(dtype=float);lo=intervals[target+'__lower'].to_numpy(dtype=float);hi=intervals[target+'__upper'].to_numpy(dtype=float)
        require(np.isfinite(lo).all() and np.isfinite(hi).all() and (lo>=-1).all() and (hi<=1).all() and (lo<=hi).all(),'Invalid bounded intervals.')
        c=(y>=lo)&(y<=hi);covered.append(c);width=hi-lo;penalty=np.maximum(lo-y,0)+np.maximum(y-hi,0)
        result[target]={label:{'coverage':float(np.average(c,weights=weight)),
            'mean_width_fraction':float(np.average(width,weights=weight)),
            'mean_width_normalized_range_2':float(np.average(width/2,weights=weight)),
            'interval_score_alpha_005':float(np.average(width+40*penalty,weights=weight)),
            **({'interval_score_native_bonferroni':float(np.average(width+2/native_alpha*penalty,weights=weight))} if native_alpha is not None else {})}
            for label,weight in [('scenario_weighted',np.ones(len(y))),('equal_group_weighted',w)]}
    joint=np.logical_and.reduce(covered)
    group_joint=pd.DataFrame({'group_id':index.group_id.to_numpy(),'covered':joint}).groupby('group_id').covered.all()
    return {'active_targets':list(active_targets),'three_target_comparable':tuple(active_targets)==TARGETS,
        'targets':result,'scenario_simultaneous_coverage':float(joint.mean()),
        'equal_group_simultaneous_coverage':float(np.average(joint,weights=w)),
        'group_all_variant_simultaneous_coverage':float(group_joint.mean()),
        'nonempty_groups':len(group_joint),'group_covered_count':int(group_joint.sum())}


def diagnostic_risk_coverage(scores,targets,index,assigned_groups,gate_pass):
    validate_outcomes(targets,index,assigned_groups)
    require(len(gate_pass)==len(targets) and all(type(v) is bool for v in gate_pass),'Explicit gate-pass booleans required.')
    if scores is None:return {'kind':'diagnostic_forced_score_prefix_not_operational_policy','aurc':None,'curve':[],'reason':'no_ranking_score'}
    scores=np.asarray(scores,dtype=float)
    require(scores.shape==(len(targets),) and np.isfinite(scores).all(),'Finite aligned scores required.')
    allowed=np.asarray(gate_pass,dtype=bool);thresholds=np.unique(scores[allowed])[::-1]
    curve=[];area=0.;previous=0.
    for threshold in thresholds:
        decisions={sid:'AUTO_DEPLOY' if allowed[n] and scores[n]>=threshold else 'HUMAN_REVIEW' for n,sid in enumerate(targets.index)}
        c=group_counts(targets,index,decisions,assigned_groups);m=metrics_from_counts(c)
        coverage=float(m['equal_group_automation']);risk=float(m['equal_group_conditional_unsafe_rate'])
        area+=risk*(coverage-previous);previous=coverage
        curve.append({'score_threshold':float(threshold),'equal_group_coverage':coverage,'conditional_unsafe_rate':risk,
                      'auto_scenarios':int(c[:,1].sum()),'auto_groups':int((c[:,1]>0).sum())})
    return {'kind':'diagnostic_forced_score_prefix_not_operational_policy','aurc':float(area) if len(thresholds)>=2 else None,
        'curve':curve,'distinct_score_blocks':len(thresholds),'reason':'score_prefix_diagnostic_only' if len(thresholds)>=2 else 'fewer_than_two_distinct_score_blocks'}


def comparison_diagnostic_scores(comparison,features,forecasts,runtime):
    """Frozen ranking definitions only; no targets enter score construction."""
    from c4.evaluation.comparison_decisions import comparison_bounds
    from c4.risk.benchmark_policy import evidence_context
    require(features.index.equals(forecasts.index),'Diagnostic feature/forecast order differs.')
    if comparison in {'B6','B7'}:return None
    if comparison=='B1':return features['c4_global_candidate_minus_champion_accuracy'].to_numpy(dtype=float)
    if comparison=='B2':return features[['c4_global_candidate_minus_champion_accuracy','c4_affected_slice_candidate_minus_champion_accuracy']].min(axis=1).to_numpy(dtype=float)
    if comparison=='B3':return np.array([evidence_context(row.to_dict())['quality']-.75 for _,row in features.iterrows()])
    if comparison=='B4':return np.min(np.column_stack([forecasts[t+'__median'].to_numpy()-BOUNDARIES[t] for t in TARGETS]),axis=1)
    interval_id='ORIGINAL' if comparison in {'PROPOSED','B8','A7','A8'} else comparison
    scores=[]
    for sid,row in features.iterrows():
        bounds,_=comparison_bounds(interval_id,forecasts.loc[[sid]],runtime['calibrations'])
        slack=min(pair[0]-BOUNDARIES[t] for t,pair in bounds.items())
        if comparison!='A7':
            policy=runtime['selected_policies']['B8' if comparison in {'B8','A8'} else comparison]
            if policy['kind']=='global':slack-=policy['buffer']
            elif policy['kind']=='adaptive':slack-=policy['buffers'][evidence_context(row.to_dict())['context']]
            else:require(policy['kind']=='review_only','Unknown selected policy.')
        scores.append(slack)
    return np.array(scores,dtype=float)
