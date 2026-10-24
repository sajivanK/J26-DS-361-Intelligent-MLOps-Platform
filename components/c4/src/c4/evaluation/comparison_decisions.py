"""Decision-time-only B1-B8 / A1-A8 pilot comparison rules."""
import math
import pandas as pd
from c4.risk.benchmark_n3 import TARGETS,BOUNDARIES
from c4.risk.benchmark_policy import GATES,CONTEXTS,number,evidence_context,policy_decision
from c4.evaluation.comparison_calibration import apply_comparison_calibration

METHOD_IDS=('PROPOSED',*[f'B{i}' for i in range(1,9)],*[f'A{i}' for i in range(1,9)])
POLICY_VARIANTS=('A1','A2','A3','A4','A5','B5','A6')


def require(condition,message):
    if not condition:raise ValueError(message)


def policy_for_active_targets(intervals,features,gates,*,supported,policy,unsupported_contexts=()):
    """A4 omits the slice decision requirement; other variants use the existing engine."""
    if set(intervals)==set(TARGETS):
        return policy_decision(intervals,features,gates,n2_supported=supported,policy=policy,unsupported_contexts=unsupported_contexts)
    require(set(intervals)=={'future_global_margin','future_nonaffected_margin'},'Only A4 may omit the slice decision target.')
    require(type(supported) is bool and isinstance(gates,dict),'Explicit support/gates required.')
    failed=[g for g in GATES if gates.get(g) is not True]
    if failed:return {'decision':'ESCALATE','reason':'mandatory_gate_failed_or_unknown','failed_gates':failed,'release_authorized':False}
    dimensions={}
    for target,pair in intervals.items():
        require(isinstance(pair,(list,tuple)) and len(pair)==2,'Two ordered bounds required.')
        lo=number(pair[0],-1,1,'Lower bound');hi=number(pair[1],-1,1,'Upper bound')
        require(lo<=hi and (supported or (lo,hi)==(-1.,1.)),'Invalid interval/support combination.')
        b=BOUNDARIES[target];status='SAFE' if lo>=b else 'UNSAFE' if hi<b else 'UNCERTAIN'
        dimensions[target]={'lower':lo,'upper':hi,'boundary':b,'status':status}
    base={'release_authorized':False,'active_targets':list(intervals),'dimensions':dimensions}
    if any(d['status']=='UNSAFE' for d in dimensions.values()):return {**base,'decision':'ESCALATE','reason':'N3_unsafe_dimension'}
    if not supported or any(d['status']!='SAFE' for d in dimensions.values()):return {**base,'decision':'HUMAN_REVIEW','reason':'N2_unsupported_or_N3_uncertain'}
    if policy['kind']=='review_only':return {**base,'decision':'HUMAN_REVIEW','reason':'policy_selection_unsupported'}
    context=evidence_context(features)
    if policy['kind']=='global':buffer=number(policy['buffer'],0,1,'Global buffer')
    else:
        require(policy['kind']=='adaptive' and set(policy['buffers'])==set(CONTEXTS),'Unknown adaptive policy.')
        values=[number(policy['buffers'][c],0,1,'Context buffer') for c in CONTEXTS]
        require(values==sorted(values) and set(unsupported_contexts)<=set(CONTEXTS),'Invalid monotone context policy.')
        if context['context'] in unsupported_contexts:return {**base,'decision':'HUMAN_REVIEW','reason':'insufficient_policy_context_groups'}
        buffer=policy['buffers'][context['context']]
    slack=min(d['lower']-d['boundary'] for d in dimensions.values())
    return {**base,'decision':'AUTO_DEPLOY' if slack>=buffer else 'HUMAN_REVIEW',
        'reason':'additional_buffer_satisfied' if slack>=buffer else 'additional_buffer_not_satisfied',
        'minimum_lower_bound_slack':slack,'additional_buffer':buffer}


def comparison_bounds(comparison,forecasts,calibrations):
    interval_id='B5' if comparison=='A6' else comparison
    calibration=calibrations[interval_id]
    adjusted=apply_comparison_calibration(calibration,forecasts).iloc[0]
    return {t:[float(adjusted[t+'__lower']),float(adjusted[t+'__upper'])] for t in calibration['active_targets']},calibration['auto_deploy_supported']


def decide_comparison(comparison,features,gates,forecast,calibrations,policies,unsupported_contexts=()):
    """No future targets, observed margins or ground-truth memberships are inputs."""
    require(comparison in METHOD_IDS,'Unknown comparison ID.')
    require(isinstance(gates,dict),'Explicit hard gates required.')
    failed=[g for g in GATES if gates.get(g) is not True]
    base={'comparison':comparison,'release_authorized':False}
    if failed:return {**base,'decision':'ESCALATE','reason':'mandatory_gate_failed_or_unknown','failed_gates':failed}
    if comparison in {'B6','B7'}:
        return {**base,'decision':'AUTO_DEPLOY' if comparison=='B6' else 'HUMAN_REVIEW','reason':'fixed_comparator_after_gates'}
    if comparison in {'B1','B2'}:
        global_margin=number(features['c4_global_candidate_minus_champion_accuracy'],-1,1,'Pre-release global accuracy margin')
        values=[global_margin]
        if comparison=='B2':values.append(number(features['c4_affected_slice_candidate_minus_champion_accuracy'],-1,1,'Pre-release diagnosed-slice accuracy margin'))
        return {**base,'decision':'AUTO_DEPLOY' if min(values)>=0 else 'ESCALATE','reason':'fixed_pre_release_accuracy_rule'}
    if comparison=='B3':
        quality=evidence_context(features)['quality']
        return {**base,'decision':'AUTO_DEPLOY' if quality>=.75 else 'HUMAN_REVIEW','reason':'heuristic_quality_threshold'}
    if comparison=='B4':
        margins=[number(forecast[t+'__median'],-1,1,'N1 median')-BOUNDARIES[t] for t in TARGETS]
        return {**base,'decision':'AUTO_DEPLOY' if min(margins)>=0 else 'ESCALATE','reason':'uncalibrated_point_forecast'}
    frame=pd.DataFrame([forecast])
    if comparison in {'PROPOSED','B8','A7','A8'}:
        bounds,supported=comparison_bounds('ORIGINAL',frame,calibrations)
        policy=policies['PROPOSED' if comparison=='PROPOSED' else 'B8']
        if comparison=='A7':policy={'policy_id':'A7-fixed-zero-buffer','kind':'adaptive','buffers':dict.fromkeys(CONTEXTS,0.)}
    else:
        bounds,supported=comparison_bounds(comparison,frame,calibrations);policy=policies[comparison]
    result=policy_for_active_targets(bounds,features,gates,supported=supported,policy=policy,unsupported_contexts=unsupported_contexts)
    return {**result,**base}
