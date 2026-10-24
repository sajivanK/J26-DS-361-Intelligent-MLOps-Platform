"""Decision-time global/adaptive buffer policies subordinate to fixed N3 safety."""
import math
from itertools import combinations_with_replacement
from numbers import Real
from c4.risk.benchmark_n3 import assess_safety

GATES=('repair_approved','privacy_check','diagnosis_stability','schema_compatible','rollback_ready')
CONTEXTS=('LOW','MEDIUM','HIGH')
BUFFER_GRID=(0.0,0.001,0.0025,0.005)


def number(value,low,high,label):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError(label+' must be finite and in range.')
    return float(value)


def evidence_context(evidence):
    """Use four decision-time features; quality is a heuristic, not a probability."""
    width=number(evidence['c1_harm_width'],0,math.inf,'Harm interval width')
    stability=number(evidence['c2_diagnosis_stability'],0,1,'Diagnosis stability')
    fidelity=number(evidence['c3_fidelity_score'],0,1,'Fidelity')
    utility=number(evidence['c3_utility_score'],-1,1,'Signed utility')
    quality=.35*stability+.25*max(0.,1.-10.*width)+.25*fidelity+.15*max(0.,min(1.,utility+.1))
    context='LOW' if quality>=.75 else 'MEDIUM' if quality>=.50 else 'HIGH'
    return {'quality':quality,'context':context,'quality_is_calibrated_probability':False}


def candidate_policies():
    global_policies=[{'policy_id':f'global-{i:02d}','kind':'global','buffer':b} for i,b in enumerate(BUFFER_GRID)]
    adaptive=[{'policy_id':f'adaptive-{i:02d}','kind':'adaptive','buffers':dict(zip(CONTEXTS,values))}
              for i,values in enumerate(combinations_with_replacement(BUFFER_GRID,3))]
    return {'global':global_policies,'adaptive':adaptive}


def policy_decision(intervals,evidence,gates,*,n2_supported,policy,unsupported_contexts=()):
    """Return a research decision signal; no observed future outcomes are inputs."""
    if type(n2_supported) is not bool:
        raise ValueError('N2 support must be an explicit boolean.')
    if not isinstance(gates,dict):
        raise ValueError('Explicit hard-gate results are required.')
    failed=[g for g in GATES if gates.get(g) is not True]
    if failed:
        return {'decision':'ESCALATE','reason':'mandatory_gate_failed_or_unknown','failed_gates':failed,
                'release_authorized':False}
    safety=assess_safety(intervals,auto_deploy_supported=n2_supported)
    result={'decision':safety['release_signal'],'n3':safety,'release_authorized':False}
    if safety['release_signal']=='ESCALATE':
        return {**result,'reason':'N3_unsafe_dimension'}
    if not n2_supported or safety['release_signal']!='AUTO_DEPLOY':
        return {**result,'decision':'HUMAN_REVIEW','reason':'N2_unsupported_or_N3_uncertain'}
    context=evidence_context(evidence)
    result['evidence_context']=context
    if policy['kind']=='review_only':
        return {**result,'decision':'HUMAN_REVIEW','reason':'policy_selection_unsupported'}
    if policy['kind']=='global':
        buffer=number(policy['buffer'],0,1,'Global buffer')
    elif policy['kind']=='adaptive':
        if set(policy['buffers'])!=set(CONTEXTS):raise ValueError('Three explicit context buffers required.')
        values=[number(policy['buffers'][c],0,1,'Adaptive buffer') for c in CONTEXTS]
        if values!=sorted(values):raise ValueError('Risk-context buffers must be monotone.')
        if not set(unsupported_contexts)<=set(CONTEXTS):raise ValueError('Unknown unsupported context.')
        if context['context'] in unsupported_contexts:
            return {**result,'decision':'HUMAN_REVIEW','reason':'insufficient_policy_context_groups'}
        buffer=policy['buffers'][context['context']]
    else:
        raise ValueError('Unknown policy kind.')
    slack=min(v['lower']-v['boundary'] for v in safety['dimensions'].values())
    return {**result,'decision':'AUTO_DEPLOY' if slack>=buffer else 'HUMAN_REVIEW',
            'reason':'additional_buffer_satisfied' if slack>=buffer else 'additional_buffer_not_satisfied',
            'minimum_lower_bound_slack':slack,'additional_buffer':float(buffer)}
