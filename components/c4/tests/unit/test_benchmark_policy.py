"""Policy strictness must never override mandatory safety or uncertainty."""
import ast
from pathlib import Path
import pytest
from c4.risk.benchmark_n3 import TARGETS,BOUNDARIES
from c4.risk.benchmark_policy import GATES,BUFFER_GRID,candidate_policies,evidence_context,policy_decision


def intervals(slack=.003):
    return {t:[b+slack,b+slack+.01] for t,b in BOUNDARIES.items()}


def evidence():
    return {'c1_harm_width':.01,'c2_diagnosis_stability':.9,'c3_fidelity_score':.9,'c3_utility_score':.01}


def gates():return dict.fromkeys(GATES,True)


def test_fixed_candidate_grid_and_monotone_context_buffers():
    c=candidate_policies()
    assert len(c['global'])==4 and len(c['adaptive'])==20
    assert [r['buffer'] for r in c['global']]==list(BUFFER_GRID)
    for r in c['adaptive']:
        assert list(r['buffers'].values())==sorted(r['buffers'].values())


def test_evidence_quality_matches_guide_formula():
    e=evidence();q=evidence_context(e)
    expected=.35*.9+.25*.9+.25*.9+.15*.11
    assert q['quality']==pytest.approx(expected)
    assert q['context']=='LOW' and not q['quality_is_calibrated_probability']


def test_medium_quality_boundary_is_inclusive():
    q=evidence_context({'c1_harm_width':0.,'c2_diagnosis_stability':0.,'c3_fidelity_score':1.,'c3_utility_score':-.1})
    assert q['quality']==.5 and q['context']=='MEDIUM'


def test_low_quality_is_high_risk_context():
    q=evidence_context({'c1_harm_width':.2,'c2_diagnosis_stability':0.,'c3_fidelity_score':0.,'c3_utility_score':-.5})
    assert q['context']=='HIGH'


@pytest.mark.parametrize('gate',GATES)
def test_any_failed_gate_escalates_before_other_inputs(gate):
    g=gates();g[gate]=False
    r=policy_decision({}, {},g,n2_supported=True,policy={'kind':'global','buffer':0.})
    assert r['decision']=='ESCALATE' and r['release_authorized'] is False


def test_missing_gate_is_not_assumed_true():
    g=gates();g.pop('rollback_ready')
    assert policy_decision({}, {},g,n2_supported=True,policy={})['decision']=='ESCALATE'


def test_n3_uncertainty_cannot_be_bypassed_by_zero_buffer():
    f=intervals();f[TARGETS[0]]=[-.01,.01]
    assert policy_decision(f,evidence(),gates(),n2_supported=True,policy={'kind':'global','buffer':0.})['decision']=='HUMAN_REVIEW'


def test_n2_fallback_cannot_be_bypassed():
    f={t:[-1.,1.] for t in TARGETS}
    assert policy_decision(f,evidence(),gates(),n2_supported=False,policy={'kind':'global','buffer':0.})['decision']=='HUMAN_REVIEW'


def test_unsafe_dimension_still_escalates_under_review_only_policy():
    f=intervals();f[TARGETS[0]]=[-.03,-.01]
    assert policy_decision(f,evidence(),gates(),n2_supported=True,policy={'kind':'review_only'})['decision']=='ESCALATE'


def test_additional_buffer_can_only_restrict_deployment():
    a=policy_decision(intervals(),evidence(),gates(),n2_supported=True,policy={'kind':'global','buffer':.001})
    b=policy_decision(intervals(),evidence(),gates(),n2_supported=True,policy={'kind':'global','buffer':.005})
    assert a['decision']=='AUTO_DEPLOY' and b['decision']=='HUMAN_REVIEW'
    assert a['release_authorized'] is False


def test_adaptive_uses_pre_release_context():
    p={'kind':'adaptive','buffers':{'LOW':.001,'MEDIUM':.0025,'HIGH':.005}}
    assert policy_decision(intervals(),evidence(),gates(),n2_supported=True,policy=p)['decision']=='AUTO_DEPLOY'
    e=evidence();e.update(c1_harm_width=.2,c2_diagnosis_stability=.1,c3_fidelity_score=.1,c3_utility_score=-.5)
    assert policy_decision(intervals(),e,gates(),n2_supported=True,policy=p)['decision']=='HUMAN_REVIEW'


def test_sparse_context_forces_review():
    p={'kind':'adaptive','buffers':{'LOW':0.,'MEDIUM':.001,'HIGH':.005}}
    r=policy_decision(intervals(),evidence(),gates(),n2_supported=True,policy=p,unsupported_contexts=('LOW',))
    assert r['decision']=='HUMAN_REVIEW'


def test_unsupported_selection_cannot_deploy():
    assert policy_decision(intervals(),evidence(),gates(),n2_supported=True,policy={'kind':'review_only'})['decision']=='HUMAN_REVIEW'


def test_nonmonotone_buffers_rejected():
    p={'kind':'adaptive','buffers':{'LOW':.005,'MEDIUM':0.,'HIGH':0.}}
    with pytest.raises(ValueError):policy_decision(intervals(),evidence(),gates(),n2_supported=True,policy=p)


@pytest.mark.parametrize('value',[float('nan'),-1.,float('inf')])
def test_invalid_harm_width_rejected(value):
    e=evidence();e['c1_harm_width']=value
    with pytest.raises(ValueError):evidence_context(e)


def test_engine_has_no_training_or_future_target_access():
    path=Path(__file__).resolve().parents[2]/'src/c4/risk/benchmark_policy.py'
    tree=ast.parse(path.read_text())
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&{'fit','predict','read_csv','read_text','read_bytes'}
