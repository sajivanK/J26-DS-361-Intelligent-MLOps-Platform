"""Paired bootstrap multiplicity, reproducibility and undefined-statistic limits."""
import numpy as np
import pytest
from c4.evaluation.pilot_evaluation_metrics import metrics_from_counts
from c4.evaluation.pilot_group_statistics import percentile_summary,paired_group_bootstrap


def counts():
    return np.array([[1,1,1,1,0,0,0,0,0],[3,3,1,1,2,0,0,0,0],[0,0,0,0,0,0,0,0,0]],dtype=float)


def test_sampled_duplicates_are_separate_group_instances():
    c=counts();duplicated=c[[0,0,1]]
    m=metrics_from_counts(duplicated)
    assert float(m['equal_group_conditional_unsafe_rate'])==pytest.approx((1+1+1/3)/3)
    assert float(m['equal_group_conditional_unsafe_rate'])!=pytest.approx((1+1/3)/2)


def test_empty_sample_instances_do_not_invent_observations():
    c=counts();m=metrics_from_counts(c[[2,2,2]])
    assert np.isnan(m['equal_group_automation']) and np.isnan(m['equal_group_conditional_unsafe_rate'])


def test_null_conditional_risk_has_no_bootstrap_interval():
    c=counts();review=c.copy();review[:,1]=0;review[:,3:5]=0;review[:,5]=review[:,0]
    r=paired_group_bootstrap({'PROPOSED':review,'B6':c},['g1','g2','empty'])
    p=r['methods']['PROPOSED']['equal_group_conditional_unsafe_rate']
    assert p['point_estimate'] is None and p['percentile_95_ci'] is None and p['valid_replicates']==0
    delta=r['paired_proposed_minus_comparator']['B6']['equal_group_conditional_unsafe_rate']
    assert delta['point_estimate'] is None and delta['valid_replicates']==0


def test_shared_paired_draws_give_exact_zero_difference_for_identical_methods():
    c=counts();r=paired_group_bootstrap({'PROPOSED':c,'B8':c.copy()},['g1','g2','empty'])
    assert r['paired_draws_shared_across_methods'] is True
    for metric in r['paired_proposed_minus_comparator']['B8'].values():
        assert metric['point_estimate']==0
        assert metric['percentile_95_ci']==[0.,0.]


def test_bootstrap_seed_and_results_reproduce():
    c=counts();a=paired_group_bootstrap({'PROPOSED':c,'B6':c},['g1','g2','empty'])
    b=paired_group_bootstrap({'PROPOSED':c,'B6':c},['g1','g2','empty'])
    assert a==b


def test_exact_95_percent_valid_replicates_allows_interval():
    r=percentile_summary(np.r_[np.ones(1900),np.full(100,np.nan)],1.)
    assert r['ci_supported'] and r['percentile_95_ci']==[1.,1.]


def test_insufficient_valid_replicates_do_not_zero_fill():
    r=percentile_summary(np.r_[np.ones(1899),np.full(101,np.nan)],1.)
    assert not r['ci_supported'] and r['percentile_95_ci'] is None
    assert r['undefined_replicates']==101


@pytest.mark.parametrize('replicates,seed',[(1000,20261016),(2000,42)])
def test_frozen_resampling_parameters_cannot_change(replicates,seed):
    with pytest.raises(ValueError,match='Frozen bootstrap'):
        paired_group_bootstrap({'PROPOSED':counts()},['g1','g2','empty'],replicates=replicates,seed=seed)
