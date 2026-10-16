"""Paired assigned-group bootstrap with explicit undefined-statistic handling."""
import hashlib
import numpy as np
from c4.evaluation.pilot_evaluation_metrics import PRIMARY_METRICS,metrics_from_counts,scalar,require


def percentile_summary(draw_values,point,minimum_valid_fraction=.95):
    values=np.asarray(draw_values,dtype=float)
    require(values.ndim==1 and len(values)>0 and minimum_valid_fraction==.95,'Frozen bootstrap validity definition differs.')
    valid=values[np.isfinite(values)];enough=len(valid)>=int(np.ceil(len(values)*minimum_valid_fraction))
    interval=[float(v) for v in np.quantile(valid,[.025,.975])] if enough else None
    return {'point_estimate':scalar(point) if point is not None else None,'percentile_95_ci':interval,
        'valid_replicates':len(valid),'undefined_replicates':len(values)-len(valid),
        'minimum_valid_fraction':minimum_valid_fraction,'ci_supported':enough}


def paired_group_bootstrap(counts_by_method,assigned_groups,*,replicates=2000,seed=20261016):
    require(replicates==2000 and seed==20261016,'Frozen bootstrap replicates/seed differ.')
    require('PROPOSED' in counts_by_method and len(assigned_groups)>0 and len(set(assigned_groups))==len(assigned_groups),'Primary method and unique assigned groups required.')
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(assigned_groups),size=(replicates,len(assigned_groups)),dtype=np.int64)
    values={};point={};reports={}
    for method,counts in counts_by_method.items():
        counts=np.asarray(counts,dtype=float)
        require(counts.shape==(len(assigned_groups),9),'Assigned group count matrix differs.')
        # Indexing retains every sampled copy as its own group instance, including empty groups.
        point[method]=metrics_from_counts(counts)
        values[method]=metrics_from_counts(counts[draws])
        reports[method]={metric:percentile_summary(v,point[method][metric]) for metric,v in values[method].items()}
    differences={}
    for method in counts_by_method:
        if method=='PROPOSED':continue
        differences[method]={}
        for metric in PRIMARY_METRICS:
            a=point['PROPOSED'][metric];b=point[method][metric]
            estimate=scalar(a-b) if np.isfinite(a) and np.isfinite(b) else None
            differences[method][metric]=percentile_summary(values['PROPOSED'][metric]-values[method][metric],estimate)
    return {'evaluation':'exploratory_group_bootstrap_not_formal_independent_dataset_inference',
        'replicates':replicates,'seed':seed,'assigned_group_ids':list(assigned_groups),
        'paired_draws_shared_across_methods':True,'draw_indices_sha256':hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
        'methods':reports,'paired_proposed_minus_comparator':differences,
        'multiplicity_controlled_significance_claim':False}
