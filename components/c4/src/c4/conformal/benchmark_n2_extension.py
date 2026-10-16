"""Frozen score pooling for the separately versioned calibration-support extension."""
from decimal import Decimal, ROUND_CEILING
import numpy as np
import pandas as pd
from c4.conformal.benchmark_n2 import TARGETS, JointCalibration, validate_forecasts, require


def fit_extended_calibration(targets, forecasts, index):
    """Use conformal rows only; one max score per nonempty replication group."""
    validate_forecasts(forecasts)
    require(list(targets.columns) == list(TARGETS) and targets.index.equals(forecasts.index),
            'Targets and forecasts must have identical scenario order and exactly three targets.')
    require(index.scenario_id.tolist() == targets.index.tolist() and index.scenario_id.is_unique,
            'Scenario index order/uniqueness differs.')
    require(index.split_role.isin(['conformal_calibration','conformal_extension']).all(), 'Only original and extension calibration roles are permitted.')
    require(index.group_id.map(lambda x: isinstance(x,str) and bool(x)).all(), 'Valid replication group IDs are required.')
    require(index.groupby('group_id').split_role.nunique().eq(1).all(), 'A group cannot belong to multiple folds.')
    y = targets.to_numpy(dtype=float)
    require(np.isfinite(y).all() and (np.abs(y) <= 1).all(), 'Targets must be finite accuracy-margin fractions in [-1,1].')
    scores = pd.DataFrame(index=targets.index)
    for target in TARGETS:
        actual = targets[target].to_numpy(dtype=float)
        scores[target+'__nonconformity'] = np.maximum.reduce([
            np.zeros(len(actual)), forecasts[target+'__lower'].to_numpy(dtype=float)-actual,
            actual-forecasts[target+'__upper'].to_numpy(dtype=float)])
    scores['joint_score'] = scores.max(axis=1)
    scores['group_id'] = index.group_id.to_numpy()
    grouped = scores.groupby('group_id', sort=True).agg(group_score=('joint_score','max'), scenario_count=('joint_score','size'))
    count = len(grouped)
    rank = int((Decimal(count+1)*Decimal('0.95')).to_integral_value(rounding=ROUND_CEILING))
    fallback = rank > count
    adjustment = None if fallback else float(np.sort(grouped.group_score.to_numpy())[rank-1])
    # This flag expresses numerical support only; it is not production approval or an exchangeability proof.
    calibration = JointCalibration(.05, 'replication_group_max_over_admitted_variants_and_three_targets',
        count, rank, adjustment, fallback, not fallback, not fallback)
    return calibration, scores, grouped

