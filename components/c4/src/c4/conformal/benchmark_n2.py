"""Frozen group-max joint conformal calibration for bounded accuracy margins."""
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING

import numpy as np
import pandas as pd

TARGETS = ('future_global_margin', 'future_slice_margin', 'future_nonaffected_margin')


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class JointCalibration:
    alpha: float
    calibration_unit: str
    nonempty_groups: int
    rank: int
    adjustment: float | None
    fallback: bool
    informative_intervals: bool
    auto_deploy_supported: bool
    target_columns: tuple = TARGETS

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values['target_columns'] = tuple(values['target_columns'])
        return cls(**values)


def validate_forecasts(forecasts):
    require(not forecasts.empty and forecasts.index.is_unique and forecasts.columns.is_unique,
            'Forecasts require nonempty unique scenario rows and columns.')
    for target in TARGETS:
        columns = [target+'__'+label for label in ('lower','median','upper')]
        require(all(c in forecasts for c in columns), 'Missing processed N1 quantiles.')
        values = forecasts.loc[:, columns].to_numpy(dtype=float)
        require(np.isfinite(values).all() and (np.abs(values) <= 1).all(), 'Forecasts must be finite margin fractions in [-1,1].')
        require((np.diff(values, axis=1) >= 0).all(), 'Processed N1 quantiles must be ordered.')


def fit_joint_calibration(targets, forecasts, index):
    """Use conformal rows only; one max score per nonempty replication group."""
    validate_forecasts(forecasts)
    require(list(targets.columns) == list(TARGETS) and targets.index.equals(forecasts.index),
            'Targets and forecasts must have identical scenario order and exactly three targets.')
    require(index.scenario_id.tolist() == targets.index.tolist() and index.scenario_id.is_unique,
            'Scenario index order/uniqueness differs.')
    require(index.split_role.eq('conformal_calibration').all(), 'Only conformal_calibration rows may calibrate N2.')
    require(index.group_id.map(lambda x: isinstance(x,str) and bool(x)).all(), 'Valid replication group IDs are required.')
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


def apply_joint_calibration(calibration, forecasts):
    """Inference accepts fixed forecasts only; current future labels are never inputs."""
    validate_forecasts(forecasts)
    require(calibration.alpha == .05 and calibration.target_columns == TARGETS, 'Unexpected calibration definition.')
    require(type(calibration.nonempty_groups) is int and calibration.nonempty_groups >= 0, 'Invalid calibration group count.')
    expected_rank = int((Decimal(calibration.nonempty_groups+1)*Decimal('0.95')).to_integral_value(rounding=ROUND_CEILING))
    require(calibration.rank == expected_rank, 'Calibration rank differs from the frozen finite-sample rule.')
    require(calibration.fallback == (calibration.rank > calibration.nonempty_groups), 'Calibration support flags are inconsistent.')
    require(calibration.informative_intervals is (not calibration.fallback) and
            calibration.auto_deploy_supported is (not calibration.fallback), 'Calibration fallback flags differ.')
    if calibration.fallback:
        require(calibration.adjustment is None, 'Insufficient-support fallback must not invent a finite adjustment.')
    else:
        require(calibration.adjustment is not None and np.isfinite(calibration.adjustment) and calibration.adjustment >= 0,
                'Conformal adjustment must be finite and nonnegative.')
    result = pd.DataFrame(index=forecasts.index)
    for target in TARGETS:
        if calibration.fallback:
            result[target+'__lower'] = -1.
            result[target+'__upper'] = 1.
        else:
            result[target+'__lower'] = np.maximum(-1., forecasts[target+'__lower']-calibration.adjustment)
            result[target+'__upper'] = np.minimum(1., forecasts[target+'__upper']+calibration.adjustment)
        result[target+'__median'] = forecasts[target+'__median']
    return result


def calibration_diagnostics(targets, intervals, index):
    """In-sample calibration diagnostics; never presented as outer-test coverage."""
    require(targets.index.equals(intervals.index) and index.scenario_id.tolist()==targets.index.tolist(),
            'Diagnostic targets, intervals and scenarios must align.')
    covered = []
    result = {}
    groups = index.group_id.to_numpy()
    sizes = pd.Series(groups).value_counts()
    weights = np.array([1./sizes[g] for g in groups])
    for target in TARGETS:
        y=targets[target].to_numpy(dtype=float)
        lower=intervals[target+'__lower'].to_numpy(dtype=float)
        upper=intervals[target+'__upper'].to_numpy(dtype=float)
        c=(y >= lower)&(y <= upper)
        covered.append(c)
        width=upper-lower
        score=width+40.*np.maximum(lower-y,0)+40.*np.maximum(y-upper,0)
        result[target]={'scenario_coverage':float(c.mean()),
            'equal_group_weighted_coverage':float(np.average(c,weights=weights)),
            'mean_width_fraction':float(width.mean()),
            'equal_group_weighted_mean_width_fraction':float(np.average(width,weights=weights)),
            'mean_interval_score_alpha_0_05':float(score.mean()),
            'equal_group_weighted_interval_score_alpha_0_05':float(np.average(score,weights=weights))}
    joint=np.column_stack(covered).all(axis=1)
    group_joint=pd.DataFrame({'group_id':groups,'covered':joint}).groupby('group_id').covered.all()
    return {'evaluation':'in_sample_calibration_diagnostics_not_outer_test', 'targets':result,
        'scenario_joint_coverage':float(joint.mean()),
        'equal_group_weighted_scenario_joint_coverage':float(np.average(joint,weights=weights)),
        'complete_group_joint_coverage':float(group_joint.mean()),
        'independent_coverage_validation':False}
