"""Decision-time safety assessment; no observed future outcomes are accepted."""
import math
from numbers import Real

TARGETS = ('future_global_margin', 'future_slice_margin', 'future_nonaffected_margin')
BOUNDARIES = dict(zip(TARGETS, (0.0, 0.0, -0.02)))
POPULATIONS = dict(zip(TARGETS, ('global', 'true_injected_slice', 'true_slice_complement')))


def assess_safety(intervals, *, auto_deploy_supported):
    """An N3 signal is advisory and never grants production release permission."""
    if type(auto_deploy_supported) is not bool:
        raise ValueError('Calibration support must be an explicit boolean.')
    if set(intervals) != set(TARGETS):
        raise ValueError('Exactly three named margin intervals are required.')
    dimensions = {}
    for target in TARGETS:
        pair = intervals[target]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('Each interval needs lower and upper bounds.')
        lower, upper = pair
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in pair):
            raise ValueError('Bounds must be finite numeric margin fractions.')
        if not -1 <= lower <= upper <= 1:
            raise ValueError('Bounds must be ordered within [-1,1].')
        if not auto_deploy_supported and (lower, upper) != (-1, 1):
            raise ValueError('Unsupported N2 calibration requires the frozen full-range fallback.')
        boundary = BOUNDARIES[target]
        status = 'SAFE' if lower >= boundary else ('UNSAFE' if upper < boundary else 'UNCERTAIN')
        dimensions[target] = {'population': POPULATIONS[target], 'lower': float(lower),
                              'upper': float(upper), 'boundary': boundary, 'status': status}
    statuses = [v['status'] for v in dimensions.values()]
    signal = ('ESCALATE' if 'UNSAFE' in statuses else
              'AUTO_DEPLOY' if all(s == 'SAFE' for s in statuses) and auto_deploy_supported else 'HUMAN_REVIEW')
    return {'dimensions': dimensions, 'release_signal': signal,
            'auto_deploy_supported': auto_deploy_supported, 'release_authorized': False}
