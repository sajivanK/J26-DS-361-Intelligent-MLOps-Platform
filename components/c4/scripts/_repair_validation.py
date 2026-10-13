"""Explicit draft screening decisions, separate from production approval."""
import math


def assess_privacy_screen(synthetic_summary, baseline_summary):
    """v1: zero exact feature copies and p05 ratio >= 1; no approval export."""
    synthetic = synthetic_summary['p05']
    baseline = baseline_summary['p05']
    copies = synthetic_summary['exact_feature_match_rows']
    for value in (synthetic, baseline):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('Tail distances must be numeric.')
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Tail distances must be finite and within [0, 1].')
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 0:
        raise ValueError('Exact-match count must be a nonnegative integer.')
    if copies > 0:
        status = 'FAIL'
        reason = 'At least one synthetic row exactly matches generator-fit features.'
    elif baseline == 0:
        status = 'INDETERMINATE'
        reason = 'The held-out lower-tail baseline is zero; the ratio rule is undefined.'
    elif synthetic < baseline:
        status = 'FAIL'
        reason = 'Synthetic p05 distance is smaller than held-out real p05 distance to the same fit reference.'
    else:
        status = 'PASS'
        reason = 'No exact feature copies and the proposed lower-tail distance rule passes.'
    return {
        'screen_status': status, 'reason': reason,
        'synthetic_p05': synthetic, 'baseline_p05': baseline,
        'tail_distance_ratio': None if baseline == 0 else synthetic / baseline,
        'exact_feature_match_rows': copies,
        'privacy_status': 'NOT_EVALUATED',
        'repair_package_approved': False,
        'candidate_training_authorized': False,
        'interpretation': 'Exploratory draft screen only; PASS is not C3 privacy approval.',
    }
