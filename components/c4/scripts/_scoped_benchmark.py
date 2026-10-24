"""Selection and scoped admission helpers; no outcome-driven selection."""
import math


def select_training_scenario(manifest):
    candidates = [row for row in manifest['scenarios']
                  if row['split_role'] == 'forecast_train'
                  and row['mechanism'] == 'marginal_multi'
                  and row['magnitude'] == .5
                  and row['diagnosis_condition'] == 'exact_slice'
                  and row['repair_condition'] == 'targeted_repair'
                  and not row['is_no_drift_control']]
    if not candidates:
        raise ValueError('No matching training-fold scenario.')
    return min(candidates, key=lambda row: row['scenario_id'])


def frozen_tail_screen(protocol, synthetic, baseline, *, protocol_kind):
    if protocol_kind == 'pilot':
        rule = protocol['privacy_screen']
        expected_metric = rule['metric']
        ratio = rule['minimum_synthetic_to_heldout_tail_ratio']
        quantile = rule['quantile']
        maximum_copies = rule['maximum_exact_feature_match_rows']
    elif protocol_kind == 'benchmark':
        rule = protocol['admission']
        expected_metric = 'equal_weight_mixed_distance_fit_ranges_clipped'
        if rule['distance_metric'] != 'equal_weight_mixed_fit_range_distance_clipped_at_one':
            raise ValueError('Unsupported frozen distance definition.')
        ratio = rule['minimum_synthetic_p05_over_real_internal_validation_p05_to_same_fit']
        quantile = rule['tail_quantile']
        maximum_copies = rule['maximum_exact_fit_feature_copies']
    else:
        raise ValueError('Unknown protocol kind.')
    if expected_metric != 'equal_weight_mixed_distance_fit_ranges_clipped' or quantile != .05:
        raise ValueError('Unsupported frozen metric or quantile.')
    if ratio != 1. or maximum_copies != 0:
        raise ValueError('This implementation does not silently change frozen screen parameters.')
    values = (synthetic['p05'], baseline['p05'])
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError('Invalid distance summaries.')
    copies = synthetic['exact_feature_match_rows']
    if type(copies) is not int or copies < 0:
        raise ValueError('Invalid copy count.')
    if synthetic['reference_rows'] != baseline['reference_rows']:
        raise ValueError('Comparisons must use the same fit reference population.')
    tail_ratio = None if baseline['p05'] == 0 else synthetic['p05'] / baseline['p05']
    status = ('FAIL' if copies > 0 else 'INDETERMINATE' if tail_ratio is None
              else 'PASS' if tail_ratio >= ratio else 'FAIL')
    return {'screen_status': status, 'tail_distance_ratio': tail_ratio,
            'exact_feature_match_rows': copies, 'synthetic_p05': synthetic['p05'],
            'baseline_p05': baseline['p05'], 'production_privacy_approved': False}


def training_admission_panel(manifest):
    """Fixed six middle-strength targeted conditions plus one control."""
    selected = select_training_scenario(manifest)
    group_id = selected['group_id']
    rows = [row for row in manifest['scenarios']
            if row['group_id'] == group_id and row['split_role'] == 'forecast_train'
            and row['repair_condition'] == 'targeted_repair'
            and ((not row['is_no_drift_control'] and row['magnitude'] == .5)
                 or row['is_no_drift_control'])]
    expected = {(mechanism, diagnosis) for mechanism in ('marginal_multi', 'interaction', 'combined')
                for diagnosis in ('exact_slice', 'broadened_slice')}
    actual = {(row['mechanism'], row['diagnosis_condition']) for row in rows
              if not row['is_no_drift_control']}
    if len(rows) != 7 or actual != expected or sum(row['is_no_drift_control'] for row in rows) != 1:
        raise ValueError('Frozen manifest does not contain the exact seven-case training panel.')
    return sorted(rows, key=lambda row: row['scenario_id'])
