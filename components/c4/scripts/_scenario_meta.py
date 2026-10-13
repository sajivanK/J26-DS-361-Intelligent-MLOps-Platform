"""Decision-time numeric features; research targets live in separate tables."""
import math
from _common import require

TARGET_COLUMNS = ('future_global_margin', 'future_slice_margin', 'future_nonaffected_margin')


def feature_row(upstream, pre_release):
    values = {}
    def add(name, value):
        require(isinstance(value, (int, float, bool)) and math.isfinite(float(value)),
                f'Feature {name} must be finite and numeric.')
        values[name] = float(value)
    c1, c2, c3 = (upstream[k] for k in ('c1_evidence', 'c2_diagnosis', 'c3_repair_package'))
    for k in ('batch_size', 'drift_detected', 'predicted_harm', 'interaction_suspected', 'detectors_fired'):
        add('c1_' + k, c1[k])
    lower, upper = c1['harm_interval']
    add('c1_harm_lower', lower); add('c1_harm_upper', upper)
    add('c1_harm_width', upper-lower)
    for section in ('detector_scores', 'prediction_stats'):
        for k in sorted(c1[section]): add('c1_' + section + '_' + k, c1[section][k])
    add('c1_affected_feature_count', len(c1['affected_features']))
    add('c1_warning_count', len(c1['warnings']))
    for k in ('slice_support', 'diagnosis_stability'): add('c2_' + k, c2[k])
    add('c2_ranked_feature_count', len(c2['affected_features_ranked']))
    for k in ('n_synthetic_rows', 'augmentation_ratio', 'fidelity_score', 'utility_score',
              'target_coverage', 'prescription_confidence'): add('c3_' + k, c3[k])
    add('c3_repair_approved', c3['repair_package_status'] == 'APPROVED')
    add('c3_privacy_screen_pass', c3['privacy_status'] == 'PASS')
    comparisons = pre_release['report']['comparisons']
    for population in ('global', 'affected_slice', 'non_affected'):
        report = comparisons[population]
        add('c4_' + population + '_rows', report['rows'])
        for model in ('champion', 'candidate', 'candidate_minus_champion'):
            for metric in ('accuracy', 'macro_f1', 'auc'):
                value = report[model][metric]
                key = 'c4_' + population + '_' + model + '_' + metric
                # A fixed encoding for an undefined one-class AUC, never fitted on targets.
                if metric == 'auc': add(key + '_missing', value is None)
                add(key, 0. if value is None and metric == 'auc' else value)
    return dict(sorted(values.items()))


def validated_targets(outcomes):
    require(outcomes['evaluation_role'] == 'sealed_future' and
            outcomes['primary_slice_basis'] == 'true_injected_slice' and
            outcomes['target_units'] == 'accuracy_margin_fraction', 'Wrong research target definition.')
    raw = outcomes['primary_targets']
    require(set(raw) == set(TARGET_COLUMNS), 'Exactly three future targets are required.')
    result = {}
    for key in TARGET_COLUMNS:
        value = raw[key]
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(value) and -1 <= value <= 1, 'Future margins must be finite fractions.')
        result[key] = float(value)
    return result
