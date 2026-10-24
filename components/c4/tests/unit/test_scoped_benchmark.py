import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('c4_scoped_benchmark', Path(__file__).resolve().parents[2] / 'scripts/_scoped_benchmark.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def protocol():
    return {'privacy_screen': {'metric': 'equal_weight_mixed_distance_fit_ranges_clipped',
                              'minimum_synthetic_to_heldout_tail_ratio': 1., 'quantile': .05,
                              'maximum_exact_feature_match_rows': 0}}


def summaries(a=.01, b=.02, copies=0):
    return ({'p05': a, 'exact_feature_match_rows': copies, 'reference_rows': 829},
            {'p05': b, 'reference_rows': 829})


def test_failed_tail_is_not_admitted():
    result = module.frozen_tail_screen(protocol(), *summaries(), protocol_kind='pilot')
    assert result['screen_status'] == 'FAIL'
    assert result['production_privacy_approved'] is False


def test_equal_tail_pass_is_simulation_only():
    result = module.frozen_tail_screen(protocol(), *summaries(.02, .02), protocol_kind='pilot')
    assert result['screen_status'] == 'PASS'
    assert result['production_privacy_approved'] is False


def test_zero_baseline_is_indeterminate():
    assert module.frozen_tail_screen(protocol(), *summaries(.01, 0.), protocol_kind='pilot')['screen_status'] == 'INDETERMINATE'


def test_exact_copy_fails_even_with_large_distance_tail():
    assert module.frozen_tail_screen(protocol(), *summaries(.1, .02, 1), protocol_kind='pilot')['screen_status'] == 'FAIL'


def test_changed_threshold_is_not_silently_accepted():
    changed = protocol()
    changed['privacy_screen']['minimum_synthetic_to_heldout_tail_ratio'] = .5
    with pytest.raises(ValueError, match='silently change'):
        module.frozen_tail_screen(changed, *summaries(), protocol_kind='pilot')


def test_different_fit_reference_sizes_are_rejected():
    synthetic, baseline = summaries()
    baseline['reference_rows'] = 100
    with pytest.raises(ValueError, match='same fit'):
        module.frozen_tail_screen(protocol(), synthetic, baseline, protocol_kind='pilot')


def test_selection_uses_only_training_fold_and_fixed_condition():
    base = {'mechanism': 'marginal_multi', 'magnitude': .5, 'diagnosis_condition': 'exact_slice',
            'repair_condition': 'targeted_repair', 'is_no_drift_control': False}
    rows = [{**base, 'scenario_id': 'outer-a', 'split_role': 'outer_test'},
            {**base, 'scenario_id': 'train-b', 'split_role': 'forecast_train'},
            {**base, 'scenario_id': 'train-a', 'split_role': 'forecast_train'}]
    assert module.select_training_scenario({'scenarios': rows})['scenario_id'] == 'train-a'


def test_selection_fails_when_only_outer_test_is_available():
    with pytest.raises(ValueError, match='No matching'):
        module.select_training_scenario({'scenarios': []})
