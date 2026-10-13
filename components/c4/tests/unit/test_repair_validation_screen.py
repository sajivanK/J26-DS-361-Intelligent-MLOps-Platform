"""Draft screening behavior must never authorize approved-package training."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / 'scripts' / '_repair_validation.py'
spec = importlib.util.spec_from_file_location('c4_draft_repair_validation', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evaluate(synthetic=.1, baseline=.05, copies=0):
    return module.assess_privacy_screen({'p05': synthetic, 'exact_feature_match_rows': copies},
                                       {'p05': baseline})


def test_copy_causes_screen_failure():
    assert evaluate(copies=1)['screen_status'] == 'FAIL'


def test_lower_synthetic_tail_causes_screen_failure():
    result = evaluate(.02, .04)
    assert result['screen_status'] == 'FAIL'
    assert result['tail_distance_ratio'] == .5


def test_equal_tail_is_boundary_pass():
    assert evaluate(.04, .04)['screen_status'] == 'PASS'


def test_zero_baseline_is_indeterminate():
    result = evaluate(.02, 0.)
    assert result['screen_status'] == 'INDETERMINATE'
    assert result['tail_distance_ratio'] is None


def test_screen_pass_never_grants_c3_approval():
    result = evaluate()
    assert result['screen_status'] == 'PASS'
    assert result['privacy_status'] == 'NOT_EVALUATED'
    assert result['repair_package_approved'] is False
    assert result['candidate_training_authorized'] is False


def test_nonfinite_or_out_of_range_distance_is_rejected():
    for value in (float('nan'), float('inf'), -.1, 1.1):
        with pytest.raises(ValueError):
            evaluate(synthetic=value)


def test_negative_exact_match_count_is_rejected():
    with pytest.raises(ValueError):
        evaluate(copies=-1)
