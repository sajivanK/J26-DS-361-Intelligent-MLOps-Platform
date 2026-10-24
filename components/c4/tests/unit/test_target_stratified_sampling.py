"""Fit-only quotas and auditable selection from a fixed synthetic pool."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[2] / 'scripts' / '_target_stratified_sampling.py'
spec = importlib.util.spec_from_file_location('c4_target_sampling', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_actual_fit_counts_produce_423_and_77_quotas():
    assert module.binary_quotas([0] * 701 + [1] * 128, 500) == {0: 423, 1: 77}


def test_largest_remainder_tie_is_deterministic():
    assert module.binary_quotas([0, 1], 3) == {0: 2, 1: 1}


def test_missing_fit_class_is_rejected():
    with pytest.raises(ValueError, match='both binary'):
        module.binary_quotas([0, 0], 500)


def test_quota_rounded_to_zero_is_rejected():
    with pytest.raises(ValueError, match='zero'):
        module.binary_quotas([0] * 999 + [1], 2)


def test_selection_preserves_records_and_quota_counts():
    pool = pd.DataFrame({'age': np.arange(8), 'target': [0, 1, 0, 1, 0, 0, 0, 1]})
    selected, positions = module.select_pool_rows(pool, {0: 3, 1: 2}, 42)
    pd.testing.assert_frame_equal(selected, pool.iloc[positions].reset_index(drop=True))
    assert selected['target'].value_counts().to_dict() == {0: 3, 1: 2}
    assert len(set(positions)) == 5


def test_selection_repeats_with_same_seed():
    pool = pd.DataFrame({'target': [0, 1] * 10})
    _, first = module.select_pool_rows(pool, {0: 5, 1: 5}, 42)
    _, second = module.select_pool_rows(pool, {0: 5, 1: 5}, 42)
    np.testing.assert_array_equal(first, second)


def test_insufficient_pool_fails_instead_of_extending_budget():
    with pytest.raises(ValueError, match='lacks enough'):
        module.select_pool_rows(pd.DataFrame({'target': [0, 0, 1]}), {0: 2, 1: 2}, 42)


def test_invalid_pool_label_is_rejected():
    with pytest.raises(ValueError, match='binary'):
        module.select_pool_rows(pd.DataFrame({'target': [0, 2]}), {0: 1, 1: 1}, 42)
