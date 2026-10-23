"""Unit tests for offline predictive utility evaluation."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from component3.src.validation.utility_evaluator import (
    OfflineUtilityEvaluator,
    evaluate_offline_utility,
)
from component3.src.validation.validation_models import ValidationStatus


@pytest.fixture
def synthetic_and_real_data():
    np.random.seed(42)
    n_train = 300
    n_test = 150
    n_syn = 100

    def make_df(n):
        age = np.random.normal(35, 10, n)
        income = np.random.normal(50000, 15000, n)
        group = np.random.choice([0, 1], size=n)
        # target depends on age and income
        prob = 1.0 / (1.0 + np.exp(-(0.05 * (age - 35) + 0.00005 * (income - 50000))))
        target = (np.random.rand(n) < prob).astype(int)
        return pd.DataFrame({"age": age, "income": income, "group": group, "target": target})

    ref_train = make_df(n_train)
    eval_test = make_df(n_test)
    syn_data = make_df(n_syn)

    return syn_data, ref_train, eval_test


def test_unauthorized_labels_return_not_evaluated(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data

    result = evaluate_offline_utility(
        synthetic_data=syn_data,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        label_authorized=False,
    )
    assert result.status == ValidationStatus.NOT_EVALUATED
    assert any("NOT_EVALUATED" in r for r in result.reasons)
    assert result.metrics["baseline_metrics"] is None


def test_missing_target_returns_not_evaluated(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data
    unlabeled_syn = syn_data.drop(columns=["target"])

    result = evaluate_offline_utility(
        synthetic_data=unlabeled_syn,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        label_authorized=True,
    )
    assert result.status == ValidationStatus.NOT_EVALUATED
    assert any("missing from synthetic" in r for r in result.reasons)


def test_authorized_evaluation_computes_all_metrics(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data

    # Pre-train baseline model
    base_model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
    base_model.fit(ref_train[["age", "income", "group"]], ref_train["target"])

    result = evaluate_offline_utility(
        synthetic_data=syn_data,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        slice_rule="group == 1",
        target_column="target",
        baseline_model=base_model,
        label_authorized=True,
        min_slice_f1_improvement=-0.10,
    )

    assert result.status == ValidationStatus.PASS
    assert result.metrics["utility_score"] is not None
    assert 0.0 <= result.metrics["utility_score"] <= 1.0

    # Verify overall, slice, and non-affected metrics exist
    aug_metrics = result.metrics["augmented_retrained_metrics"]
    assert "overall" in aug_metrics
    assert "affected_slice" in aug_metrics
    assert "non_affected" in aug_metrics
    assert "delta_overall_f1" in result.metrics["deltas"]


def test_single_class_slice_handled_safely(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data
    # Force affected slice in test set to only have target=0
    test_mod = eval_test.copy()
    test_mod.loc[test_mod["group"] == 1, "target"] = 0

    result = evaluate_offline_utility(
        synthetic_data=syn_data,
        reference_train_data=ref_train,
        evaluation_test_data=test_mod,
        slice_rule="group == 1",
        label_authorized=True,
        min_slice_f1_improvement=-1.0,
    )
    assert result.status == ValidationStatus.PASS
    # roc_auc should be None for single-class slice without crashing
    assert result.metrics["augmented_retrained_metrics"]["affected_slice"]["roc_auc"] is None


def test_negative_transfer_fails(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data
    # Corrupt synthetic targets completely (inverted labels) to simulate harmful augmentation
    corrupt_syn = syn_data.copy()
    corrupt_syn["target"] = 1 - corrupt_syn["target"]
    # Repeat corrupt rows to dominate training
    corrupt_heavy = pd.concat([corrupt_syn] * 10, ignore_index=True)

    result = evaluate_offline_utility(
        synthetic_data=corrupt_heavy,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        slice_rule="group == 1",
        label_authorized=True,
        max_non_affected_degradation=0.01,
        min_slice_f1_improvement=0.0,
    )
    assert result.status == ValidationStatus.FAIL
    assert len(result.reasons) > 0


def test_input_immutability(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data
    syn_copy = syn_data.copy()
    ref_copy = ref_train.copy()
    test_copy = eval_test.copy()

    _ = evaluate_offline_utility(
        synthetic_data=syn_data,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        label_authorized=True,
    )

    pd.testing.assert_frame_equal(syn_data, syn_copy)
    pd.testing.assert_frame_equal(ref_train, ref_copy)
    pd.testing.assert_frame_equal(eval_test, test_copy)


def test_evaluator_class_interface(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data
    evaluator = OfflineUtilityEvaluator(
        min_slice_f1_improvement=-0.05,
        max_non_affected_degradation=0.05,
        target_column="target",
    )
    res = evaluator.evaluate(
        synthetic_data=syn_data,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        label_authorized=True,
    )
    assert isinstance(res.to_dict(), dict)


def test_invalid_target_classes_fail(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data
    syn_invalid = syn_data.copy()
    syn_invalid["target"] = 99  # Class not in reference {0, 1}

    result = evaluate_offline_utility(
        synthetic_data=syn_invalid,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        label_authorized=True,
    )
    assert result.status == ValidationStatus.FAIL
    assert any("invalid classes" in r for r in result.reasons)


def test_target_nans_fail(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data
    syn_invalid = syn_data.copy()
    syn_invalid.loc[0, "target"] = np.nan

    result = evaluate_offline_utility(
        synthetic_data=syn_invalid,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        label_authorized=True,
    )
    assert result.status == ValidationStatus.FAIL
    assert any("missing values" in r for r in result.reasons)


def test_object_categorical_columns_handled(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data
    # Add string object column
    ref_train = ref_train.assign(category_str=["cat_a", "cat_b", "cat_a"] * 100)
    syn_data = syn_data.assign(category_str=["cat_a", "cat_b"] * 50)
    eval_test = eval_test.assign(category_str=["cat_a", "cat_b", "cat_a"] * 50)

    result = evaluate_offline_utility(
        synthetic_data=syn_data,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        label_authorized=True,
        min_slice_f1_improvement=-0.10,
    )
    assert result.status == ValidationStatus.PASS


def test_controlled_delta_vs_real_only(synthetic_and_real_data):
    syn_data, ref_train, eval_test = synthetic_and_real_data
    base_model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    base_model.fit(ref_train[["age", "income", "group"]], ref_train["target"])

    result = evaluate_offline_utility(
        synthetic_data=syn_data,
        reference_train_data=ref_train,
        evaluation_test_data=eval_test,
        baseline_model=base_model,
        label_authorized=True,
    )
    assert "deltas" in result.metrics
    assert "deltas_vs_baseline" in result.metrics
    # deltas are computed against real_only_retrained_metrics
    real_f1 = result.metrics["real_only_retrained_metrics"]["overall"]["f1"]
    aug_f1 = result.metrics["augmented_retrained_metrics"]["overall"]["f1"]
    expected_delta = round(aug_f1 - real_f1, 4)
    assert result.metrics["deltas"]["delta_overall_f1"] == expected_delta

