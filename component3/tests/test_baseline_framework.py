"""Unit tests for Component 3 Baseline Model Framework and Adapter."""

import numpy as np
import pytest

from component3.src.baselines.baseline_trainer import (
    HAS_LIGHTGBM,
    BaselineModelTrainer,
    ModelFamily,
)
from component3.src.baselines.model_adapter import BaselineModelAdapter


@pytest.fixture
def synthetic_classification_data():
    rng = np.random.default_rng(42)
    n = 150
    X = rng.normal(size=(n, 8)).astype(np.float32)
    # Binary classification target with some separation
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)

    # 100 train, 25 val, 25 test
    return (
        X[:100], y[:100],
        X[100:125], y[100:125],
        X[125:], y[125:],
    )


def test_baseline_trainer_xgboost(synthetic_classification_data, tmp_path):
    X_train, y_train, X_val, y_val, X_test, y_test = synthetic_classification_data

    save_path = tmp_path / "model_xgb.pkl"
    model, record = BaselineModelTrainer.train_and_evaluate(
        dataset_id="test_ds",
        family=ModelFamily.XGBOOST,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        hyperparams={"n_estimators": 10, "max_depth": 3},
        seed=42,
        model_save_path=save_path,
    )

    assert save_path.is_file()
    assert record.model_family == "xgboost"
    assert 0.0 <= record.test_metrics.accuracy <= 1.0
    assert 0.0 <= record.test_metrics.f1 <= 1.0
    assert 0.0 <= record.test_metrics.roc_auc <= 1.0
    assert "xgboost" in record.dependency_versions


def test_baseline_trainer_random_forest(synthetic_classification_data):
    X_train, y_train, X_val, y_val, X_test, y_test = synthetic_classification_data

    model, record = BaselineModelTrainer.train_and_evaluate(
        dataset_id="test_ds",
        family=ModelFamily.RANDOM_FOREST,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        hyperparams={"n_estimators": 10, "max_depth": 3},
        seed=42,
    )

    assert record.model_family == "random_forest"
    assert record.test_metrics.support == 25
    assert len(record.test_metrics.confusion_matrix) == 2


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
def test_baseline_trainer_lightgbm(synthetic_classification_data):
    X_train, y_train, X_val, y_val, X_test, y_test = synthetic_classification_data

    model, record = BaselineModelTrainer.train_and_evaluate(
        dataset_id="test_ds",
        family=ModelFamily.LIGHTGBM,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        hyperparams={"n_estimators": 10, "max_depth": 3},
        seed=42,
    )

    assert record.model_family == "lightgbm"
    assert record.test_metrics.support == 25


def test_model_adapter_existing_uci_model():
    model_path = "shared/uci_adult/models/base_model_adult.pkl"
    adapter = BaselineModelAdapter.load(model_path, model_family="xgboost")

    # UCI Adult model takes 12 features
    dummy_X = np.zeros((10, 12), dtype=np.float32)
    preds = adapter.predict(dummy_X)
    probs = adapter.predict_proba(dummy_X)

    assert len(preds) == 10
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_model_adapter_evaluate_slice():
    model_path = "shared/uci_adult/models/base_model_adult.pkl"
    adapter = BaselineModelAdapter.load(model_path, model_family="xgboost")

    dummy_X = np.random.randn(20, 12).astype(np.float32)
    dummy_y = np.random.choice([0, 1], size=20)
    slice_mask = np.array([True] * 5 + [False] * 15)

    slice_m, non_aff_m = adapter.evaluate_slice(dummy_X, dummy_y, slice_mask)
    assert slice_m.support == 5
    assert non_aff_m.support == 15
    assert 0.0 <= slice_m.accuracy <= 1.0
    assert 0.0 <= non_aff_m.accuracy <= 1.0


def test_baseline_adapter_legacy_preprocessor_tuple_handling():
    """Regression test: adapter must handle legacy preprocessors returning (df, y) tuples."""
    import pandas as pd
    model_path = "shared/uci_adult/models/base_model_adult.pkl"
    prep_path = "shared/uci_adult/models/preprocessor.pkl"
    adapter = BaselineModelAdapter.load(model_path, model_family="xgboost", preprocessor_path=prep_path)

    # Reference data DataFrame input
    df = pd.read_csv("shared/uci_adult/data/reference.csv", nrows=10)
    preds = adapter.predict(df)
    probs = adapter.predict_proba(df)

    assert len(preds) == 10
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

