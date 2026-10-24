"""Unit tests for Component 3 Tabular Data Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from component3.src.preprocessing.tabular_preprocessor import (
    TabularDataPreprocessor,
)


@pytest.fixture
def sample_tabular_data():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "age": [25.0, 35.0, np.nan, 45.0, 50.0, 60.0],
        "hours": ["40", "50", "?", "45", "35", "60"],
        "education": ["Bachelors", "Masters", "Bachelors", "Doctorate", "Masters", "Bachelors"],
        "city": ["NY", "SF", "?", "NY", "SF", "LA"],
        "leakage_col": [100, 200, 300, 400, 500, 600],
        "target": ["<=50K", ">50K", "<=50K", ">50K", "<=50K", ">50K"],
    })


def test_fit_transform_three_representations(sample_tabular_data):
    prep = TabularDataPreprocessor(
        numerical_features=["age", "hours"],
        categorical_features=["education", "city"],
        target_column="target",
        target_mapping={"<=50K": 0, ">50K": 1},
        id_columns=["id"],
        leakage_prone_columns=["leakage_col"],
        missing_representations=["?"],
    )

    bundle = prep.fit_transform(sample_tabular_data)

    # Representation 1: Raw DataFrame intact
    assert bundle.raw_df.shape == (6, 7)
    assert "target" in bundle.raw_df.columns
    assert "id" in bundle.raw_df.columns

    # Representation 2: Generator DataFrame
    # Target and id/leakage columns excluded, discrete columns preserved, no one-hot explosion
    assert bundle.generator_df.shape == (6, 4)
    assert "target" not in bundle.generator_df.columns
    assert "id" not in bundle.generator_df.columns
    assert "leakage_col" not in bundle.generator_df.columns
    assert bundle.generator_discrete_columns == ["education", "city"]
    # Check that missing values in generator frame were imputed
    assert not bundle.generator_df.isna().any().any()

    # Representation 3: Model Transformed Feature Matrix
    assert bundle.X.shape == (6, 4)
    assert bundle.y is not None
    assert bundle.y.shape == (6,)
    assert set(bundle.y) == {0, 1}
    assert bundle.feature_names == ["age", "hours", "education", "city"]


def test_handle_unknown_categories(sample_tabular_data):
    prep = TabularDataPreprocessor(
        numerical_features=["age"],
        categorical_features=["education"],
        target_column="target",
        target_mapping={"<=50K": 0, ">50K": 1},
    )
    prep.fit(sample_tabular_data)

    # Test data with unseen category 'PostDoc'
    test_df = pd.DataFrame({
        "age": [30.0],
        "education": ["PostDoc"],
        "target": ["<=50K"],
    })

    bundle = prep.transform(test_df)
    # PostDoc should be encoded safely as -1 without raising ValueError
    assert bundle.X.shape == (1, 2)
    cat_feature_idx = bundle.feature_names.index("education")
    assert bundle.X[0, cat_feature_idx] == -1.0


def test_feature_ordering_preservation(sample_tabular_data):
    prep = TabularDataPreprocessor(
        numerical_features=["age", "hours"],
        categorical_features=["education", "city"],
        target_column="target",
    )
    prep.fit(sample_tabular_data)

    # Test data with shuffled column order
    shuffled_test = sample_tabular_data[["city", "age", "target", "education", "hours"]].copy()
    bundle = prep.transform(shuffled_test)

    # Column ordering must remain strict: age, hours, education, city
    assert bundle.feature_names == ["age", "hours", "education", "city"]
    assert bundle.X.shape == (6, 4)


def test_transform_synthetic_records(sample_tabular_data):
    prep = TabularDataPreprocessor(
        numerical_features=["age", "hours"],
        categorical_features=["education", "city"],
        target_column="target",
    )
    prep.fit(sample_tabular_data)

    # Synthetic data from generator (no target column)
    synthetic_df = pd.DataFrame({
        "age": [28.0, 42.0],
        "hours": [38.0, 48.0],
        "education": ["Masters", "Bachelors"],
        "city": ["NY", "SF"],
    })

    X_syn = prep.transform_synthetic(synthetic_df)
    assert X_syn.shape == (2, 4)
    assert isinstance(X_syn, np.ndarray)


def test_preprocessor_serialization(sample_tabular_data, tmp_path):
    prep = TabularDataPreprocessor(
        numerical_features=["age", "hours"],
        categorical_features=["education", "city"],
        target_column="target",
        target_mapping={"<=50K": 0, ">50K": 1},
    )
    bundle_orig = prep.fit_transform(sample_tabular_data)

    model_path = tmp_path / "fitted_preprocessor.joblib"
    prep.save(model_path)
    assert model_path.is_file()

    loaded_prep = TabularDataPreprocessor.load(model_path)
    assert loaded_prep.is_fitted is True

    bundle_loaded = loaded_prep.transform(sample_tabular_data)
    np.testing.assert_allclose(bundle_orig.X, bundle_loaded.X, rtol=1e-5)


def test_class_weights_computation(sample_tabular_data):
    prep = TabularDataPreprocessor(
        numerical_features=["age"],
        categorical_features=["education"],
        target_column="target",
        target_mapping={"<=50K": 0, ">50K": 1},
    )
    prep.fit(sample_tabular_data)
    weights = prep.get_class_weights()
    assert 0 in weights and 1 in weights
    # Since 3 rows are class 0 and 3 rows are class 1, balanced weights should be 1.0 each
    assert np.isclose(weights[0], 1.0)
    assert np.isclose(weights[1], 1.0)


def test_real_uci_adult_end_to_end_smoke(tmp_path):
    """Audit 2 End-to-End Real Data Smoke Test."""
    from component3.src.datasets.dataset_registry import DatasetRegistry
    from component3.src.splitting.splitter import DataSplitter, assert_no_split_leakage
    from component3.src.generators.tvae_generator import TVAEGenerator
    from component3.src.baselines.model_adapter import BaselineModelAdapter

    reg = DatasetRegistry()
    meta = reg.get("uci_adult")
    df = reg.load_dataset("uci_adult", nrows=100)

    # 1. Leak-free split
    splits, manifest = DataSplitter.split(df, metadata=meta, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert_no_split_leakage(manifest)

    # 2. Fit preprocessor on train only
    prep = TabularDataPreprocessor(
        numerical_features=meta.numerical_features,
        categorical_features=meta.categorical_features,
        target_column=meta.target_column,
        target_mapping=meta.target_mapping,
        drop_columns=meta.drop_columns,
    )
    train_bundle = prep.fit_transform(splits.train)
    val_bundle = prep.transform(splits.val)

    # 3. Serialization parity
    save_file = tmp_path / "adult_prep.joblib"
    prep.save(save_file)
    reloaded = TabularDataPreprocessor.load(save_file)
    reloaded_val = reloaded.transform(splits.val)
    np.testing.assert_allclose(val_bundle.X, reloaded_val.X, rtol=1e-5)

    # 4. Generate small synthetic sample via TVAE (1 epoch)
    gen = TVAEGenerator(epochs=1)
    gen.fit(train_bundle.generator_df)
    syn_df = gen.sample(5)

    # 5. Transform synthetic records through preprocessor
    X_syn = prep.transform_synthetic(syn_df)
    assert X_syn.shape == (5, len(meta.feature_columns))

    # 6. Evaluate on baseline adapter
    adapter = BaselineModelAdapter.load("shared/uci_adult/models/base_model_adult.pkl")
    preds = adapter.predict(X_syn)
    assert len(preds) == 5

