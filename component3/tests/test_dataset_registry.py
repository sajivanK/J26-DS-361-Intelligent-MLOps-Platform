"""Unit tests for Component 3 Dataset Registry."""

import pytest
from component3.src.datasets.dataset_registry import (
    DatasetMetadata,
    DatasetRegistry,
    DatasetStatus,
    SplitStrategy,
    get_dataset_registry,
)


def test_registry_contains_default_datasets():
    reg = get_dataset_registry()
    datasets = reg.list_datasets()
    assert "uci_adult" in datasets
    assert "telco_churn" in datasets
    assert "home_credit" in datasets
    assert "ieee_cis" in datasets
    assert "bank_marketing" in datasets
    assert len(datasets) == 5


def test_uci_adult_metadata():
    reg = get_dataset_registry()
    uci = reg.get("uci_adult")
    assert uci.status == DatasetStatus.VERIFIED
    assert uci.target_column == "target"
    assert len(uci.numerical_features) == 5
    assert len(uci.categorical_features) == 7
    assert len(uci.feature_columns) == 12
    assert uci.split_strategy == SplitStrategy.STRATIFIED
    assert uci.is_local_available is True


def test_pending_datasets_metadata():
    reg = DatasetRegistry()
    telco = reg.get("telco_churn")
    assert telco.status == DatasetStatus.PENDING_DOWNLOAD
    assert telco.target_column == "Churn"
    assert "customerID" in telco.id_columns
    assert "customerID" not in telco.feature_columns

    bank = reg.get("bank_marketing")
    assert bank.status == DatasetStatus.PENDING_DOWNLOAD
    assert "duration" in bank.drop_columns
    assert "duration" not in bank.feature_columns

    home = reg.get("home_credit")
    assert home.status == DatasetStatus.PENDING_DOWNLOAD
    assert home.target_column == "TARGET"


def test_load_dataset_missing_raises_informative_error():
    reg = get_dataset_registry()
    with pytest.raises(FileNotFoundError) as exc_info:
        reg.load_dataset("home_credit")
    assert "PENDING_DOWNLOAD" in str(exc_info.value)
    assert "Colab" in str(exc_info.value)


def test_load_uci_adult_compatibility():
    reg = get_dataset_registry()
    df = reg.load_dataset("uci_adult", nrows=50)
    assert len(df) == 50
    assert "target" in df.columns
    assert "age" in df.columns


def test_register_custom_dataset():
    reg = DatasetRegistry()
    custom_meta = DatasetMetadata(
        dataset_id="custom_test_ds",
        display_name="Custom Test Dataset",
        version="0.1.0",
        source="internal",
        license_info="MIT",
        status=DatasetStatus.VERIFIED,
        target_column="label",
        target_mapping={0: 0, 1: 1},
        numerical_features=["f1", "f2"],
        categorical_features=["cat1"],
        id_columns=["id"],
        drop_columns=["id"],
    )
    reg.register(custom_meta)
    assert "custom_test_ds" in reg.list_datasets()
    retrieved = reg.get("custom_test_ds")
    assert retrieved.feature_columns == ["f1", "f2", "cat1"]


def test_unknown_dataset_raises_key_error():
    reg = get_dataset_registry()
    with pytest.raises(KeyError) as exc_info:
        reg.get("non_existent_dataset_123")
    assert "non_existent_dataset_123" in str(exc_info.value)
