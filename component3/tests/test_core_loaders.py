import pytest
import pandas as pd

from component3.src.core.config_loader import (
    ConfigurationError,
    load_config,
    resolve_project_path,
)
from component3.src.core.data_loader import (
    DataLoadingError,
    load_csv,
    load_json_file,
    load_reference_dataset,
    load_current_dataset,
    load_c1_evidence,
    load_c2_diagnosis,
    load_integration_contracts,
)
from component3.src.core.model_loader import (
    ModelLoadingError,
    load_baseline_artifacts,
    load_trusted_pickle,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_configuration_loads(config):
    assert config["component"] == "C3"
    assert config["dataset"]["name"] == "uci_adult"


def test_project_path_resolution():
    path = resolve_project_path(
        "shared/uci_adult/data/reference.csv"
    )
    assert path.is_file()
    assert path.is_absolute()


def test_missing_configuration():
    with pytest.raises(ConfigurationError):
        load_config("component3/configs/not_existing.json")


def test_empty_project_path():
    with pytest.raises(ConfigurationError):
        resolve_project_path("")


def test_reference_dataset(config):
    data = load_reference_dataset(config)
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (26048, 13)
    assert "target" in data.columns


@pytest.mark.parametrize(
    "scenario_id",
    [f"S{i:02d}" for i in range(8)],
)
def test_current_datasets(config, scenario_id):
    data = load_current_dataset(config, scenario_id)
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (6000, 13)


@pytest.mark.parametrize(
    "scenario_id",
    [f"S{i:02d}" for i in range(8)],
)
def test_c1_evidence(config, scenario_id):
    c1 = load_c1_evidence(config, scenario_id)
    assert isinstance(c1, dict)
    assert len(c1) == 13


@pytest.mark.parametrize(
    "scenario_id",
    [f"S{i:02d}" for i in range(8)],
)
def test_c2_diagnosis(config, scenario_id):
    c2 = load_c2_diagnosis(config, scenario_id)
    assert isinstance(c2, dict)
    assert len(c2) == 9


def test_integration_contracts(config):
    c1_contract, c2_contract = load_integration_contracts(config)
    assert isinstance(c1_contract, dict)
    assert isinstance(c2_contract, dict)


def test_unsupported_scenario_id(config):
    with pytest.raises(DataLoadingError):
        load_current_dataset(config, "S99")


def test_missing_csv_dataset():
    with pytest.raises(DataLoadingError):
        load_csv("component3/data/not_existing.csv")


def test_missing_json_evidence():
    with pytest.raises(DataLoadingError):
        load_json_file("component3/data/not_existing.json")


def test_unapproved_model_loading(config):
    with pytest.raises(ModelLoadingError):
        load_baseline_artifacts(config, trusted=False)


def test_trusted_model_loading(config):
    model, preprocessor = load_baseline_artifacts(config, trusted=True)
    assert hasattr(model, "predict")
    assert type(preprocessor).__name__ == "UCIAdultPreprocessor"


def test_missing_trusted_model_file():
    with pytest.raises(ModelLoadingError):
        load_trusted_pickle("shared/uci_adult/models/not_existing.pkl", trusted=True)
