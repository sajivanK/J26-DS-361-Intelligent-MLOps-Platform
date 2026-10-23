"""Automated regression test suite for C1/C2 input contract validation."""

import sys
from copy import deepcopy
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component3.src.contracts.input_validator import (
    InputContractError,
    load_contracts,
    load_json,
    validate_c1_input,
    validate_c2_input,
    validate_cross_component_consistency,
)

C1_CONTRACT, C2_CONTRACT = load_contracts()
REFERENCE = pd.read_csv(ROOT / "shared/uci_adult/data/reference.csv")
SCENARIO_IDS = [f"S{i:02d}" for i in range(8)]


def load_scenario(scenario_id: str):
    """Helper to load C1 evidence, C2 diagnosis, and current batch for a scenario."""
    c1 = load_json(ROOT / f"component3/data/simulated_c1/c1_evidence_{scenario_id}.json")
    c2 = load_json(ROOT / f"component3/data/simulated_c2/c2_diagnosis_{scenario_id}.json")
    current = pd.read_csv(ROOT / f"component3/data/drifted/{scenario_id}.csv")
    return c1, c2, current


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_valid_scenario_contracts(scenario_id):
    """Test that all 8 simulated scenarios pass C1, C2, and consistency checks."""
    c1, c2, current = load_scenario(scenario_id)

    c1_res = validate_c1_input(c1, C1_CONTRACT)
    assert c1_res["valid"] is True
    assert c1_res["field_count"] == 13

    c2_res = validate_c2_input(c2, C2_CONTRACT, REFERENCE)
    assert c2_res["valid"] is True
    assert c2_res["field_count"] == 9

    consistency = validate_cross_component_consistency(c1, c2, current)
    assert consistency["valid"] is True
    assert consistency["errors"] == []
    assert consistency["warning_count"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("predicted_harm", "high"),
        ("detectors_fired", 7),
    ],
)
def test_invalid_c1_values(field, value):
    """Test that invalid values in C1 payload raise InputContractError."""
    c1, _, _ = load_scenario("S01")
    invalid = deepcopy(c1)
    invalid[field] = value
    with pytest.raises(InputContractError):
        validate_c1_input(invalid, C1_CONTRACT)


def test_missing_c1_field():
    """Test that missing required field in C1 payload raises InputContractError."""
    c1, _, _ = load_scenario("S01")
    del c1["model_id"]
    with pytest.raises(InputContractError):
        validate_c1_input(c1, C1_CONTRACT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_cause_category", "unsupported"),
        ("temporal_window", 0),
    ],
)
def test_invalid_c2_values(field, value):
    """Test that invalid values in C2 payload raise InputContractError."""
    _, c2, _ = load_scenario("S01")
    invalid = deepcopy(c2)
    invalid[field] = value
    with pytest.raises(InputContractError):
        validate_c2_input(invalid, C2_CONTRACT, REFERENCE)


def test_missing_c2_field():
    """Test that missing required field in C2 payload raises InputContractError."""
    _, c2, _ = load_scenario("S01")
    del c2["root_cause_feature"]
    with pytest.raises(InputContractError):
        validate_c2_input(c2, C2_CONTRACT, REFERENCE)


@pytest.mark.parametrize(
    "field,value",
    [
        ("slice_support", 0.10),
        ("affected_slice_rule", "nonexistent_feature > 0"),
    ],
)
def test_cross_component_errors(field, value):
    """Test that fatal inconsistencies between C1 and C2 produce valid=False with errors."""
    c1, c2, current = load_scenario("S01")
    invalid = deepcopy(c2)
    invalid[field] = value
    result = validate_cross_component_consistency(c1, invalid, current)
    assert result["valid"] is False
    assert len(result["errors"]) > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_cause_category", "interaction"),
        ("root_cause_feature", "race"),
    ],
)
def test_cross_component_warnings(field, value):
    """Test that non-fatal diagnostic divergences between C1 and C2 produce warnings."""
    c1, c2, current = load_scenario("S01")
    invalid = deepcopy(c2)
    invalid[field] = value
    result = validate_cross_component_consistency(c1, invalid, current)
    assert result["valid"] is True
    assert result["warning_count"] > 0
    assert len(result["warnings"]) > 0
