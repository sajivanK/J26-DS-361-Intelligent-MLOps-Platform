"""Regression tests for C3 input schemas and safe slice evaluation."""

from copy import deepcopy
from math import inf, nan
import pandas as pd
import pytest

from component3.src.contracts.input_validator import load_contracts
from component3.src.contracts.schema_validator import (
    SchemaValidationError,
    validate_c1_schema,
    validate_c2_schema,
)
from component3.src.contracts.safe_slice_evaluator import (
    UnsafeSliceRuleError,
    evaluate_slice_rule,
)
from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import (
    load_c1_evidence,
    load_c2_diagnosis,
    load_reference_dataset,
)


@pytest.fixture(scope="module")
def resources():
    config = load_config()
    c1_contract, c2_contract = load_contracts()
    return {
        "config": config,
        "c1_contract": c1_contract,
        "c2_contract": c2_contract,
        "reference": load_reference_dataset(config),
    }


@pytest.mark.parametrize("scenario_number", range(8))
def test_valid_schema_pairs(resources, scenario_number):
    scenario = f"S{scenario_number:02d}"
    config = resources["config"]
    validate_c1_schema(
        load_c1_evidence(config, scenario),
        resources["c1_contract"],
    )
    validate_c2_schema(
        load_c2_diagnosis(config, scenario),
        resources["c2_contract"],
        resources["reference"],
    )


@pytest.mark.parametrize(
    "field,value,remove",
    [
        ("model_id", None, True),
        ("predicted_harm", "high", False),
        ("predicted_harm", nan, False),
        ("predicted_harm", inf, False),
        ("predicted_harm", True, False),
        ("harm_interval", [0.8, 0.2], False),
        ("detector_scores", {}, False),
        ("confidence_stats", {"mean_confidence": "bad"}, False),
        ("detectors_fired", True, False),
    ],
)
def test_invalid_c1_payloads(resources, field, value, remove):
    c1 = deepcopy(load_c1_evidence(resources["config"], "S05"))
    if remove:
        c1.pop(field, None)
    else:
        c1[field] = value
    with pytest.raises(SchemaValidationError):
        validate_c1_schema(c1, resources["c1_contract"])


@pytest.mark.parametrize(
    "field,value,remove",
    [
        ("temporal_window", None, True),
        ("root_cause_category", "invalid", False),
        ("slice_support", nan, False),
        ("slice_support", inf, False),
        ("diagnosis_stability", -0.1, False),
        ("temporal_window", True, False),
        ("affected_slice_rule", "__import__('os').system('echo unsafe')", False),
        ("affected_slice_rule", "unknown_column > 10", False),
    ],
)
def test_invalid_c2_payloads(resources, field, value, remove):
    c2 = deepcopy(load_c2_diagnosis(resources["config"], "S05"))
    if remove:
        c2.pop(field, None)
    else:
        c2[field] = value
    with pytest.raises(SchemaValidationError):
        validate_c2_schema(c2, resources["c2_contract"], resources["reference"])


@pytest.mark.parametrize(
    "rule",
    [
        "__import__('os').system('echo unsafe')",
        "age.__class__",
        "age[0] == 20",
        "age + 10 > 40",
        "age > 20; print('unsafe')",
        "unknown_column > 10",
        "age > 20 > 10",
        "",
    ],
)
def test_unsafe_slice_rules(rule):
    df = pd.DataFrame({"age": [20, 35, 50, 65], "workclass": [1, 2, 3, 2]})
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df, rule)


@pytest.mark.parametrize(
    "rule,expected",
    [
        ("age >= 35", [False, True, True, True]),
        ("age < 50", [True, True, False, False]),
        ("age == 35 or age == 65", [False, True, False, True]),
        ("not (age < 35)", [False, True, True, True]),
        ("workclass in [1, 3]", [True, False, True, False]),
        ("workclass not in [1, 3]", [False, True, False, True]),
    ],
)
def test_valid_slice_operators(rule, expected):
    df = pd.DataFrame({"age": [20, 35, 50, 65], "workclass": [1, 2, 3, 2]})
    mask = evaluate_slice_rule(df, rule)
    assert mask.tolist() == expected


def test_duplicate_columns():
    df = pd.DataFrame([[20, 30], [40, 50]], columns=["age", "age"])
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df, "age > 25")


def test_empty_dataset():
    df = pd.DataFrame(columns=["age"])
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df, "age > 25")


def test_excessively_long_rule():
    df = pd.DataFrame({"age": [20, 35, 50]})
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df, "age > 20 " + " " * 2100)


def test_oversized_membership_list():
    df = pd.DataFrame({"age": [20, 35, 50]})
    rule = "age in [" + ",".join(["20"] * 101) + "]"
    with pytest.raises(UnsafeSliceRuleError):
        evaluate_slice_rule(df, rule)


def test_missing_values():
    df = pd.DataFrame({"age": [20.0, None, 50.0]})
    mask = evaluate_slice_rule(df, "age > 25")
    assert mask.tolist() == [False, False, True]
