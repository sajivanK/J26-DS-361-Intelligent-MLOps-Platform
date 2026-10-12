import json
from pathlib import Path

import pandas as pd
import pytest

from c4.contracts.models import C2Diagnosis
from c4.scenarios.diagnosis import (
    diagnosed_slice_mask,
    parse_diagnosed_slice,
)


def diagnosis_for(rule):
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures" / "valid_upstream_input.json"
    )
    fields = json.loads(
        fixture.read_text(encoding="utf-8")
    )["c2_diagnosis"]
    fields["affected_slice_rule"] = rule
    return C2Diagnosis.model_validate(fields)


def test_numeric_rule_preserves_source_indices():
    frame = pd.DataFrame(
        {"age": [-1.0, 0.0, 1.0]},
        index=[10, 20, 30],
    )
    mask = diagnosed_slice_mask(
        frame,
        diagnosis_for("age <= 0.0"),
        rule_feature_space="processed",
    )

    assert mask.index.tolist() == [10, 20, 30]
    assert mask.tolist() == [True, True, False]

    # Membership was determined on clean data.
    frame.loc[10, "age"] = 99.0
    assert mask.loc[10]


def test_and_rule_uses_all_clauses():
    frame = pd.DataFrame({
        "education_num": [0.8, 0.8, 0.2],
        "occupation": [3, 2, 3],
    })
    mask = diagnosed_slice_mask(
        frame,
        diagnosis_for("education_num > 0.5 AND occupation == 3"),
        rule_feature_space="processed",
    )

    assert mask.tolist() == [True, False, False]


@pytest.mark.parametrize(
    "rule",
    [
        "occupation == 'Tech-support'",
        "age > 0 OR age <= 0",
        "target == 1",
        "occupation > 3",
    ],
)
def test_unsupported_or_invalid_rule_is_rejected(rule):
    with pytest.raises(ValueError):
        parse_diagnosed_slice(rule)


def test_raw_feature_space_is_rejected():
    frame = pd.DataFrame({"age": [20.0, 40.0]})

    with pytest.raises(ValueError, match="processed-space"):
        diagnosed_slice_mask(
            frame,
            diagnosis_for("age <= 30"),
            rule_feature_space="raw",
        )


def test_nonfinite_slice_values_are_rejected():
    frame = pd.DataFrame({"age": [0.0, float("nan")]})

    with pytest.raises(ValueError, match="missing or infinite"):
        diagnosed_slice_mask(
            frame,
            diagnosis_for("age <= 0"),
            rule_feature_space="processed",
        )