"""Tests for frozen reference-derived slice rules."""

import pandas as pd
import pytest
from pydantic import ValidationError

from c4.scenarios.slices import SliceRule, reference_quantile_slice


def test_threshold_is_derived_from_reference():
    reference = pd.DataFrame({"age": [0.0, 1.0, 2.0, 3.0]})

    rule = reference_quantile_slice(
        reference,
        name="lower_age_half",
        feature="age",
        quantile=0.5,
    )

    assert rule.value == 1.5


def test_frozen_rule_preserves_membership_indices():
    rule = SliceRule(
        name="lower_age",
        feature="age",
        operator="le",
        value=1.5,
    )
    clean = pd.DataFrame(
        {"age": [1.0, 2.0]},
        index=[101, 205],
    )

    membership = rule.mask(clean)
    perturbed = clean.copy()
    perturbed["age"] += 10.0

    assert membership.to_dict() == {101: True, 205: False}
    assert not rule.mask(perturbed).any()
    # Previously recorded membership remains unchanged.
    assert membership.to_dict() == {101: True, 205: False}


def test_target_cannot_define_a_slice():
    with pytest.raises(ValidationError):
        SliceRule(
            name="invalid",
            feature="target",
            operator="eq",
            value=1.0,
        )


def test_categorical_codes_cannot_use_ordered_comparisons():
    with pytest.raises(ValidationError):
        SliceRule(
            name="invalid",
            feature="marital_status",
            operator="le",
            value=2.0,
        )


def test_missing_slice_values_are_rejected():
    rule = SliceRule(
        name="age_slice",
        feature="age",
        operator="le",
        value=1.5,
    )

    with pytest.raises(ValueError, match="missing or infinite"):
        rule.mask(pd.DataFrame({"age": [float("nan")]}))


def test_slice_rule_survives_json_round_trip():
    rule = SliceRule(
        name="age_slice",
        feature="age",
        operator="le",
        value=1.5,
    )

    restored = SliceRule.model_validate_json(rule.model_dump_json())

    assert restored == rule