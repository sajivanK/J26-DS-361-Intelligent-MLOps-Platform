"""Restricted C2 slice-rule interpretation for processed Adult data."""

from __future__ import annotations

import re

import pandas as pd

from c4.contracts.models import C2Diagnosis
from c4.scenarios.config import CATEGORICAL_FEATURES
from c4.scenarios.slices import SliceRule


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CLAUSE = re.compile(
    rf"([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>|==)\s*({_NUMBER})"
)


def parse_diagnosed_slice(rule_text: str) -> tuple[SliceRule, ...]:
    """Accept numeric <=, > and == clauses joined by AND."""

    if not isinstance(rule_text, str) or not rule_text.strip():
        raise ValueError("C2 slice rule must be a nonempty string.")

    clauses = re.split(
        r"\s+AND\s+",
        rule_text.strip(),
        flags=re.IGNORECASE,
    )
    rules = []
    operators = {"<=": "le", ">": "gt", "==": "eq"}

    for position, clause in enumerate(clauses):
        match = _CLAUSE.fullmatch(clause.strip())

        if match is None:
            raise ValueError(
                "Unsupported C2 slice rule. Use numeric <=, > or == "
                "clauses joined by AND."
            )

        feature, operator, value_text = match.groups()
        value = float(value_text)

        rule = SliceRule(
            name=f"diagnosed_clause_{position}",
            feature=feature,
            operator=operators[operator],
            value=value,
        )

        if feature in CATEGORICAL_FEATURES and not value.is_integer():
            raise ValueError(
                "Processed categorical comparisons require integer codes."
            )

        rules.append(rule)

    return tuple(rules)


def diagnosed_slice_mask(
    clean_frame: pd.DataFrame,
    diagnosis: C2Diagnosis,
    *,
    rule_feature_space: str,
) -> pd.Series:
    """Evaluate C2's rule before perturbation, preserving source IDs.

    Raw-space rules must first be translated using the verified
    reference-fitted preprocessor. This function does not translate them.
    """

    if rule_feature_space != "processed":
        raise ValueError(
            "Only explicitly processed-space C2 rules are supported."
        )

    if clean_frame.empty or not clean_frame.index.is_unique:
        raise ValueError(
            "Clean evaluation rows must be nonempty with unique indices."
        )

    rules = parse_diagnosed_slice(diagnosis.affected_slice_rule)
    mask = pd.Series(True, index=clean_frame.index, dtype=bool)

    for rule in rules:
        mask &= rule.mask(clean_frame)

    mask.name = "diagnosed_slice_member"
    return mask