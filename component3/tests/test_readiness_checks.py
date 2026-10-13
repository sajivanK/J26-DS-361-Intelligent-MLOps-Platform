"""Unit tests for independent Component 3 readiness checks."""

import pytest

from component3.src.readiness.readiness_checks import (
    check_data_sufficiency,
    check_slice_sufficiency,
)


@pytest.mark.parametrize(
    "reference_rows,current_rows,expected_reference,expected_current",
    [
        (26048, 6000, True, True),
        (500, 200, True, True),
        (499, 200, False, True),
        (500, 199, True, False),
        (499, 199, False, False),
        (0, 0, False, False),
    ],
)
def test_data_sufficiency(
    reference_rows,
    current_rows,
    expected_reference,
    expected_current,
):
    result = check_data_sufficiency(
        reference_rows=reference_rows,
        current_rows=current_rows,
        min_reference_rows=500,
        min_current_rows=200,
    )
    assert result["reference_ok"] is expected_reference
    assert result["current_ok"] is expected_current
    assert result["passed"] is (
        expected_reference and expected_current
    )


@pytest.mark.parametrize(
    "affected_rows,expected_passed",
    [
        (342, True),
        (50, True),
        (49, False),
        (0, False),
        (6000, True),
    ],
)
def test_slice_sufficiency(
    affected_rows,
    expected_passed,
):
    result = check_slice_sufficiency(
        affected_slice_rows=affected_rows,
        min_affected_slice_rows=50,
    )
    assert result["passed"] is expected_passed
    assert result["affected_slice_rows"] == affected_rows
    assert result["min_affected_slice_rows"] == 50


from component3.src.readiness.readiness_checks import (
    check_diagnosis_stability,
)


@pytest.mark.parametrize(
    "stability,expected_passed",
    [
        (0.88, True),
        (0.60, True),
        (0.59, False),
        (0.0, False),
        (1.0, True),
        (None, False),
        (float("nan"), False),
        (float("inf"), False),
        (float("-inf"), False),
        (True, False),
        ("0.88", False),
    ],
)
def test_diagnosis_stability(
    stability,
    expected_passed,
):
    """Diagnosis stability must meet the minimum threshold."""
    result = check_diagnosis_stability(
        diagnosis_stability=stability,
        min_diagnosis_stability=0.60,
    )
    assert result["passed"] is expected_passed
    assert result["min_diagnosis_stability"] == 0.60


from component3.src.readiness.readiness_checks import (
    check_label_trust,
)


@pytest.mark.parametrize(
    "case_name,provenance,require_trusted,allow_unlabeled,expected_generation,expected_retraining",
    [
        (
            "missing_provenance",
            None,
            True,
            True,
            True,
            False,
        ),
        (
            "labels_unavailable",
            {"labels_available": False},
            True,
            True,
            True,
            False,
        ),
        (
            "unverified_source",
            {
                "labels_available": True,
                "trusted": True,
                "source_verified": False,
            },
            True,
            True,
            True,
            False,
        ),
        (
            "untrusted_flag",
            {
                "labels_available": True,
                "trusted": False,
                "source_verified": True,
            },
            True,
            True,
            True,
            False,
        ),
        (
            "both_untrusted",
            {
                "labels_available": True,
                "trusted": False,
                "source_verified": False,
            },
            True,
            True,
            True,
            False,
        ),
        (
            "verified_trusted",
            {
                "labels_available": True,
                "trusted": True,
                "source_verified": True,
            },
            True,
            True,
            True,
            True,
        ),
        (
            "unverified_source_relaxed_config",
            {
                "labels_available": True,
                "trusted": True,
                "source_verified": False,
            },
            False,
            True,
            True,
            False,
        ),
        (
            "untrusted_flag_relaxed_config",
            {
                "labels_available": True,
                "trusted": False,
                "source_verified": True,
            },
            False,
            True,
            True,
            False,
        ),
        (
            "verified_trusted_relaxed_config",
            {
                "labels_available": True,
                "trusted": True,
                "source_verified": True,
            },
            False,
            True,
            True,
            True,
        ),
        (
            "missing_provenance_unlabeled_disallowed",
            None,
            True,
            False,
            False,
            False,
        ),
        (
            "labels_unavailable_unlabeled_disallowed",
            {"labels_available": False},
            True,
            False,
            False,
            False,
        ),
        (
            "verified_trusted_unlabeled_disallowed",
            {
                "labels_available": True,
                "trusted": True,
                "source_verified": True,
            },
            True,
            False,
            True,
            True,
        ),
    ],
)
def test_label_trust(
    case_name,
    provenance,
    require_trusted,
    allow_unlabeled,
    expected_generation,
    expected_retraining,
):
    """Assess label eligibility without treating configuration as proof."""
    result = check_label_trust(
        label_provenance=provenance,
        require_trusted_labels=require_trusted,
        allow_unlabeled_current_data=allow_unlabeled,
    )
    assert result["generation_allowed"] is expected_generation, case_name
    assert result["supervised_retraining_ready"] is expected_retraining, case_name
    assert result["passed"] is expected_generation, case_name


@pytest.mark.parametrize(
    "reference_rows,current_rows,min_reference,min_current",
    [
        ("500", 6000, 500, 200),
        (26048, None, 500, 200),
        (True, 6000, 1, 200),
        (-10, 6000, 500, 200),
        (26048, 6000, 0, 200),
        (26048, 6000, 500, "200"),
    ],
)
def test_data_sufficiency_rejects_invalid_inputs(
    reference_rows,
    current_rows,
    min_reference,
    min_current,
):
    result = check_data_sufficiency(
        reference_rows,
        current_rows,
        min_reference,
        min_current,
    )
    assert result["passed"] is False


@pytest.mark.parametrize(
    "affected_rows,minimum_rows",
    [
        ("100", 50),
        (-10, 50),
        (True, 1),
        (100, 0),
        (100, "50"),
        (None, 50),
    ],
)
def test_slice_sufficiency_rejects_invalid_inputs(
    affected_rows,
    minimum_rows,
):
    result = check_slice_sufficiency(
        affected_rows,
        minimum_rows,
    )
    assert result["passed"] is False


@pytest.mark.parametrize(
    "stability,min_stability",
    [
        (1.5, 0.60),
        (-0.5, 0.60),
        (0.88, 1.50),
        (0.88, -0.10),
        (0.88, float("nan")),
        (0.88, float("inf")),
        (0.88, float("-inf")),
        (0.88, True),
        (0.88, "0.60"),
    ],
)
def test_diagnosis_stability_rejects_invalid_inputs(
    stability,
    min_stability,
):
    result = check_diagnosis_stability(
        stability,
        min_stability,
    )
    assert result["passed"] is False



