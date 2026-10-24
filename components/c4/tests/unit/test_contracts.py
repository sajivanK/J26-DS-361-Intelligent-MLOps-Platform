import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from c4.contracts import UpstreamInput


FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "valid_upstream_input.json"
)


def load_valid_payload() -> dict:
    with FIXTURE_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_valid_v11_upstream_input_is_accepted() -> None:
    payload = load_valid_payload()

    validated = UpstreamInput.model_validate(payload)

    assert validated.c1_evidence.model_id == "adult-xgb-v1"
    assert validated.c1_evidence.harm_severity == "high"
    assert validated.c1_evidence.marginal_extent == "single"
    assert validated.c1_evidence.interaction_suspected == 0.85
    assert validated.c2_diagnosis.diagnosis_stability == 0.85
    assert validated.c3_repair_package.repair_package_status == "APPROVED"


def test_unknown_field_is_rejected() -> None:
    payload = load_valid_payload()
    payload["c1_evidence"]["unexpected_field"] = "invalid"

    with pytest.raises(ValidationError):
        UpstreamInput.model_validate(payload)


def test_old_v10_drift_type_is_rejected() -> None:
    payload = load_valid_payload()
    payload["c1_evidence"]["drift_type"] = "combined"

    with pytest.raises(ValidationError):
        UpstreamInput.model_validate(payload)


def test_invalid_harm_interval_is_rejected() -> None:
    payload = load_valid_payload()
    payload["c1_evidence"]["harm_interval"] = [0.20, 0.10]

    with pytest.raises(ValidationError):
        UpstreamInput.model_validate(payload)


def test_negative_interval_lower_bound_is_accepted() -> None:
    payload = load_valid_payload()
    payload["c1_evidence"]["predicted_harm"] = 0.01
    payload["c1_evidence"]["harm_interval"] = [-0.02, 0.04]
    payload["c1_evidence"]["harm_severity"] = "low"

    validated = UpstreamInput.model_validate(payload)

    assert validated.c1_evidence.harm_interval[0] == -0.02


def test_invalid_diagnosis_stability_is_rejected() -> None:
    payload = load_valid_payload()
    payload["c2_diagnosis"]["diagnosis_stability"] = 1.50

    with pytest.raises(ValidationError):
        UpstreamInput.model_validate(payload)


def test_detector_count_mismatch_is_rejected() -> None:
    payload = load_valid_payload()
    payload["c1_evidence"]["detectors_fired"] = 5

    with pytest.raises(ValidationError):
        UpstreamInput.model_validate(payload)


def test_invalid_repair_utility_margin_is_rejected() -> None:
    payload = load_valid_payload()
    payload["c3_repair_package"]["utility_score"] = -1.50

    with pytest.raises(ValidationError):
        UpstreamInput.model_validate(payload)