"""Mock engineering integration tests, not research experiments.

Repair rows and upstream scores are test fixtures. No CTGAN/TVAE is
trained here, and fixture privacy/quality values are not measurements.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from c4.contracts import UpstreamInput
from c4.data import FEATURE_COLUMNS
from c4.data.split_manifest import (
    load_adult_split_manifest,
    save_adult_split_manifest,
)
from c4.models import (
    ChampionConfig,
    load_candidate,
    load_champion,
    save_pre_release_evidence,
    train_and_save_candidate,
    train_and_save_champion,
)
from c4.risk.hard_gates import run_hard_gates
from c4.scenarios.adapter import transform_role
from c4.scenarios.config import ScenarioConfig
from c4.scenarios.slices import SliceRule


def run_test_workflow(
    *,
    root,
    upstream,
    clean_release,
    evaluated_release,
    manifest_directory,
    schema_compatible,
    rollback_ready,
    training_function=train_and_save_candidate,
):
    """Test-only orchestration of currently implemented C4 functions."""

    passed, gates = run_hard_gates(
        upstream,
        schema_compatible=schema_compatible,
        rollback_ready=rollback_ready,
    )

    if not passed:
        return {
            "decision": "ESCALATE",
            "gates": gates,
            "candidate": None,
            "evidence": None,
        }

    candidate = training_function(
        upstream.c3_repair_package,
        artifact_version="mock-candidate-001",
        champion_version="v1",
        data_root=root,
    )

    evidence = save_pre_release_evidence(
        champion_directory=(
            root / "c4_artifacts" / "champion" / "adult" / "v1"
        ),
        candidate_directory=candidate.model_path.parent,
        split_manifest_directory=manifest_directory,
        clean_release_validation=clean_release,
        release_validation=evaluated_release,
        diagnosis=upstream.c2_diagnosis,
        rule_feature_space="processed",
        scenario_id="mock-integration-001",
        data_root=root,
    )

    return {
        # Passing pre-release checks does not authorize deployment.
        # N1-N4 and the final statistical controller are still pending.
        "decision": None,
        "gates": gates,
        "candidate": candidate,
        "evidence": evidence,
    }


@pytest.fixture
def workflow_inputs(tmp_path):
    data_directory = tmp_path / "uci_adult_data"
    data_directory.mkdir()

    reference = pd.DataFrame({
        feature: np.arange(40, dtype=float)
        for feature in FEATURE_COLUMNS
    })
    reference["target"] = np.arange(40) % 2
    reference.to_csv(data_directory / "reference.csv", index=False)

    test = pd.DataFrame({
        feature: np.arange(100, 180, dtype=float)
        for feature in FEATURE_COLUMNS
    })
    test["age"] = np.arange(80, dtype=float)
    test["target"] = np.arange(80) % 2
    test.to_csv(data_directory / "test.csv", index=False)

    champion = train_and_save_champion(
        data_root=tmp_path,
        config=ChampionConfig(
            n_estimators=3,
            max_depth=2,
            n_jobs=1,
        ),
    )

    manifest_artifact = save_adult_split_manifest(data_root=tmp_path)
    manifest_directory = manifest_artifact.manifest_path.parent
    manifest = load_adult_split_manifest(
        manifest_directory,
        data_root=tmp_path,
    )

    release_ids = manifest["partitions"][
        "release_validation"
    ]["source_row_ids"]
    repair_ids = manifest["partitions"]["repair"]["source_row_ids"]

    clean_release = test.loc[release_ids].copy()

    # A small real-row fixture substitutes for a repair CSV in this test.
    # It is not claimed to be synthetic data.
    mock_repair = test.loc[repair_ids[:8]].copy()
    mock_repair.to_csv(tmp_path / "mock_repair.csv", index=False)

    evaluated_release = transform_role(
        clean_release,
        config=ScenarioConfig(
            mechanism="marginal_multi",
            features=["education_num", "hours_per_week"],
            magnitude=0.5,
            seed=42,
        ),
        true_slice=SliceRule(
            name="mock_true_slice",
            feature="age",
            operator="le",
            value=39.0,
        ),
        role="release_validation",
    ).frame

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures" / "valid_upstream_input.json"
    )
    fields = json.loads(
        fixture_path.read_text(encoding="utf-8")
    )

    fields["c1_evidence"]["engine_version"] = "mock-integration"
    fields["c2_diagnosis"]["affected_slice_rule"] = (
        f"age <= {float(clean_release['age'].median())}"
    )
    fields["c2_diagnosis"]["diagnosis_version"] = "mock-diagnosis-001"

    fields["c3_repair_package"].update(
        repair_dataset_uri="mock_repair.csv",
        n_synthetic_rows=len(mock_repair),
        augmentation_ratio=len(mock_repair) / len(reference),
        repair_version="mock-repair-001",
    )

    upstream = UpstreamInput.model_validate(fields)

    return {
        "root": tmp_path,
        "upstream": upstream,
        "clean_release": clean_release,
        "evaluated_release": evaluated_release,
        "manifest_directory": manifest_directory,
        # Explicit development/test inputs, not full runtime checks.
        "schema_compatible": True,
        "rollback_ready": True,
        "champion_artifact": champion,
    }


def test_mock_workflow_trains_versions_and_saves_evidence(workflow_inputs):
    inputs = workflow_inputs.copy()
    champion = inputs.pop("champion_artifact")
    original_champion = champion.model_path.read_bytes()

    result = run_test_workflow(**inputs)

    assert result["decision"] is None
    assert result["gates"]["overall"]["passed"] is True

    candidate = result["candidate"]
    evidence = result["evidence"]

    assert candidate.metadata["reference_rows"] == 40
    assert candidate.metadata["repair_rows"] == 8
    assert candidate.metadata["training_rows"] == 48

    candidate_model, _ = load_candidate(candidate.model_path.parent)
    load_champion(champion.model_path.parent)

    assert champion.model_path.read_bytes() == original_champion

    features = inputs["evaluated_release"].loc[
        :, list(FEATURE_COLUMNS)
    ]
    probabilities = candidate_model.predict_proba(features)
    assert probabilities.shape == (len(features), 2)
    assert np.isfinite(probabilities).all()

    payload = json.loads(
        evidence.evidence_path.read_text(encoding="utf-8")
    )

    assert hashlib.sha256(
        evidence.evidence_path.read_bytes()
    ).hexdigest() == evidence.evidence_sha256

    assert payload["release_membership_verified"] is True
    assert payload["clean_release_contents_verified"] is True
    assert payload["slice_basis"] == "c2_diagnosed_slice"
    assert payload["repair_version"] == "mock-repair-001"
    assert payload["candidate_model_sha256"] == (
        candidate.metadata["model_sha256"]
    )

    assert payload["report"]["comparisons"]["global"]["rows"] == (
        len(inputs["clean_release"])
    )


@pytest.mark.parametrize(
    "failure",
    [
        "repair_rejected",
        "privacy_failed",
        "unstable_diagnosis",
        "schema_failed",
        "rollback_failed",
    ],
)
def test_failed_gate_stops_before_training(workflow_inputs, failure):
    inputs = workflow_inputs.copy()
    inputs.pop("champion_artifact")
    fields = inputs["upstream"].model_dump()

    if failure == "repair_rejected":
        fields["c3_repair_package"]["repair_package_status"] = "REJECTED"
    elif failure == "privacy_failed":
        fields["c3_repair_package"]["privacy_status"] = "FAIL"
    elif failure == "unstable_diagnosis":
        fields["c2_diagnosis"]["diagnosis_stability"] = 0.49
    elif failure == "schema_failed":
        inputs["schema_compatible"] = False
    elif failure == "rollback_failed":
        inputs["rollback_ready"] = False

    inputs["upstream"] = UpstreamInput.model_validate(fields)

    def forbidden_training(*args, **kwargs):
        pytest.fail("Candidate training ran after a mandatory gate failure.")

    result = run_test_workflow(
        **inputs,
        training_function=forbidden_training,
    )

    assert result["decision"] == "ESCALATE"
    assert result["candidate"] is None
    assert result["evidence"] is None
    assert not (
        inputs["root"] / "c4_artifacts" / "candidate"
    ).exists()