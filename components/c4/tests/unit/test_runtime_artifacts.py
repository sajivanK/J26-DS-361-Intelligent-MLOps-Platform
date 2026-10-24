"""Persistence tests with temporary synthetic bytes, not real model artifacts."""
from copy import deepcopy
import hashlib
import json
import pytest
from c4.models.runtime_candidate import RuntimeCandidateResult
from c4.models.runtime_artifacts import save_runtime_candidate, load_runtime_candidate


def result():
    model = b"synthetic-test-model"
    return RuntimeCandidateResult(model, {
        "status": "CANDIDATE_EVALUATED_RELEASE_PENDING", "run_id": "run-1",
        "candidate_model_sha256": hashlib.sha256(model).hexdigest(),
        "candidate_training_executed": True, "candidate_reload_verified": True,
        "model_runtime_validated": True, "production_release_authorized": False,
        "forecast_applicable": False, "N1_executed": False, "N2_executed": False,
        "future_outcomes_read": False, "validation_used_for_training": False,
        "artifact_persistence_completed": False,
    })


def save(tmp_path, value=None, version="runtime-1"):
    return save_runtime_candidate(value or result(), data_root=tmp_path, version=version)


def test_roundtrip_separate_location_no_report_mutation(tmp_path):
    original = result(); before = deepcopy(original.report)
    artifact = save(tmp_path, original)
    assert artifact.directory == tmp_path / "c4_artifacts/operational_candidates/adult/runtime-1"
    loaded, manifest = load_runtime_candidate(artifact.directory,
        expected_manifest_sha256=artifact.manifest_sha256)
    assert loaded.candidate_model_bytes == original.candidate_model_bytes
    assert loaded.report == original.report == before
    assert manifest["artifact_persistence_completed"] is True
    assert manifest["production_release_authorized"] is False


def test_existing_version_never_overwritten(tmp_path):
    artifact = save(tmp_path)
    before = (artifact.directory / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError): save(tmp_path)
    assert (artifact.directory / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("version", ["../escape", "a/b", ".", "", "CON", "NUL.txt", "bad."])
def test_invalid_versions(tmp_path, version):
    with pytest.raises(ValueError): save(tmp_path, version=version)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["candidate.ubj", "evaluation.json", "manifest.json"])
def test_tampering_detected(tmp_path, name):
    artifact = save(tmp_path)
    (artifact.directory / name).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        load_runtime_candidate(artifact.directory, expected_manifest_sha256=artifact.manifest_sha256)


def test_incomplete_directory_rejected_and_preserved(tmp_path):
    artifact = save(tmp_path)
    (artifact.directory / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_runtime_candidate(artifact.directory, expected_manifest_sha256=artifact.manifest_sha256)
    assert (artifact.directory / "candidate.ubj").is_file()
    with pytest.raises(FileExistsError): save(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("production_release_authorized", True), ("forecast_applicable", True),
    ("candidate_reload_verified", False), ("future_outcomes_read", True),
    ("candidate_model_sha256", "0" * 64), ("artifact_persistence_completed", True),
])
def test_bad_report_rejected_before_writes(tmp_path, field, value):
    candidate = result(); candidate.report[field] = value
    with pytest.raises(ValueError): save(tmp_path, candidate)
    assert not list(tmp_path.iterdir())


def test_nonfinite_report_rejected_before_writes(tmp_path):
    candidate = result(); candidate.report["metric"] = float("nan")
    with pytest.raises(ValueError): save(tmp_path, candidate)
    assert not list(tmp_path.iterdir())


def test_independent_checksum_anchor_required(tmp_path):
    artifact = save(tmp_path)
    with pytest.raises(ValueError):
        load_runtime_candidate(artifact.directory, expected_manifest_sha256="invalid")
