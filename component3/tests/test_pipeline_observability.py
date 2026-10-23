"""Focused regression tests for identity, live events and persisted evidence."""
import json
import sys
from pathlib import Path

import pytest

from component3.tests.test_live_repair import supplied, IndependentSampler
from component3.src import live_repair as live
from component3.src.contracts.external_contracts import ContractValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "webapp/backend"))
from app import orchestrator


@pytest.mark.parametrize("drift,harm,severity,expected", [
    (False, .03, "low", False), (True, .005, "none", False),
    (True, .03, "moderate", True), (True, -.01, "none", False),
])
def test_meaningful_repair_gate(drift, harm, severity, expected):
    assert orchestrator.repair_gate({"drift_detected": drift, "predicted_harm": harm,
                                     "harm_severity": severity}) is expected


def test_saved_events_and_output_refer_to_exact_run(supplied):
    payload, root = supplied
    payload["data_context"].update(run_id="pipeline-a", batch_id="batch-a", drift_event_id="drift-a")
    package, log = live.execute_live_repair(payload, root / "out", generator_selector=IndependentSampler())
    directory = root / "out" / package.run_id
    events = json.loads((directory / "progress.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    assert events["pipeline_run_id"] == result["pipeline_run_id"] == "pipeline-a"
    assert events["batch_id"] == result["batch_id"] == "batch-a"
    assert result["provenance"]["drift_event_id"] == "drift-a"
    assert len(events["stages"]) == 14
    assert events["stages"][5]["status"] == "Completed"
    assert events["stages"][13]["status"] == "Completed"
    assert result["provenance"]["generated_rows"] == result["provenance"]["generator_configuration"]["requested_rows"]
    assert result["provenance"]["quality_thresholds"]["fidelity_ks_max"] == .35


def test_blocked_run_does_not_claim_generation(supplied):
    payload, root = supplied
    payload["c2_diagnosis"]["diagnosis_stability"] = .1
    package, _ = live.execute_live_repair(payload, root / "out", generator_selector=IndependentSampler())
    events = json.loads((root / "out" / package.run_id / "progress.json").read_text())
    assert events["status"] == "BLOCKED"
    assert events["stages"][1]["status"] == "Blocked"
    assert events["stages"][5]["status"] == "Waiting"


@pytest.mark.parametrize("field", ["dataset_id", "run_id", "batch_id", "drift_event_id"])
def test_mismatched_upstream_identity_rejected(supplied, field):
    payload, root = supplied
    payload["data_context"][field] = "expected"
    payload["c2_diagnosis"][field] = "different"
    with pytest.raises(ContractValidationError, match=field):
        live.execute_live_repair(payload, root / "out", generator_selector=IndependentSampler())


def test_pipeline_publishes_running_before_execution(monkeypatch):
    published = []
    monkeypatch.setattr(orchestrator.runs, "save_latest", lambda result: published.append(result))
    class Stage:
        name, title, source, loaded, produces = "c3", "test", "real", True, ()
        def run(self, batch, upstream):
            assert published[-1]["steps"][0]["status"] == "running"
            return {}
    monkeypatch.setattr(orchestrator, "PIPELINE", [Stage()])
    result = orchestrator.run_pipeline(None, {"batch_id": "b"})
    assert result["status"] == "completed"
    assert result["context"]["batch_id"] == "b"


@pytest.mark.parametrize("field,value", [("fidelity_ks_max", .7), ("privacy_dcr_min", .001),
    ("minimum_utility_gain", .001), ("max_near_duplicate_ratio", .8),
    ("control_utility_tolerance", .05), ("require_bootstrap_positive", False)])
def test_approval_policy_cannot_be_relaxed_by_request(supplied, field, value):
    payload, root = supplied
    payload["data_context"][field] = value
    generator = IndependentSampler()
    with pytest.raises(ContractValidationError, match=field):
        live.execute_live_repair(payload, root / "out", generator_selector=generator)
    assert generator.seen is None


def test_historical_manifest_can_be_loaded_without_live_result(tmp_path, monkeypatch):
    from app.routers import c3
    from component3.src.contracts.repair_package import RepairPackage, RepairPackageStatus
    run_id = "c3-" + "a" * 32
    package = RepairPackage(run_id=run_id, model_id="historical-model", package_status=RepairPackageStatus.BLOCKED,
                            rejection_reasons=("Missing labels",))
    package.save_package(tmp_path / run_id)
    monkeypatch.setattr(c3, "run_root", lambda: tmp_path)
    records = c3.repair_runs()
    assert len(records) == 1
    assert records[0]["progress"] is None
    assert records[0]["result"]["execution_status"] == "BLOCKED"
    assert records[0]["result"]["model_id"] == "historical-model"
    assert records[0]["result"]["preview_rows"] == []
    assert records[0]["result"]["download_url"] is None
