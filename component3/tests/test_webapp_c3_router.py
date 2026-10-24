"""Web integration: actual contracts, uploads, approved downloads and fake-input guard."""
import io
import json
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webapp/backend"))

from app.routers import c3 as router_module
from app import orchestrator
from app.components.c3 import C3Component
from component3.src import live_repair as live
from component3.tests.test_live_repair import supplied, IndependentSampler


@pytest.fixture
def client(supplied, monkeypatch):
    _, root = supplied
    application = FastAPI()
    application.include_router(router_module.router)
    monkeypatch.setattr(router_module, "REPO_ROOT", root)
    monkeypatch.setattr(router_module, "C3_UPLOAD_DIR", root / "uploads")
    monkeypatch.setattr(router_module, "run_root", lambda: root / "component3/results/live_runs")
    execute = live.execute_live_repair
    monkeypatch.setattr(router_module, "execute_live_repair", lambda payload:
        execute(payload, root / "component3/results/live_runs", generator_selector=IndependentSampler()))
    return TestClient(application)


def test_c3_info_endpoint(client):
    response = client.get("/api/c3/info")
    assert response.status_code == 200
    assert response.json()["engine_version"] == "2.0.0"
    assert "tabular_binary_classification" in response.json()["supported_tasks"]
    assert response.json()["decision_modes"] == ["RULE_BASED"]


def test_actual_contract_executes_and_download_is_integrity_checked(client, supplied):
    payload, root = supplied
    response = client.post("/api/c3/repair", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["repair_package_status"] == "APPROVED"
    assert result["provenance"]["artifact_paths"]["current_dataset"] == payload["data_context"]["current_dataset_uri"]
    assert response.headers["x-run-id"] == result["run_id"]
    download = client.get(result["download_url"])
    assert download.status_code == 200
    assert "outcome" in download.text
    path = root / "component3/results/live_runs" / result["run_id"] / "repair.csv"
    path.write_text("modified", encoding="utf-8")
    assert client.get(result["download_url"]).status_code == 409


def test_upload_saves_unique_artifact_paths(client):
    first = client.post("/api/c3/artifacts", data={"role": "current_dataset"}, files={"file": ("../batch.csv", b"x,y\n1,0\n", "text/csv")})
    second = client.post("/api/c3/artifacts", data={"role": "current_dataset"}, files={"file": ("batch.csv", b"x,y\n1,0\n", "text/csv")})
    assert first.status_code == second.status_code == 200
    assert first.json()["uri"] != second.json()["uri"]
    assert ".." not in first.json()["uri"]
    assert len(first.json()["sha256"]) == 64
    invalid = client.post("/api/c3/artifacts", data={"role": "model"}, files={"file": ("model.py", b"x")})
    assert invalid.status_code == 422


def test_missing_or_bad_contract_rejected(client, supplied):
    assert client.post("/api/c3/repair", json={}).status_code == 422
    payload, _ = supplied
    payload["c2_diagnosis"]["affected_slice_rule"] = "unknown > 0"
    assert client.post("/api/c3/repair", json=payload).status_code == 422
    assert client.get("/api/c3/packages/not-a-run/dataset").status_code == 404


def test_latest_inputs_reject_fake_c2(client, supplied, monkeypatch):
    payload, _ = supplied
    monkeypatch.setattr(router_module.runs, "latest", lambda: {
        "context": payload["data_context"], "outputs": {"c1": payload["c1_evidence"], "c2_to_c3": payload["c2_diagnosis"]},
        "component_sources": {"c1": "real", "c2": "fake"}})
    assert client.get("/api/c3/upstream/latest").status_code == 409


def test_latest_real_inputs_are_forwarded(client, supplied, monkeypatch):
    payload, _ = supplied
    monkeypatch.setattr(router_module.runs, "latest", lambda: {
        "context": payload["data_context"], "outputs": {"c1": payload["c1_evidence"], "c2_to_c3": payload["c2_diagnosis"]},
        "component_sources": {"c1": "real", "c2": "real"}})
    output = client.get("/api/c3/upstream/latest").json()
    assert output["c1_evidence"] == payload["c1_evidence"]
    assert output["c2_diagnosis"] == payload["c2_diagnosis"]
    assert output["data_context"]["model_uri"] == payload["data_context"]["model_uri"]


def test_platform_c3_uses_actual_batch_and_model(supplied, monkeypatch):
    import pandas as pd
    payload, root = supplied
    execute = live.execute_live_repair
    monkeypatch.setattr(live, "execute_live_repair", lambda value:
        execute(value, root / "out", generator_selector=IndependentSampler()))
    batch = pd.read_csv(payload["data_context"]["current_dataset_uri"]).drop(columns="outcome")
    component = C3Component()
    output = component.run(batch, {"context": payload["data_context"],
        "c1": payload["c1_evidence"], "c2_to_c3": payload["c2_diagnosis"]})["c3"]
    assert output["repair_package_status"] == "APPROVED"
    assert output["provenance"]["artifact_paths"]["model"] == payload["data_context"]["model_uri"]
    with pytest.raises(ValueError, match="actual monitored batch"):
        component.run(batch.iloc[:20], {"context": payload["data_context"],
            "c1": payload["c1_evidence"], "c2_to_c3": payload["c2_diagnosis"]})


def test_orchestrator_stops_on_output_contract_violation(monkeypatch):
    class Broken:
        name, title, source, loaded, produces = "c1", "test", "real", True, ("c1",)
        def run(self, batch, upstream): return {"c1": {"harm_severity": "high"}}
    class Never:
        name, title, source, loaded = "c2", "test", "real", True
        def run(self, *args): pytest.fail("Invalid output reached downstream")
    monkeypatch.setattr(orchestrator, "PIPELINE", [Broken(), Never()])
    result = orchestrator.run_pipeline(None, {})
    assert result["status"] == "failed"
    assert len(result["steps"]) == 1
