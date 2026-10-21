"""Automated tests for the Component 3 backend API server."""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
import requests
import uvicorn

from component3.src.server import app, health, run_demo_repair, run_repair


def test_server_health_handler():
    data = health()
    assert data["status"] == "ok"
    assert data["component"] == "c3"
    assert data["version"] == "1.0.0"
    assert "TVAE" in data["supported_generators"]
    assert data["default_decision_mode"] == "RULE_BASED"


def test_server_repair_empty_payload_guard():
    resp_none = run_repair(None)
    assert resp_none.status_code == 400
    body_none = json.loads(resp_none.body.decode("utf-8"))
    assert body_none["status"] == "MISSING_PAYLOAD"
    assert "cannot be empty" in body_none["detail"]

    resp_empty = run_repair({})
    assert resp_empty.status_code == 400
    body_empty = json.loads(resp_empty.body.decode("utf-8"))
    assert body_empty["status"] == "MISSING_PAYLOAD"


def test_server_repair_dependency_error_handling():
    bad_payload = {
        "c1_evidence": {
            "drift_detected": True,
            "drift_type": "COVARIATE_SHIFT",
            "predicted_accuracy_drop": 0.12,
            "confidence_interval": [0.06, 0.18],
            "drift_severity": "MODERATE",
            "affected_features": ["education-num"],
        },
        "c2_diagnosis": {
            "root_cause_feature": "education-num",
            "root_cause_category": "COVARIATE_SHIFT",
            "affected_slice_rule": "education_num > 12",
            "affected_features_ranked": ["education-num"],
            "slice_support": 0.20,
            "diagnosis_stability": 0.85,
            "drift_impact_score": 0.85,
        },
        "data_context": {
            "model_id": "base_model_adult",
            "dataset_id": "uci_adult",
            "reference_dataset_uri": "shared/uci_adult/data/reference.csv",
            "current_dataset_uri": "shared/uci_adult/data/current.csv",
            "target_column": "income",
            "preprocessing_uri": "shared/uci_adult/models/preprocessor.pkl",
        },
    }
    resp = run_repair(bad_payload)
    assert resp.status_code == 400
    body = json.loads(resp.body.decode("utf-8"))
    assert body["status"] == "DEPENDENCY_RESOLUTION_ERROR"
    assert body["error_type"] == "DependencyResolutionError"
    assert "shared/uci_adult/data/current.csv" in body["detail"]


def test_server_demo_repair_executes_telco():
    resp = run_demo_repair(None)
    assert isinstance(resp, dict), f"Expected dict response, got: {type(resp)}"
    assert resp["status"] == "COMPLETED_APPROVED"
    assert resp["package_status"] == "APPROVED_FOR_REVIEW"
    assert resp["generator"] == "TVAE"
    assert resp["quantity_selected"] > 0
    assert resp["duration_seconds"] > 0
    assert resp["c4_flat_contract"] is not None
    assert resp["c4_flat_contract"]["repair_package_status"] == "APPROVED"
    assert resp["candidate_dataset_path"] is not None


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_real_http_wire_requests():
    port = _get_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(30):
        try:
            response = requests.get(f"{base_url}/api/c3/health", timeout=1.0)
            if response.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("FastAPI server failed to start within timeout.")

    try:
        health_response = requests.get(f"{base_url}/api/c3/health", timeout=2.0)
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ok"
        assert health_response.json()["component"] == "c3"

        empty_response = requests.post(
            f"{base_url}/api/c3/repair",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=2.0,
        )
        assert empty_response.status_code == 400
        assert empty_response.json()["status"] == "MISSING_PAYLOAD"
    finally:
        server.should_exit = True
        thread.join(timeout=2.0)
