# Component 3 Backend Connection and REST API Guide

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Author:** Mathusan K (IT23174580)  
**Date:** October 2026

---

## 1. Overview

Component 3 includes a standalone, production-style FastAPI server (`component3/src/server.py`) that exposes REST API endpoints for health checks and repair execution.

---

## 2. Starting the Server

Run from the project root using the project virtual environment:

```powershell
.venv\Scripts\python.exe -m component3.src.server --host 127.0.0.1 --port 8003
```

### CLI Arguments
- `--host`: Host to bind (default: `127.0.0.1`)
- `--port`: Port number (default: `8003`)

## 3. API Endpoints

### 3.1 Health & Engine Metadata
- **Endpoint:** `GET /api/c3/health`
- **Response:**
```json
{
  "status": "ok",
  "component": "c3",
  "title": "C3 Synthetic Data Repair",
  "owner": "Mathusan K",
  "version": "1.0.0",
  "contract_version": "1.0.0",
  "supported_generators": ["TVAE", "CTGAN", "TabDDPM"],
  "default_decision_mode": "RULE_BASED",
  "supported_datasets": [
    {
      "id": "telco_churn",
      "label": "Telco Customer Churn",
      "status": "operational",
      "source_model_verified": true
    },
    {
      "id": "uci_adult",
      "label": "UCI Adult (Census Income)",
      "status": "reference_only",
      "note": "Reference dataset available; current batch pending C1/C2 delivery"
    }
  ],
  "storage_root": "component3/results/runs"
}
```

### 3.2 Execute Live Repair Workflow
- **Endpoint:** `POST /api/c3/repair`
- **Payload:** Accepts the agreed 3-section contract (`c1_evidence` + `c2_diagnosis` + `data_context`). If body is empty (`{}` or null), executes the canonical Telco Churn high covariate drift repair on CPU.
- **Success Response (HTTP 200):**
```json
{
  "status": "SUCCESS",
  "package_id": "PKG-23DC67142EB4",
  "package_status": "APPROVED_FOR_REVIEW",
  "chosen_strategy": "TARGETED_SYNTHETIC_REPAIR",
  "generator": "TVAE",
  "quantity_selected": 100,
  "duration_seconds": 15.899,
  "c4_flat_contract": {
    "repair_dataset_uri": "component3/results/live_runs/pkg-23dc67142eb4_candidates.csv",
    "repair_package_status": "APPROVED",
    "generator": "TVAE",
    "n_synthetic_rows": 100,
    "augmentation_ratio": 0.0267,
    "fidelity_score": 0.7236,
    "utility_score": 0.64,
    "privacy_status": "PASS",
    "target_coverage": 1.0,
    "prescription_confidence": 0.8,
    "repair_version": "c3-23dc6714",
    "repair_timestamp": "2026-10-21T11:55:31.000000+00:00"
  },
  "candidate_dataset_sha256": "6cf32eca7855b50027edbc356216099ffdd1dcb7c9a78cf6141692f9682a0be2"
}
```

### 3.3 Dependency Resolution Error Handling
- **Status:** HTTP 400
- **Response:**
```json
{
  "status": "DEPENDENCY_RESOLUTION_ERROR",
  "error_type": "DependencyResolutionError",
  "detail": "Current drifted dataset not found at URI 'shared/uci_adult/data/current.csv'. Component 3 will not silently substitute another dataset.",
  "resolution_recommendation": "Deliver required external dataset batch before running repair."
}
```

---

## 4. Resource Bounds & Safeguards

- **CPU-Only Execution:** All generator plugins (TVAE, CTGAN, TabDDPM) operate on CPU using bounded epoch counts (10 epochs for demonstration repair).
- **Time Bounds:** Live executions complete in 10-15 seconds.
- **Zero-Substitution Guard:** Prohibits replacing missing Adult assets with Telco assets.
