# Component 3 — Remaining External Dependencies & Readiness Roadmap

**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Project:** J26-DS-361 Intelligent MLOps Platform  
**Owner:** Mathusan K (IT23174580)  
**Verification Date:** 2026-10-21  
**Current Status:** **Level 2 (Contract-Ready for Integration Testing)**  

---

## 1. Executive Summary

Component 3 has completed all internal engineering, contract adaptation, validation pipelines, safety evaluations, error handling hardening, and simulated consumer verifications. All functionality within Component 3's boundary is complete, reproducible, and robustly tested.

In strict adherence to the project charter and independent QA standards:
- **Component 3 claims Level 2 Readiness.**
- **Component 3 explicitly does NOT claim Level 3 or Level 4 Readiness.** Level 3 requires live end-to-end integration across all four components running synchronously, while Level 4 requires independent operational validation in a production-like staging environment.

This document clearly itemizes the specific external deliverables and actions required from teammates and platform architects to advance the platform to Level 3 and Level 4.

---

## 2. Itemized External Dependencies Inventory

| Dependency ID | Item Description | Responsible Party / Component | Impact on Component 3 | Required Action |
| :--- | :--- | :--- | :--- | :--- |
| **DEP-01** | **`shared/uci_adult/data/current.csv`** | Component 1 (Drift Detection) / Component 2 (Root Cause) | Multi-component pipeline cannot run UCI Adult drift repair end-to-end without drifted batch. | C1/C2 must generate or provide `current.csv` representing the drifted batch in `shared/uci_adult/data/`. |
| **DEP-02** | **`champion-v12` Model Registration** | Platform Orchestrator / Component 4 | C3 dependency resolver rejects payloads specifying `champion-v12` because the file does not exist on disk. | Platform orchestrator must either provide `champion-v12` artifact or update payload to reference registered model `base_model_adult.pkl`. |
| **DEP-03** | **Live Component 4 Consumer Validation** | Component 4 (Kesigan M) | Live handoff verification currently relies on simulated C3-side consumer tests. | Kesigan M must run C4 retraining gatekeeper against C3 exported JSON (`c4_contract_output.json`) and confirm candidate ingestion. |
| **DEP-04** | **Shared Artifact Storage Root** | Platform Architecture Team | Local storage uses `component3/results/runs/` by default. Deployment storage requires consensus. | Platform team must define the shared volume/directory and set the `C3_ARTIFACT_STORAGE_ROOT` environment variable across environments. |
| **DEP-05** | **Pre-commit Cross-Component Schema CI** | Integration Architecture / Team Lead | Manual validation required until automated multi-component contract CI is integrated. | Add automated GitHub Actions / GitLab CI step validating payloads against `shared/contracts/` schemas. |

---

## 3. Detailed Technical Analysis of Blockers

### A. The `champion-v12` Missing Artifact
In the contract documentation and example payloads agreed across the team, the model identifier is specified as:
```json
"data_context": {
  "model_id": "champion-v12",
  "reference_dataset_uri": "shared/uci_adult/data/reference.csv",
  "current_dataset_uri": "shared/uci_adult/data/current.csv"
}
```
**Why Component 3 Cannot Invent It:**
Component 3 is an independent microservice engine. Silently creating a mock model named `champion-v12` would mask integration defects, violate software engineering hygiene, and invalidate downstream evaluation metrics in Component 4. Component 3's dependency resolver will raise `DependencyResolutionError` until the actual artifact is supplied.

### B. The `shared/uci_adult/data/current.csv` Dataset
In `shared/uci_adult/data/`, the repository currently contains:
- `reference.csv` (32,561 rows)
- `test.csv` (16,281 rows)
- `current.csv` (**MISSING**)

Component 1 and Component 2 are responsible for detecting drift between `reference.csv` and a live `current.csv`. For integration testing on Adult, C1/C2 must deposit `current.csv` into `shared/uci_adult/data/`.

---

## 4. Multi-Component Readiness Level Matrix

| Readiness Level | Definition | C3 Status | Blocker / Requirement to Advance |
| :---: | :--- | :---: | :--- |
| **Level 1** | **Internal Engine Working**<br>Core modules, generators, validation gates, and metrics calculate correctly internally. | **100% Complete** | Achieved in Phases 1-3. |
| **Level 2** | **Contract-Ready for Integration Testing**<br>Adapts C1/C2 inputs, strictly adheres to 12-field C4 output schema, portable storage implemented, simulated consumer passes. | **100% Complete** | **CURRENT STATUS ACHIEVED.** |
| **Level 3** | **Real Multi-Component Integration Verified**<br>Live C1 -> C2 -> C3 -> C4 pipeline executes end-to-end with real teammate code and shared artifacts. | **Blocked Externally** | Requires DEP-01 (`current.csv`), DEP-02 (`champion-v12`), and DEP-03 (Live C4 consumer verification). |
| **Level 4** | **Independently Validated Operational Deployment**<br>Automated staging deployment, canary verification, SLA monitoring, and production governance approval. | **Future Phase** | Requires deployment infrastructure, platform orchestrator rollout, and final team sign-off. |
