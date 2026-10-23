# Component 3 Remaining External Integration Dependencies

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Author:** Mathusan K (IT23174580)  
**Readiness Level:** Level 2 (Contract-Ready for Integration Testing)  
**Target for Level 3:** Multi-Component Live Execution on Integrated Platform  
**Date:** October 2026

---

## 1. Overview

Component 3 has satisfied all internal engineering, contract compliance, and backend testing requirements. Both recorded demonstration and live CPU-bounded execution modes are verified.

Advancing from **Level 2 (Contract-Ready)** to **Level 3 (Full Team Live Integration)** requires resolving 6 external cross-component dependencies.

---

## 2. Dependency Tracking Matrix

| Dependency ID | Category | External Owner | Required Action | Current Workaround / Status |
|---|---|---|---|---|
| **DEP-01** | Dataset Asset | Teammate (C1/C2) | Provide drifted dataset batch `shared/uci_adult/data/current.csv` with known ground-truth labels for retraining utility evaluation. | Component 3 enforces fail-fast error (`DependencyResolutionError`). Tested in Scenario 4. |
| **DEP-02** | Model Asset | Teammate (C1/C4) | Provide trained baseline model artifact `champion-v12` referenced in teammate contracts, or formally standardize on `base_model_adult.pkl`. | Baseline `base_model_adult.pkl` verified on disk; alias mapping documented. |
| **DEP-03** | Infrastructure | Platform Lead (Sajivan K) | Designate shared artifact storage volume for cross-container exchange (e.g., shared Docker volume or cloud bucket). | Component 3 currently emits relative repository paths (`component3/results/runs/`) with forward slashes. Configurable via `C3_ARTIFACT_STORAGE_ROOT`. |
| **DEP-04** | Consumer Verification | Teammate (C4) | Verify that Component 4's retraining coordinator reads and parses the flat 12-field contract exported by Component 3. | Validated on Component 3 side using simulated C4 consumer test (`shared/contracts/c3_output_schema.json`). |
| **DEP-05** | End-to-End Orchestration | Full Team | Execute live chain: C1 Drift Detection &rarr; C2 Diagnosis &rarr; C3 Repair &rarr; C4 Retraining on shared platform environment. | Component 3 CLI (`c3 contract-repair`) and REST API (`POST /api/c3/repair`) stand ready to receive pipeline triggers. |

---

## 3. Recommended Resolution Protocol for Team Meeting

1. **DEP-01 & DEP-02:** Coordinate with C1 and C2 owners to place `current.csv` into `shared/uci_adult/data/`.
2. **DEP-03:** Standardize on Docker volume path `/app/shared_artifacts` across all 4 container definitions.
3. **DEP-04:** Provide C4 lead with sample approved JSON payload (`pkg-859b8f17c113_c4_flat.json`) to confirm schema compatibility.
