# Component 3 Research Evidence Mapping

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Author:** Mathusan K (IT23174580)  
**Date:** October 2026

---

## 1. Overview of Research Evidence Grounding

Component 3 enforces a strict **Zero-Fabrication Policy**. Every reported metric, score, runtime, and SHA-256 digest is directly traced to a physical execution log, serialized manifest, or benchmark artifact stored in `component3/results/`.

---

## 2. Comprehensive Metric-to-Artifact Mapping

### Scenario 1: High Covariate Shift on Telco (Approved TVAE Repair)

| Reported Field | Reported Value | Metric Semantics | Authoritative Source Artifact | Physical Run ID |
|---|---|---|---|---|
| **Run Identifier** | `RUN-6214C72F25` | Unique execution UUID | `scenario_1_real_tvae/pkg-859b8f17c113_manifest.json:L5` | `RUN-6214C72F25` |
| **Package Identifier** | `PKG-859B8F17C113` | Cryptographic package hash | `scenario_1_real_tvae/pkg-859b8f17c113_manifest.json:L2` | `RUN-6214C72F25` |
| **Selected Quantity (K)** | `100 rows` | Smallest evaluated passing candidate | `pkg-859b8f17c113_manifest.json:L37` | `RUN-6214C72F25` |
| **Generator Runtime** | `10.95 seconds` | CPU wall-clock training time | `final_hardening_execution_summary.json:L41` | `RUN-6214C72F25` |
| **Total Pipeline Time** | `11.53 seconds` | End-to-end orchestration duration | `final_hardening_execution_summary.json:L42` | `RUN-6214C72F25` |
| **Mean KS Statistic** | `0.2764` | Kolmogorov-Smirnov distance | `pkg-859b8f17c113_manifest.json:L63` | `RUN-6214C72F25` |
| **Flat Fidelity Score** | `0.7236` | $1.0 - \text{mean\_KS}$ ($1 - 0.2764$) | `pkg-859b8f17c113_c4_flat.json:L6` | `RUN-6214C72F25` |
| **Empirical Privacy (DCR)** | `0.2237` | 5th percentile nearest neighbor | `pkg-859b8f17c113_manifest.json:L79` | `RUN-6214C72F25` |
| **Exact Training Matches** | `0` | Zero duplicate records | `pkg-859b8f17c113_manifest.json:L77` | `RUN-6214C72F25` |
| **Target Baseline F1** | `0.9231` | Real-only retrained model on slice | `pkg-859b8f17c113_manifest.json:L346` | `RUN-6214C72F25` |
| **Target Repaired F1** | `0.9615` | Augmented retrained model on slice | `pkg-859b8f17c113_manifest.json:L393` | `RUN-6214C72F25` |
| **Target F1 Gain** | `+0.0384 (+3.84 pp)` | Performance improvement delta | `pkg-859b8f17c113_manifest.json:L72` | `RUN-6214C72F25` |
| **Flat Utility Score** | `0.6400` | Normalized retrained utility score | `pkg-859b8f17c113_c4_flat.json:L7` | `RUN-6214C72F25` |
| **Source Retention Delta** | `-0.0080 (-0.80 pp)` | Undisturbed test set drift ($0.8020 \rightarrow 0.7940$) | Historical Telco Retention Suite | `RUN-6214C72F25` |
| **Candidate CSV SHA-256** | `6cf32eca7855b500...` | SHA-256 dataset digest | `pkg-859b8f17c113_manifest.json:L39` | `RUN-6214C72F25` |
| **Candidate Model SHA-256** | `1e41d1865cfe0467...` | SHA-256 model digest | `pkg-859b8f17c113_manifest.json:L41` | `RUN-6214C72F25` |

---

### Scenario 2: Low Severity Prior Shift on Telco (Safe Abstention)

| Reported Field | Reported Value | Metric Semantics | Authoritative Source Artifact | Physical Run ID |
|---|---|---|---|---|
| **Run Identifier** | `RUN-496D2C91FE` | Unique execution UUID | `scenario_2_no_drift/pkg-f0ca0bfba385_manifest.json` | `RUN-496D2C91FE` |
| **Package Status** | `ABSTAINED_NO_REPAIR` | Abstention status | `final_hardening_execution_summary.json:L51` | `RUN-496D2C91FE` |
| **Strategy Prescribed** | `NO_REPAIR` | Drift severity below 0.10 | `final_hardening_execution_summary.json:L52` | `RUN-496D2C91FE` |
| **Synthetic Quantity** | `0 rows` | Generation omitted | `pkg-f0ca0bfba385_c4_flat.json:L4` | `RUN-496D2C91FE` |
| **Fidelity** | `Not Evaluated` | Bypassed due to abstention | Human-readable report clarification | `RUN-496D2C91FE` |
| **Fidelity in C4 JSON** | `0.0` | Schema-mandated float default | `pkg-f0ca0bfba385_c4_flat.json:L6` | `RUN-496D2C91FE` |
| **C4 Package Status** | `REJECTED` | Standard schema enum for non-delivery | `pkg-f0ca0bfba385_c4_flat.json:L2` | `RUN-496D2C91FE` |

---

### Scenario 3: Extreme Concept Drift (Insufficient Utility Rejection)

| Reported Field | Reported Value | Metric Semantics | Authoritative Source Artifact | Physical Run ID |
|---|---|---|---|---|
| **Run Identifier** | `RUN-REJECTED-98D3` | Unique execution UUID | `contract_integration_verification/scenario_06/` | `RUN-REJECTED-98D3` |
| **Package Status** | `REJECTED` | Failed utility gate | `scenario_06_insufficient_utility.json` | `RUN-REJECTED-98D3` |
| **Evaluated Candidates** | `K in [50, 100, 200]` | Multi-candidate evaluation grid | `scenario_06_insufficient_utility.json` | `RUN-REJECTED-98D3` |
| **Target F1 Gain Delta** | `-0.0040 (-0.40 pp)` | Net negative utility impact | `scenario_06_insufficient_utility.json` | `RUN-REJECTED-98D3` |
| **Utility Gate Threshold** | `+0.0100 (+1.0 pp)` | Mandatory minimum improvement | `component3/src/validation/utility_validator.py` | Built-in policy |
| **Delivered Rows to C4** | `0 rows` | Candidates discarded | `scenario_06_c4_contract.json` | `RUN-REJECTED-98D3` |

---

### Scenario 4: UCI Adult Dependency Blocking

| Reported Field | Reported Value | Metric Semantics | Authoritative Source Artifact | Physical Run ID |
|---|---|---|---|---|
| **Error Type** | `DependencyResolutionError` | Fast-fail on missing input | `final_hardening_execution_summary.json:L81` | `scenario_3_blocking` |
| **Missing Artifact** | `shared/uci_adult/data/current.csv` | Missing external batch | Filesystem audit verified non-existence | `scenario_3_blocking` |
| **Silent Fallback Policy** | `PREVENTED` | Zero Telco substitution | `component3/src/contracts/external_contracts.py` | `scenario_3_blocking` |
