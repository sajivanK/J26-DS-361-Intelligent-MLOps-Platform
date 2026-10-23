# Component 3 Research Reproducibility and Scientific Integrity Audit

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Author:** Mathusan K (IT23174580)  
**Date:** October 2026  
**Status:** Ground-Truth Audited & Defensible

---

## 1. Research Objectives and Scope

The core scientific hypothesis of Component 3 states that:
> *Targeted synthetic data augmentation, guided by diagnostic slice localization and resource-aware quantity planning, can restore downstream model performance on failing sub-populations under distribution shift while preserving source-task competence, outperforming blind global augmentation.*

To defend this hypothesis rigorously, all experimental findings must be transparently grounded in physical execution artifacts.

---

## 2. Investigation of Reported Metric Discrepancy

A key priority of this audit was resolving the apparent discrepancy between two reported performance outcomes:
- **Result A:** Target F1 $0.6275 \rightarrow 0.6400$
- **Result B:** Target F1 $0.9231 \rightarrow 0.9615$

### Investigation Findings
An exhaustive grep search across all 90 documentation files and results directories revealed that these numbers originate from two distinct experimental evaluations:

| Parameter | Result A ($0.6275 \rightarrow 0.6400$) | Result B ($0.9231 \rightarrow 0.9615$) |
|---|---|---|
| **Authoritative Source Artifact** | `component3/results/utility_evaluation/S04_ctgan_utility_e1_n100_seed42.json`<br>`component3/docs/sprint05_offline_utility_evaluation.md` | `component3/results/final_integration_hardening/scenario_1_real_tvae/pkg-859b8f17c113_manifest.json`<br>`component3/results/runs/RUN-6214C72F25/` |
| **Experimental Phase** | Phase 1 (Sprint 05 — Early Offline Feasibility) | Phase 2 & 4 (Sprint 06 / E2E Integration Hardening) |
| **Generator Family** | CTGAN ($N=100$) | TVAE ($K=100$, smallest passing candidate) |
| **Evaluation Set** | Global / Unpartitioned Telco holdout test set | Diagnosed Target Slice (`Contract == 'Month-to-month' and tenure <= 24`) |
| **Metric Meaning** | Global test set positive-class F1 under blind augmentation | Target-slice retrained model positive-class F1 under targeted repair |
| **Baseline Performance** | $0.6275$ | $0.9231$ |
| **Repaired Performance** | $0.6400$ ($+1.25\text{ pp}$) | $0.9615$ ($+3.84\text{ pp}$) |
| **Source Retention** | Not evaluated in early sprint | Evaluated: $0.8020 \rightarrow 0.7940$ ($-0.80\text{ pp}$, preserved) |

### Conclusion
Both results represent authentic experiments. They must **not** be presented as interchangeable.
- The final integration reports use **Result B ($0.9231 \rightarrow 0.9615$)**, as it evaluates the actual diagnosed target slice under the complete multi-gate validation pipeline.
- Result A remains documented as historical exploratory evidence for early CTGAN feasibility.

---

## 3. Metric Traceability Matrix

| Reported Metric | Value | Evaluation Context | Authoritative Artifact Path | Run ID |
|---|---|---|---|---|
| **Target Baseline F1** | `0.9231` | Baseline model on target slice | `pkg-859b8f17c113_manifest.json:L346` | `RUN-6214C72F25` |
| **Target Repaired F1** | `0.9615` | Model retrained with $K=100$ TVAE on slice | `pkg-859b8f17c113_manifest.json:L411` | `RUN-6214C72F25` |
| **Target Gain** | `+0.0384 (+3.84 pp)` | Exceeds $+0.0100$ threshold | `pkg-859b8f17c113_manifest.json:L415` | `RUN-6214C72F25` |
| **Source Baseline F1** | `0.8020` | Baseline model on source holdout | `pkg-859b8f17c113_manifest.json:L355` | `RUN-6214C72F25` |
| **Source Repaired F1** | `0.7940` | Repaired model on source holdout | `pkg-859b8f17c113_manifest.json:L420` | `RUN-6214C72F25` |
| **Source Retention Delta** | `-0.0080 (-0.80 pp)` | Well within $0.050$ degradation bound | `pkg-859b8f17c113_manifest.json:L424` | `RUN-6214C72F25` |
| **Continuous KS Distance** | `0.2764` | Mean KS over continuous features | `pkg-859b8f17c113_manifest.json:L260` | `RUN-6214C72F25` |
| **Flat Fidelity Score** | `0.7236` | Computed as $1.0 - 0.2764$ | `pkg-859b8f17c113_c4_flat.json:L7` | `RUN-6214C72F25` |
| **5th %ile DCR** | `0.2237` | Distance to Closest Record (privacy) | `pkg-859b8f17c113_manifest.json:L285` | `RUN-6214C72F25` |
| **Exact Duplicates** | `0` | Training record identity check | `pkg-859b8f17c113_manifest.json:L290` | `RUN-6214C72F25` |
| **Selected Quantity** | `100` | Smallest evaluated passing quantity | `pkg-859b8f17c113_manifest.json:L18` | `RUN-6214C72F25` |

---

## 4. Scientific Rigor and Verification

### 4.1 Data Leakage Audit
- Evaluated `ExperimentalPartitionBuilder.build_splits()` on Telco Churn.
- Checked intersection of primary identifier `customerID`:
  - `Train & Eval customerID overlap: 0`
  - `Train & Adapt customerID overlap: 0`
  - `Eval & Adapt customerID overlap: 0`
- **Result:** Partitions are 100% disjoint. Zero data leakage detected.

### 4.2 Statistical Separation of Continuous vs Discrete Features
- In `fidelity_validator.py`, the Kolmogorov-Smirnov 2-sample test is applied strictly to continuous columns (`tenure`, `MonthlyCharges`, `TotalCharges`).
- Discrete/categorical columns are evaluated using Total Variation Distance (TVD) and Jensen-Shannon divergence, preventing mathematical errors from discrete step functions in KS tests.

### 4.3 Empirical Privacy Caveats
- Distance-to-closest-record (DCR) and exact duplicate checking provide empirical verification of data memorization risk.
- Component 3 documentation explicitly notes that empirical DCR is **not** a formal mathematical guarantee of $(\epsilon, \delta)$-differential privacy.

### 4.4 Rule-Based Default Prescription
- The 10-gate rule-based prescription engine remains the default operational mode.
- The learned candidate ranker (`learned_prescription_model.joblib`) is maintained strictly in advisory/shadow mode until external benchmark evaluations across multiple datasets are completed.
