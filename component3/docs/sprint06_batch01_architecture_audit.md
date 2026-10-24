# Sprint 06 — Batch 1: Architecture Audit & Technical Foundation Plan

## 1. Executive Summary & Objective

Component 3 (C3) of the **J26-DS-361 Intelligent MLOps Platform** is the **Adaptive Intelligent Synthetic Data Repair Engine**. Across Sprints 01–05, Component 3 developed:
1. Input contract validation and AST-safe slice extraction (`contracts/`).
2. Data readiness governance and safe target decoupling (`readiness/`).
3. Pluggable synthetic tabular generators (`generators/`: CTGAN, TVAE, TabDDPM).
4. Multi-stage quality, fidelity, privacy, and utility validation (`validation/`).
5. Acceptance policy governance (`acceptance_policy.py`).
6. Automated repair experiment runner and 15 audited experiment records (`experiments/`).

**Sprint 06 Objective:** Develop an **Intelligent Repair Prescription Engine** that recommends an appropriate synthetic data generator, repair quantity, and hyperparameter configuration for any detected data drift problem.

**Batch 1 Focus:** Build the multi-dataset, multi-model, multi-drift experimental foundation supporting scalable experimentation (up to 2,000–4,000+ candidate runs), Google Colab integration, leak-free preprocessing, and rigorous data contracts.

---

## 2. Audit of Existing Component 3 Architecture

### 2.1 Reusable Modules (To Preserve Intact)

| Module Path | Current Responsibility | Reusability in Sprint 06 | Compatibility Requirement |
|---|---|---|---|
| `component3/src/core/` | Configuration loading (`config_loader.py`), dataset loading (`data_loader.py`), legacy model loading (`model_loader.py`). | **100% Reused** | Must support dynamic dataset routing beyond hardcoded `uci_adult`. |
| `component3/src/contracts/` | AST-safe slice parsing (`safe_slice_evaluator.py`), input contract validation (`input_validator.py`), schema validation (`schema_validator.py`). | **100% Reused** | Preserved for safe condition evaluation across any dataset schema. |
| `component3/src/readiness/` | Sample size gates, slice support checks, label safety verification (`readiness_engine.py`, `readiness_models.py`). | **100% Reused** | Universal readiness gate across all multi-dataset experiments. |
| `component3/src/generators/` | Plugin manager (`plugin_manager.py`), generator base interface, data preparer (`data_preparer.py`), CTGAN, TVAE, TabDDPM plugins. | **100% Reused** | Core generator execution engine for all synthetic data generation. |
| `component3/src/validation/` | 4-stage validation engine: Structural, Fidelity, Privacy, Utility, Acceptance Policy (`acceptance_policy.py`). | **100% Reused** | Canonical evaluator for all multi-dataset candidate repairs. |
| `component3/src/experiments/` | `automated_repair_runner.py`, `build_repair_dataset.py`. | **Extended** | Base runner extended to handle multi-dataset registry and standardized contracts. |

---

### 2.2 Compatibility Requirements & Limitations Identified

1. **Dataset Generalization:**
   - *Current State:* `c3_config.json` and `data_loader.py` were historically tailored around `uci_adult` with simulated C1/C2 outputs.
   - *Requirement for Sprint 06:* Must introduce a formal `DatasetRegistry` that abstracts dataset metadata, target columns, column types, and splitting logic without breaking existing `uci_adult` workflows.
2. **Three-Tier Preprocessing Separation:**
   - *Current State:* Generators received prepared raw slices, but model evaluation relied on preprocessor pickle from `shared/uci_adult/`.
   - *Requirement for Sprint 06:* Need a unified `TabularDataPreprocessor` maintaining three distinct representations:
     - Tier 1: Original raw tabular data.
     - Tier 2: Generator-compatible tabular data (discrete column names preserved, no one-hot dimensional explosion).
     - Tier 3: Baseline-model-compatible transformed data (scaled numeric, encoded categorical, target isolated).
3. **Data Leakage & Pre-Decision Demarcation:**
   - *Current State:* Audit 03 revealed that 15 raw experiment records contain both pre-decision features and post-generation evaluation metrics.
   - *Requirement for Sprint 06:* Must enforce a strict `PRE_DECISION_FEATURE_WHITELIST` so that prescription models only receive information available *before* generation runs.
4. **Hardware and Environment Constraints:**
   - *Current State:* Local execution is constrained to CPU. TabDDPM and high-epoch CTGAN training are computationally intensive.
   - *Requirement for Sprint 06:* Architecture must decouple local development/testing from high-throughput experiment execution via standalone, path-agnostic Google Colab workflows.

---

## 3. Sprint 06 Batch 1 Architectural Component Design

```mermaid
flowchart TD
    subgraph Multi-Dataset Layer
        DR[Dataset Registry] -->|Metadata & Specs| DP[Tabular Preprocessor]
        DR -->|Splitting Strategy| DS[Data Splitter]
    end

    subgraph Modeling & Drift Layer
        DS -->|Train Split| BM[Baseline Model Trainer & Adapter]
        DS -->|Test Split| DC[Drift Experiment Configurator]
        DC -->|Drift Scenarios| SE[Slice Extractor & Readiness Engine]
    end

    subgraph Generation & Validation Layer (Preserved C3 Engine)
        SE --> GP[Generator Plugin Manager: CTGAN, TVAE, TabDDPM]
        GP --> VE[4-Stage Validation Engine]
        VE --> AP[Acceptance Policy Engine]
    end

    subgraph Prescription Contract & Matrix Layer
        AP --> PC[Standardized Prescription Data Contract]
        PC -->|Pre-Decision Whitelist| ML[Prescription Meta-Learner Planning]
        MP[Experimental Matrix Planner] -->|Cartesian Grid| COLAB[Google Colab Execution Workflows]
    end
```

---

## 4. Newly Planned Subpackages in `component3/src/`

1. `component3/src/datasets/`:
   - `dataset_registry.py`: Formal dataset catalog supporting `uci_adult`, `telco_churn`, `home_credit`, `ieee_cis`, and `bank_marketing`.
2. `component3/src/preprocessing/`:
   - `tabular_preprocessor.py`: Implements fit/transform with three-tier representation, unseen category safety, and serialization.
3. `component3/src/splitting/`:
   - `splitter.py`: Stratified, temporal, and group-aware partitioning with manifest logging and leakage assertions.
4. `component3/src/baselines/`:
   - `baseline_trainer.py`: Unified trainer for XGBoost, Random Forest, and LightGBM with metadata tracking.
   - `model_adapter.py`: Adapter for wrapping pre-trained baseline models without retraining.
5. `component3/src/drift/`:
   - `drift_config.py`: Standardized configuration for 8 drift categories, separating injected strength, measured severity, and model degradation.
6. `component3/src/contracts/prescription_contract.py`:
   - Standardized schema definitions (Sections A–E) and pre-decision feature whitelist enforcement.
7. `component3/src/experiments/matrix_planner.py`:
   - Full matrix generation (up to 4,320 combinations), deduplication, eligibility validation, runtime budgeting, and stratified sampling.
8. `component3/colab/`:
   - Standalone, path-agnostic Colab execution scripts and notebooks.

---

## 5. Audit Conclusion & Progression to Task 2

The existing Component 3 core architecture is sound, modular, and 100% backward-compatible. All existing 435 unit tests can be safely preserved while adding the generalized multi-dataset foundation. Proceeding to **Task 2: Multi-Dataset Registry**.
