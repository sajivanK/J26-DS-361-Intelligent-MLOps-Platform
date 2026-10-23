# SPRINT 06 — BATCH 3: TECHNICAL DOCUMENTATION AND RESEARCH SPECIFICATION

**Project:** J26-DS-361 — Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan`  
**Working Directory:** `component3/`  
**Execution Environment:** Local VS Code Jupyter Kernel (`c3-mlops-venv`, Python 3.11, Windows Workstation)  
**Author:** Senior Machine Learning Engineer, MLOps Architect & Research Scientist  

---

## 1. Executive Summary

Sprint 06 Batch 3 implements, executes, and validates controlled data distribution shift simulations across the multi-dataset tabular benchmark suite established in Batches 1 and 2. 

Key milestones delivered:
1. **Calibrated 24-Scenario Drift Matrix:** 8 canonical drift categories across 3 benchmark datasets (`uci_adult_full`, `telco_churn`, `bank_marketing`), resulting in 24 distinct scenarios.
2. **Comprehensive 72-Evaluation Matrix:** 3 baseline model families (`xgboost`, `random_forest`, `lightgbm`) evaluated against all 24 drift scenarios on duplicate-safe validation partitions (`val_indices`).
3. **Frozen Concept-Drift Thresholds:** Explicit conditioning thresholds derived exclusively from development (training) data and permanently frozen in scenario configurations to guarantee zero evaluation leakage.
4. **Clarified Scientific Terminology:** Precise formulation of concept drift as controlled synthetic label-inversion simulations rather than natural observational shifts.
5. **Partition Protocol for Future Repair:** A binding 6-point evaluation protocol to protect future synthetic data generation and repair evaluation in Notebook 04 and beyond.

---

## 2. Frozen Scenario Thresholds and Provenance

To guarantee mathematical reproducibility across evaluation and test sets, conditioning thresholds and perturbation parameters are derived strictly from development (training) partitions (`train_indices`) and frozen into configuration metadata (`scenario_version = "1.0.0"`).

### 2.1 Concept-Drift Conditioning Thresholds

| Scenario ID | Dataset | Target Feature | Conditioning Rule | Frozen Threshold | Derivation Source | Target Dtype Preserved |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ADULT_CONCEPT` | `uci_adult_full` | `hours_per_week` | `hours_per_week > 40.0` | `40.0` | Training Partition Median | `object` (`<=50K`, `>50K`) |
| `TELCO_CONCEPT` | `telco_churn` | `tenure` | `tenure > 29.0` | `29.0` | Training Partition Median | `object` (`No`, `Yes`) |
| `BANK_CONCEPT` | `bank_marketing` | `euribor3m` | `euribor3m > 4.857` | `4.857` | Training Partition Median | `int64` (`0`, `1`) |

### 2.2 Numerical and Covariate Shift Standard Deviations

To prevent runtime data dependency during perturbation scaling, feature standard deviations are similarly frozen from training partitions:

| Scenario ID | Dataset | Target Feature | Frozen Feature $\sigma$ | Derivation Source |
| :--- | :--- | :--- | :--- | :--- |
| `ADULT_NUM` | `uci_adult_full` | `age` | `13.7000` | Training Partition Std |
| `ADULT_COV` | `uci_adult_full` | `education_num` | `2.5645` | Training Partition Std |
| `TELCO_NUM` | `telco_churn` | `MonthlyCharges` | `30.0196` | Training Partition Std |
| `TELCO_COV` | `telco_churn` | `tenure` | `24.6110` | Training Partition Std |
| `BANK_NUM` | `bank_marketing` | `euribor3m` | `1.7340` | Training Partition Std |
| `BANK_COV` | `bank_marketing` | `euribor3m` | `1.7340` | Training Partition Std |

---

## 3. Scientific Terminology and Concept Drift Formulation

### 3.1 Controlled Synthetic Label-Inversion Simulations
Concept-drift scenarios (`ADULT_CONCEPT`, `TELCO_CONCEPT`, `BANK_CONCEPT`) are designed and reported as **controlled synthetic label-inversion simulations**. 
- They do **not** claim to reflect naturally occurring, historical, or real-world concept shifts in UCI Adult, Telco Churn, or Bank Marketing.
- Rather, they represent mathematically controlled interventions modifying the conditional distribution $P(Y|X) \to P_{\text{sim}}(Y|X)$ on specific subpopulation slices to evaluate downstream synthetic repair feasibility under known ground-truth changes.

### 3.2 Simulated Post-Drift vs. Observed Real-World Labels
In real-world deployment, when the target concept $P(Y|X)$ drifts, historical labels are unavailable or invalid without new annotation. In our simulation:
- Ground-truth post-drift labels $Y_{\text{sim}}$ are explicitly generated using the frozen subpopulation policy to measure model collapse.
- In scenarios categorized as `STRESS_TEST_UNCERTAIN` (feature perturbations without known target relationships), original labels are not treated as true post-drift outcomes; instead, model behavior is quantified via prediction flip rate $P(\hat{Y}_{\text{drift}} \ne \hat{Y}_{\text{orig}})$.

### 3.3 Calibration Data Requirement for Future Repair
Concept-drift repair cannot occur purely through unsupervised feature modeling. Repairing a model collapsed by concept drift fundamentally requires access to a limited, separately allocated set of **post-drift labeled calibration records** ($D_{\text{calib}}$) to learn the new conditional mapping $P_{\text{new}}(Y|X)$.

---

## 4. Partition Protocol for Future Repair Evaluation (Notebook 04+)

To ensure absolute scientific integrity in Notebook 04 and subsequent repair experiments, the following protocol is strictly mandated:

1. **Partition Isolation:** Generator training (CTGAN, TVAE, TabDDPM) and repair model training must never consume evaluation records (`val_indices` or `test_indices`).
2. **Calibration Sample Separation:** Post-drift calibration records used to guide synthetic generation or model fine-tuning must come from a dedicated calibration partition, completely disjoint from the final repair evaluation samples.
3. **Zero Test-Guided Selection:** Scenario selection, hyperparameter tuning, and repair strategy decisions must never be informed by final test partition outcomes.
4. **Frozen Scenario Definitions:** All drift definitions, slice rules, and conditioning thresholds must remain frozen prior to evaluating shifted test data.
5. **Preservation of Untouched Test Partition:** The untouched original test partition must be evaluated in parallel with the shifted test partition to verify that repairing shifted slices does not degrade performance on unshifted distributions (catastrophic forgetting prevention).
6. **Source-Record Identity Tracking:** Global row indices and split manifests must track all records to prevent duplicate overlap or data leakage during synthetic resampling.

---

## 5. Summary of Empirical Results

Across 72 evaluations (24 scenarios $\times$ 3 models):
- **Class Prior Shift:** Elevated positive-class $F_1$ across all models (e.g., Adult XGBoost $+0.1601$, Telco $+0.1804$, Bank $+0.2721$) due to mathematical base-rate expansion ($F_1 = \frac{2TP}{2TP + FP + FN}$).
- **Concept Drift Collapse:** Triggered substantial model performance drops (Adult XGBoost $\Delta F_1 = -0.1504$, Telco $\Delta F_1 = -0.2599$, Bank $\Delta F_1 = -0.1706$), demonstrating severe degradation when $P(Y|X)$ changes.
- **Prediction Sensitivity:** Highest prediction flip rates were observed under feature correlation shifts (Adult flip rate $\approx 0.32$, Telco $\approx 0.28$) and high-divergence numerical shifts.
- **Resource Footprint:** Sequential execution completed in $< 30$ seconds CPU time with peak memory utilization $< 1.2$ GB RAM.
