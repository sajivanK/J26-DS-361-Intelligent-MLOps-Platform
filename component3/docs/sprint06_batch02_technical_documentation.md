# SPRINT 06 — BATCH 2: TECHNICAL DOCUMENTATION AND RESEARCH SPECIFICATION

**Project:** J26-DS-361 — Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan`  
**Working Directory:** `component3/`  
**Author:** Senior Machine Learning Engineer, MLOps Architect & Research Scientist  

---

## 1. Executive Overview

Sprint 06 Batch 2 upgrades Component 3 from a single-dataset prototype (UCI Adult) into a scientifically rigorous, multi-dataset, multi-generator experimental platform with end-to-end Google Colab readiness.

Key milestones delivered in Batch 2:
1. **Multi-Dataset Architecture:** Verified schema, SHA-256 integrity, leakage detection, and automated acquisition for 5 candidate benchmarks: UCI Adult, Telco Customer Churn, Bank Marketing, Home Credit Default Risk, and IEEE-CIS Fraud Detection.
2. **Standardized Preprocessor Ecosystem:** Preprocessors supporting three discrete representations: Raw Tabular, Generator-Compatible, and Baseline-Model-Compatible, strictly fitted on training splits only to prevent data leakage.
3. **Controlled Drift Injectors:** 8 canonical tabular drift categories implemented with empirical 2-sample Kolmogorov-Smirnov (KS) and Total Variation Distance (TVD) divergence measurements, slice degradation tracking, and C1/C2 diagnostic manifest generation.
4. **Three-Generator Integration:** Complete support for CTGAN, TVAE, and TabDDPM (PyTorch Denoising Diffusion Probabilistic Model) under the unified `GeneratorPluginManager`.
5. **Full Sprint 05 Validation Reuse:** Structural, target coverage, empirical fidelity, differential privacy (Distance-to-Closest-Record), and offline model utility validation evaluated against the strict Sprint 05 Acceptance Policy (`APPROVED`, `REVIEW`, `REJECTED`) and zero-leakage prescription contract serialization.
6. **Atomic Checkpointing & Pilot Execution:** Verified atomic checkpointing and resume logic (`pilot_checkpoint.json`) and successful execution of the representative 12-experiment pilot suite across UCI Adult, Telco Churn, XGBoost, Random Forest, CTGAN, TVAE, and TabDDPM with 0 failures.
7. **Google Colab Notebook Suite:** 4 production notebooks updated with GPU acceleration, Google Drive persistence, and explicit execution status notices.

---

## 2. Dataset Registry and Acquisition Status

| Dataset Identifier | Task Type | Features (Num / Cat) | Target | License & Source | Current Local Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`uci_adult`** | Binary Classification | 12 (5 num / 7 cat) | `income` (<=50K / >50K) | CC BY 4.0 (UCI ML Repository) | **VERIFIED** (Shared local asset) |
| **`telco_churn`** | Binary Classification | 19 (3 num / 16 cat) | `Churn` (No / Yes) | Apache 2.0 (IBM Open Data) | **VERIFIED** (Acquired & Checked) |
| **`bank_marketing`** | Binary Classification | 19 (7 num / 12 cat) | `y` (no / yes) | CC BY 4.0 (UCI ML Repository) | **VERIFIED** (Acquired & Checked) |
| **`home_credit`** | Binary Classification | 23 (12 num / 11 cat) | `TARGET` (0 / 1) | Kaggle Competition License | **PENDING_DOWNLOAD** (Manual/API Token) |
| **`ieee_cis`** | Binary Classification | 45 (25 num / 20 cat) | `isFraud` (0 / 1) | Kaggle / Vesta Research License | **PENDING_DOWNLOAD** (Manual/API Token) |

### Integrity & Schema Verification Rules
- `DatasetVerifier` validates SHA-256 checksums, identifies duplicate rows, verifies target class balance, and checks for prohibited leakage columns (e.g., `customerID` in Telco, `duration` in Bank Marketing, `SK_ID_CURR` in Home Credit, `TransactionID` in IEEE-CIS).
- Leakage columns are strictly excluded from baseline feature spaces and generator representations.

---

## 3. Dataset-Specific Preprocessing Architecture

Each dataset is paired with a fitted `TabularDataPreprocessor` instantiated via `get_preprocessor_for_dataset(dataset_id)`:
1. **Raw Tabular Representation:** Complete DataFrame loaded from disk, with missing string representations (`"?"`, `""`, `"NA"`, `"null"`) unified to `np.nan`.
2. **Generator-Compatible Representation:** Discrete categorical columns preserved for conditional GAN/VAE categorical sampling, numeric columns imputed using training-set medians, and dropped/leakage columns excluded.
3. **Baseline-Model-Compatible Representation:** Imputed, scaled (StandardScaler), and one-hot / ordinal encoded numeric matrix (`X: np.ndarray`, `y: np.ndarray`) aligned with the baseline classifier family.

> **Strict Scientific Rule:** Preprocessing transformations are fitted exclusively on the `train` partition. Transformations are applied to `val` and `test` without parameter updates, preventing data leakage.

---

## 4. Controlled Drift Injection Engine

The `DriftInjector` simulates operational distribution shifts across 8 canonical drift categories:
1. **Numerical Distribution Shift:** Additive mean shifts $(\mu + \delta \cdot \sigma)$ and noise scaling applied to numerical features.
2. **Categorical Distribution Shift:** Frequency redistribution or dominant category replacement on categorical features.
3. **Mixed-Feature Shift:** Coordinated shifts across correlated numerical and categorical feature subsets.
4. **Missingness Shift:** Systematic injection of missing values (`np.nan`) simulating sensor or API dropouts.
5. **Feature Correlation Shift:** Linear combination perturbation shifting pairwise correlation structure.
6. **Class-Prior Shift:** Resampling of the target variable changing the base positive rate $P(Y)$ while preserving feature conditional distributions $P(X|Y)$.
7. **Covariate Shift:** Input distribution shift $P(X)$ restricted to a localized feature subpopulation.
8. **Genuine Concept Drift:** Controlled conditional distribution shift $P(Y|X)$ on verified ground-truth labels.

### Empirical Metrics
- **Two-Sample Kolmogorov-Smirnov Test:** $D = \sup_x |F_1(x) - F_2(x)|$ computed per numerical feature.
- **Total Variation Distance (TVD):** $\delta(P, Q) = \frac{1}{2}\sum |P_i - Q_i|$ computed per categorical feature.
- **Degradation Quantification:** Overall $F_1$ drop ($\Delta F_{1,\text{overall}}$) and affected-slice $F_1$ drop ($\Delta F_{1,\text{slice}}$) measured against the frozen baseline classifier.

---

## 5. Complete Three-Generator Repair Engine

Component 3 integrates all three generator plugins under `GeneratorPluginManager`:
1. **CTGAN:** Conditional Generative Adversarial Network with mode-specific normalization and pacGAN discriminator.
2. **TVAE:** Variational Autoencoder utilizing Gumbel-Softmax reparameterization for discrete columns and Gaussian latent modeling.
3. **TabDDPM:** Denoising Diffusion Probabilistic Model for tabular data, using a 100-step Gaussian forward diffusion process and an MLP denoiser with sinusoidal time embeddings.

### Failure Handling & Scientific Integrity
- If any generator encounters out-of-memory or convergence errors, the failure is recorded explicitly in the experiment record (`structural_status = "FAIL"`, `fidelity_score = 0.0`, `package_status = "REJECTED"`).
- **No Silent Substitutions:** TabDDPM is never silently replaced with CTGAN or TVAE.

---

## 6. Sprint 05 Validation Engine & Governance Policy

Every repair candidate passes through the complete multi-stage validation engine:
1. **Structural Validation:** Schema column preservation, dtype compliance, and missing value ratio verification.
2. **Target Coverage:** Slice support verification ensuring sufficient synthetic samples cover the diagnosed slice.
3. **Empirical Fidelity:** Mean KS statistic $\le 0.35$ and empirical fidelity score $\ge 0.65$.
4. **Differential Privacy:** Distance-to-Closest-Record (DCR $P_{05} \ge 0.05$) and exact duplicate detection.
5. **Offline Model Utility:** Downstream retrained classification performance measuring $\Delta F_{1,\text{slice}}$ and $\Delta F_{1,\text{overall}}$.
6. **Acceptance Policy:** Packages are assigned `APPROVED`, `REVIEW`, or `REJECTED`, with `prescriptive_viability_label` tracking research promise without relaxing production standards.

---

## 7. Pilot Experiment Results & Verification

A representative 12-experiment pilot was executed locally on CPU:
- **Matrix:** 2 Datasets (`uci_adult`, `telco_churn`) $\times$ 2 Baseline Models (`xgboost`, `random_forest`) $\times$ 3 Generators (`ctgan`, `tvae`, `tabddpm`).
- **Execution Summary:** 12 executed, 0 skipped initially, 0 failed. Total runtime: **9.18 seconds**.
- **Resume Test:** Re-running the identical pilot successfully loaded and skipped all 12 experiments in **0.00 seconds**, proving atomic checkpointing and resume integrity.
- **Panel Figures:** 4 publication-quality PNG figures generated in `component3/results/figures/`:
  1. `pilot_runtime_by_generator.png`
  2. `pilot_fidelity_comparison.png`
  3. `pilot_slice_f1_recovery.png`
  4. `pilot_acceptance_decisions.png`

---

## 8. Google Colab Execution Instructions

For full-scale high-throughput training across large sample sizes ($n \ge 1000$) and higher epochs ($E \ge 50$), use the provided Google Colab notebooks:

1. Open **Google Colab** (https://colab.research.google.com).
2. Upload the desired notebook from `component3/colab/`:
   - `01_colab_baseline_training.ipynb`: Multi-dataset baseline training.
   - `02_colab_drift_experiments.ipynb`: 8-category drift injection.
   - `03_colab_synthetic_repair.ipynb`: 3-generator repair with Sprint 05 validation.
   - `04_colab_prescription_training.ipynb`: Meta-learning prescription engine training.
3. Navigate to **Runtime > Change runtime type** and select **T4 GPU** (or V100/A100 if available).
4. Connect Google Drive when prompted; results will automatically persist into `/content/drive/MyDrive/J26-DS-361-MLOps/component3/` with atomic checkpointing.
5. Download completed JSON records from Google Drive into `component3/results/repair_experiments/`.
