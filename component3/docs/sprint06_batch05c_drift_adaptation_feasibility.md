# Post-Drift Adaptation Feasibility and Scientific Experiment Design

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Sprint:** Sprint 06 — Batch 5C  
**Branch:** `c3-mathusan`  
**Milestone:** Feasibility Study, Label-Provenance Audit, and Defensible Experiment Design  
**Author:** Component 3 Research Lead & MLOps Architecture Team  
**Status:** DESIGN AND FEASIBILITY ONLY (No model training or retraining performed)  

---

## 1. Executive Summary & Context

The primary goal of Component 3 within the J26-DS-361 Intelligent MLOps Platform is to generate high-fidelity, privacy-preserving, and structurally valid synthetic tabular data to repair machine learning models degraded by distribution shifts. Following the completion of the generator pilot (Notebook 04, commit `0f488df`) and the clean minority-class augmentation experiment (Notebook 05 Track A, commit `9a3b2b9`), the research initiative must address **post-drift model adaptation**.

In Batch 5A, preliminary investigation revealed a fundamental scientific problem: **generators trained on historical pre-drift positive records cannot represent shifted concept distributions ($P_{\text{target}}(Y \mid X) \neq P_{\text{source}}(Y \mid X)$)**. Naively drawing samples from pre-drift generators to "repair" a drifted classifier merely reinforces outdated decision boundaries. Consequently, Track B concept-drift adaptation was formally marked as **BLOCKED / PREREQUISITE-GATED**.

This document presents the **Sprint 06 — Batch 5C comprehensive feasibility study and scientific protocol design**. It rigorously determines whether existing datasets, split architectures, drift simulators, validation pipelines, and generator implementations can support an unassailable post-drift adaptation experiment.

### Key Conclusions:
1. **Label-Validity Boundary:** Only two drift categories possess mathematically valid evaluation ground truth: authentic subpopulation covariate shift (`VERIFIED_OBSERVATIONAL_GROUND_TRUTH`) and controlled synthetic concept drift (`SYNTHETIC_CONCEPT_TRUTH`). Pure feature perturbations where historical labels are naively retained are categorized as `STRESS_TEST_UNCERTAIN` and are scientifically invalid for supervised model adaptation.
2. **Recommended Benchmark Domain:** **Telco Customer Churn** is the optimal dataset for the primary adaptation experiment due to domain interpretability, existing preprocessors, fast CPU-safe training, and established domain plausibility constraints.
3. **Partition Isolation Architecture:** To eliminate data snooping without touching the locked canonical final test set ($N=1,056$), we define an **Experimental 4-Way Development Partitioning Scheme** carved strictly from the permitted historical development scope ($N=4,931$).
4. **Mandatory Authentic Control:** Any claim of synthetic repair value must be benchmarked against an **Authentic-Only Adaptation Control** ($A_1$) retrained on the exact same budget of authentic adaptation records ($D_{\text{adapt}}$). Synthetic augmentation ($A_4$) is only scientifically beneficial if $F1(A_4) > F1(A_1)$.
5. **Final Recommendation:** **CONDITIONAL GO** for a 7-stage staged experiment, contingent on satisfying 5 explicit architectural prerequisites prior to any model training.

---

## 2. Current Implementation Audit

A systematic audit of the `component3/` repository was conducted to identify reusable assets and missing prerequisites.

| Subsystem / Artifact | Current File Location | Existing Capabilities | Reusability Status | Identified Gaps / Missing Prerequisites |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Registry & Acquisition** | `component3/src/datasets/dataset_registry.py`<br>`component3/src/datasets/acquisition.py` | Metadata and automated acquisition for UCI Adult, UCI Adult Full, Telco Churn, and Bank Marketing. | **REUSABLE** | None. Schemas and loaders are deterministic and fully functional. |
| **Split Manifests** | `component3/results/split_manifests/*.json` | Canonical 70/15/15 stratified splits for all three datasets with duplicate-group isolation. | **REUSABLE (Read-Only)** | Does not provide sub-partitioning for post-drift adaptation pools ($D_{\text{adapt}}$ vs $D_{\text{target,eval}}$). Must carve internally without modifying canonical manifests. |
| **Baseline Model Artifacts** | `component3/models/baselines/` | 9 frozen baseline models (Random Forest, XGBoost, LightGBM) across 3 datasets + preprocessors. | **REUSABLE (Frozen Reference)** | Retraining directly on baseline artifacts would destroy historical baseline integrity. Must train separately named experimental models ($M_{\text{source}}$). |
| **Drift Injection Engine** | `component3/src/drift/drift_config.py`<br>`component3/src/drift/drift_injector.py` | 8 canonical drift families registered with KS, TVD, Wasserstein distance, slice extraction, and $\Delta F1$. | **REUSABLE** | Current injector applies shifts globally or via slice rules, but lacks explicit adaptation budget subsampling mechanisms. |
| **Synthetic Generators** | `component3/src/generators/base.py`<br>`ctgan_generator.py`, `tvae_generator.py`, `tabddpm_generator.py` | Standardized `fit()` and `sample()` API for VAE, GAN, and Diffusion architectures. | **REUSABLE** | Checkpoints in `models/` were fitted exclusively on pre-drift data. Post-drift generator fine-tuning or fitting logic is not yet encapsulated. |
| **Candidate Validation Rules** | `component3/src/validation/` (`structural_validator.py`, `fidelity_validator.py`, `privacy_validator.py`, `utility_evaluator.py`, `acceptance_policy.py`) | 4-stage validation pipeline: Tier 1 structural, Tier 2 hard safety, Tier 3 empirical plausibility, Tier 4 privacy/DCR. | **REUSABLE** | Needs explicit adaptation-specific acceptance policies (e.g. testing fidelity against $D_{\text{adapt}}$ rather than $D_{\text{source}}$). |
| **Notebook 03 (Drift Sim)** | `component3/notebooks/03_drift_simulation.ipynb` | Empirical degradation catalog across 8 drift families x 3 datasets x 3 models ($N=72$ combinations). | **REUSABLE (Benchmark)** | Evaluated degradation on full validation partitions; did not perform adaptation or repair. |
| **Notebook 04 (Gen Pilot)** | `component3/notebooks/04_synthetic_generator_pilot.ipynb` | Pilot training of CTGAN, TVAE, TabDDPM on pre-drift positive records; DCR privacy auditing. | **REUSABLE (Benchmark)** | Proved pre-drift generator feasibility, but established that pre-drift models cannot repair post-drift concept shift. |
| **Notebook 05 (Augmentation)** | `component3/notebooks/05_synthetic_augmentation_evaluation.ipynb` | Track A minority augmentation on clean data ($K=200$); rigorous bootstrap CIs and permutation tests. | **REUSABLE (Methodology)** | Proved empirical statistical rigor (bootstrap and permutation testing framework). Track B remained blocked. |
| **Component Contracts** | `component3/src/contracts/prescription_contract.py`<br>`safe_slice_evaluator.py` | AST slice evaluation without `eval()`; strict `PRE_DECISION_FEATURE_WHITELIST` preventing leakage. | **REUSABLE** | Contract fields lack explicit tracking for adaptation sample size ($N_{\text{adapt}}$) and target-domain adaptation metrics. |

---

## 3. Label-Provenance Assessment

The validity of any supervised learning evaluation rests entirely on the integrity of the ground-truth target labels ($Y$). In drift adaptation research, naive feature perturbation often corrupts this relationship. We establish a formal 4-tier label provenance taxonomy.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      LABEL PROVENANCE TAXONOMY                         │
├──────────────────────────────────┬─────────────────────────────────────┤
│ Category                         │ Scientific Validity for Adaptation  │
├──────────────────────────────────┼─────────────────────────────────────┤
│ VERIFIED_RESAMPLED_GROUND_TRUTH  │ HIGH (Authentic observed labels)    │
│ VERIFIED_OBSERVATIONAL_GROUND_TRUTH│ HIGH (Natural subpopulation slice)│
│ SYNTHETIC_CONCEPT_TRUTH          │ CONDITIONAL (Synthetic benchmark)   │
│ TEACHER_PSEUDO_LABEL             │ LOW / COMPARATIVE ONLY (No GT)      │
│ STRESS_TEST_UNCERTAIN            │ INVALID / BLOCKED FOR ADAPTATION    │
└──────────────────────────────────┴─────────────────────────────────────┘
```

### 3.1 Taxonomy Definitions

#### 1. `VERIFIED_RESAMPLED_GROUND_TRUTH` & `VERIFIED_OBSERVATIONAL_GROUND_TRUTH`
* **Provenance:** Authentic labels collected from genuine business outcomes (e.g., actual customer cancellations or actual census earnings).
* **Generation Mechanism:** Created by stratified subsampling or natural subpopulation filtering (e.g., evaluating strictly on `Contract == 'Month-to-month'`). The underlying $(X, Y)$ joint relationship is purely observational and unmanipulated.
* **Feature vs. Label Changes:** Features and labels remain authentic; only the sampling probability $P(X)$ changes.
* **Adaptation Role:** Fully permitted for generator training, model retraining, and independent target evaluation.

#### 2. `SYNTHETIC_CONCEPT_TRUTH`
* **Provenance:** Programmatically assigned counterfactual labels generated by a mathematically deterministic concept shift function:
  $$Y_{\text{target}} = f_{\text{concept}}(X, Y_{\text{source}})$$
  Example: In `TELCO_CONCEPT`, customers with $\text{tenure} > 29.0$ have their churn propensity inverted under a frozen development threshold.
* **Feature vs. Label Changes:** Feature vectors $X$ remain authentic; target labels $Y$ are explicitly reassigned.
* **Scientific Boundaries:** Ground truth is known *by mathematical construction*. It represents a valid, reproducible synthetic benchmark / stress test. However, **it must NEVER be reported as real-world customer behavior**.
* **Adaptation Role:** Permitted for synthetic benchmark evaluation if and only if:
  1. Labeled adaptation records $D_{\text{adapt}}$ are drawn from this distribution.
  2. Generators are trained on $D_{\text{adapt}}$.
  3. The synthetic nature of the concept is explicitly disclosed.

#### 3. `STRESS_TEST_UNCERTAIN`
* **Provenance:** Unverified synthetic labels. Features are perturbed ($X' = X + \delta$), but historical labels are naively retained ($Y' = Y$).
* **Scientific Flaw:** If a customer's monthly bill is synthetically inflated from $30 to $120, their actual real-world probability of churn would drastically increase. Retaining their historical label ($Y=\text{'No'}$) creates an unverified, contradictory $(X', Y)$ pair.
* **Scientific Status:** Useful solely as an adversarial input stress test on frozen models. **STRICTLY BLOCKED FOR SUPERVISED ADAPTATION EXPERIMENTS.** Training a downstream classifier or generator on unverified labels forces the model to memorize contradictory noise.

#### 4. `TEACHER_PSEUDO_LABEL`
* **Provenance:** Labels inferred by an algorithmic teacher model:
  $$\hat{Y} = \arg\max P_{\text{teacher}}(Y \mid X')$$
* **Scientific Flaw:** Confirmation bias; errors made by the teacher model are amplified during retraining.
* **Scientific Status:** May serve as an exploratory semi-supervised baseline arm, but **CANNOT serve as ground-truth evaluation evidence**.

---

## 4. Eight-Scenario Feasibility Matrix

We evaluate all eight canonical drift families implemented in Component 3 against scientific adaptation criteria, utilizing empirical findings from Notebook 03 ($N=72$ evaluations).

| Drift Family | Representative Scenario ID | Target Features | Label Provenance | Observed Model Degradation (RF) | Feasible Adaptation Pool Available? | Can Generators Learn Pattern? | Scientific Defensibility for Adaptation | Adaptation Feasibility Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Numerical Shift** | `TELCO_NUM` | `MonthlyCharges` | `STRESS_TEST_UNCERTAIN` | $\Delta F1 = -0.0006$<br>$\Delta \text{Acc} = -0.0076$ | No (Labels unverified) | No (Generators cannot infer unverified labels) | Invalid: Feature shift without label adjustment creates contradictory tuples. Degradation negligible. | **BLOCKED / UNSUITABLE** |
| **2. Categorical Shift** | `TELCO_CAT` | `Contract` | `STRESS_TEST_UNCERTAIN` | $\Delta F1 = -0.0003$<br>$\Delta \text{Acc} = -0.0038$ | No (Labels unverified) | No | Invalid: Retained labels unverified; degradation near zero. | **BLOCKED / UNSUITABLE** |
| **3. Mixed Shift** | `TELCO_MIXED` | `MonthlyCharges`, `Contract` | `STRESS_TEST_UNCERTAIN` | $\Delta F1 = +0.0002$<br>$\Delta \text{Acc} = -0.0114$ | No (Labels unverified) | No | Invalid: Retained labels unverified; F1 shows zero degradation. | **BLOCKED / UNSUITABLE** |
| **4. Missingness Shift** | `TELCO_MISS` | `TotalCharges`, `TechSupport` | `VERIFIED_OBSERVATIONAL_MISSINGNESS` | $\Delta F1 = -0.0053$<br>$\Delta \text{Acc} = -0.0019$ | Yes (Authentic records with missing entries) | Yes (Generators handle NaNs via imputer) | Unsuitable: Tree models handle missingness natively. Degradation is statistically indistinguishable from zero. | **UNSUITABLE (No Degradation)** |
| **5. Correlation Shift** | `TELCO_CORR` | `tenure`, `TotalCharges` | `STRESS_TEST_UNCERTAIN` | $\Delta F1 = -0.0051$<br>$\Delta \text{Acc} = -0.0123$ | No (Uncoupled tuples unverified) | No (Synthesizing broken correlation yields garbage) | Invalid: Breaking physical correlation (e.g. tenure=1, TotalCharges=$5000) violates domain laws without valid labels. | **BLOCKED / UNSUITABLE** |
| **6. Class-Prior Shift** | `TELCO_PRIOR` | `Churn` | `VERIFIED_RESAMPLED_GROUND_TRUTH` | $\Delta F1 = +0.1675$<br>$\Delta \text{Acc} = -0.0086$ | Yes (Authentic stratified samples) | Yes (Generators sample at new class ratio) | Paradoxical: Positive F1 *increases* mechanically due to higher positive prevalence. Problem is threshold calibration, not synthetic data repair. | **UNSUITABLE FOR REPAIR (Calibration Task)** |
| **7. Covariate Shift (Feature Perturbation)** | `TELCO_COV` | `tenure` (conditioned on `Month-to-month`) | `STRESS_TEST_UNCERTAIN` | $\Delta F1 = -0.0132$ (RF)<br>$\Delta F1 = -0.0498$ (XGB)<br>$\Delta F1 = -0.0494$ (LGBM) | No (Perturbed features with unverified historical labels) | No (Fitting generators on artificially shifted features with original labels reinforces invalid $(X', Y)$ tuples) | **UNSUITABLE FOR SUPERVISED ADAPTATION**: Modifies `tenure` by adding $+24.611$ months while retaining original churn labels. Valid only as a diagnostic model sensitivity stress test, NOT as observational ground truth. Cannot be used to claim verified recovery. (Note: A true covariate shift benchmark requires an **Authentic Population-Selection Shift** via filtering/reweighting original records without feature mutation). | **BLOCKED / STRESS TEST ONLY** |
| **8. Concept Drift** | `TELCO_CONCEPT` | `tenure` (rule: `tenure > 29.0`) | `SYNTHETIC_CONCEPT_TRUTH` | $\Delta F1 = -0.1957$ (RF)<br>$\Delta F1 = -0.1708$ (XGB)<br>$\Delta \text{Acc} = -0.3305$ (RF) | **YES** (Synthetically labeled target pool) | **YES** (Generators fitted on post-drift labeled sample) | **DEFENSIBLE UNDER DISCLOSED SYNTHETIC BENCHMARK CONDITIONS**: Degradation is massive and unambiguous. Allows clean testing of concept adaptation given post-drift labels. | **RECOMMENDED (PRIMARY BENCHMARK)** |

---

## 5. Dataset Feasibility Comparison

We compare the three benchmark datasets in the repository to determine which can best host the post-drift adaptation experiment.

| Evaluation Criterion | UCI Adult Full (`uci_adult_full`) | Telco Customer Churn (`telco_churn`) | Bank Marketing (`bank_marketing`) |
| :--- | :--- | :--- | :--- |
| **Sample Size & Partitioning** | Total: 48,842 rows.<br>Train: 34,189; Val: 7,326; Test: 7,327.<br>Ample rows for extensive sub-partitioning. | Total: 7,043 rows.<br>Train: 4,931; Val: 1,056; Test: 1,056.<br>Sufficient rows if carved carefully. | Total: 41,188 rows.<br>Train: 28,832; Val: 6,178; Test: 6,178.<br>Ample rows for extensive sub-partitioning. |
| **Feature Dimensionality & Types** | 14 features (6 numeric, 8 categorical). High cardinality in occupation/native-country. | 20 features (3 numeric, 16 categorical, 1 target). Well-balanced feature structure. | 16 features (7 numeric, 9 categorical). Severe macroeconomic skew. |
| **Baseline Model Performance** | RF F1: 0.6774, Acc: 0.8032.<br>XGB F1: 0.7007, Acc: 0.8245. | RF F1: 0.6209, Acc: 0.7595.<br>XGB F1: 0.6111, Acc: 0.7481. | RF F1: 0.4570, Acc: 0.8385.<br>XGB F1: 0.4412, Acc: 0.8253. |
| **Minority Class Proportion** | 23.9% (>50K). Ample positive instances. | 26.5% (Churn=Yes). Well-balanced minority. | 11.3% (y=yes). Extreme minority starvation. |
| **Drift Degradation Under Concept Drift** | $\Delta F1 = -0.1775$ (RF), $\Delta \text{Acc} = -0.1391$. | $\Delta F1 = -0.1957$ (RF), $\Delta \text{Acc} = -0.3305$. | $\Delta F1 = -0.2641$ (RF), $\Delta \text{Acc} = -0.4075$. |
| **Domain Plausibility Constraints** | Generic demographic constraints. Hard to define strict cross-feature rules. | **Rich cross-feature billing logic**:<br>$\text{TotalCharges} \approx \text{tenure} \times \text{MonthlyCharges}$. Validated in Notebook 04/05. | Limited cross-feature rules. Macroeconomic indicators decoupled from individual. |
| **CPU Training Feasibility (Laptop)** | **SLOW**: TabDDPM requires ~45 min, CTGAN ~20 min on CPU. High OOM risk. | **FAST & SAFE**: TVAE: 15s, CTGAN: 45s, TabDDPM: 120s on CPU. Zero OOM risk. | **MODERATE**: TabDDPM requires ~35 min on CPU. |
| **Leakage & Duplicate History** | Known duplicate rows across census collection. Handled via index hashing. | Full duplicate-safe group audit completed in Notebook 01/04. | Known duplicate contact rows. |
| **Overall Dataset Suitability** | **Secondary (Scalability Benchmark)** | **PRIMARY (Recommended Benchmark)** | **Tertiary (Imbalance Benchmark)** |

### Conclusion:
**Telco Customer Churn** is selected as the primary benchmark dataset for the post-drift adaptation experiment. It offers the optimal balance of measurable degradation, verified cross-feature constraints, rapid CPU execution, and robust baseline model stability.

---

## 6. Prospective Adaptation Protocol & Partition Architecture

### 6.1 Fundamental Scientific Rule of Partitioning
To guarantee absolute scientific integrity:
1. **The Canonical Test Set ($N=1,056$) remains strictly locked, untouched, and unread.** It must never be used for adaptation, model selection, or hyperparameter tuning.
2. **The Canonical Validation Set ($N=1,056$) is preserved** as the independent reference partition for historical clean baseline evaluations.
3. **All adaptation records ($D_{\text{adapt}}$) and target evaluation records ($D_{\text{target,eval}}$) must be carved exclusively from the permitted historical development partition ($N=4,931$).**

### 6.2 Proposed Experimental 4-Way Development Partitioning Scheme

We partition the 4,931 permitted development records into four mutually exclusive, group-aware subsets:

```
```
[Permitted Historical Development Partition: N = 4,931]
   │
   ├── [D_source,train: N = 3,452] (70.01% of Dev)
   │     └── Used to train the Experimental Source Model (M_source)
   │
   ├── [D_source,eval: N = 493] (10.00% of Dev)
   │     └── Evaluates Source-Domain Retention & Catastrophic Forgetting
   │
   ├── [D_target,adapt_pool: N = 493] (10.00% of Dev) ──► Subject to Drift Injection
   │     └── Subsampled into Budgeted Adaptation Sets (D_adapt: N_adapt in {50, 100, 200})
   │           ├── Fits Post-Drift Synthetic Generators (G_target)
   │           └── Retrains Adapted Downstream Models (A1-A6)
   │
   └── [D_target,eval: N = 493] (10.00% of Dev) ──► Subject to Drift Injection
         └── STRICTLY HELD-OUT Target Evaluation Partition
               ├── NEVER enters generator training
               ├── NEVER enters model retraining
               ├── NEVER used for threshold tuning
               └── Measures Final Adapted Model Recovery
```

### 6.3 Exact Roles of Data Partitions

| Partition Identifier | Sample Size ($N$) | Drift Status | Ground Truth Source | Operational Role in Experiment | Prohibited Uses |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **$D_{\text{source,train}}$** | 3,452 | Pre-Drift (Clean) | `VERIFIED_OBSERVATIONAL_GROUND_TRUTH` | Fits experimental source model $M_{\text{source}}$. Baseline training pool for all adaptation arms. | Cannot be evaluated as post-drift target data. |
| **$D_{\text{source,eval}}$** | 493 | Pre-Drift (Clean) | `VERIFIED_OBSERVATIONAL_GROUND_TRUTH` | Independent source-domain evaluation. Tests whether adaptation causes catastrophic forgetting. | Cannot enter generator training or model retraining. |
| **$D_{\text{target,adapt}}$** | 50, 100, 200 | Post-Drift | `SYNTHETIC_CONCEPT_TRUTH` or `VERIFIED_OBSERVATIONAL_GROUND_TRUTH` | **The sole permitted post-drift data**. Used to fit post-drift generators and retrain adapted models. | Cannot be used for final performance reporting. |
| **$D_{\text{target,eval}}$** | 493 | Post-Drift | `SYNTHETIC_CONCEPT_TRUTH` or `VERIFIED_OBSERVATIONAL_GROUND_TRUTH` | **Independent post-drift target evaluation**. Evaluates unadapted baseline $A_0$ and adapted models $A_1$–$A_6$. | **STRICTLY PROHIBITED** from generator training, model retraining, or hyperparameter selection. |
| **Canonical Val ($N=1,056$)** | 1,056 | Clean | `VERIFIED_OBSERVATIONAL_GROUND_TRUTH` | Preserved for cross-experiment consistency with Sprint 06 Batches 5A/5B. | Not used in adaptation protocol loop. |
| **Canonical Test ($N=1,056$)** | 1,056 | Locked | Locked | Untouched. Reserved for final platform verification. | **ZERO ACCESS PERMITTED.** |

### 6.4 Preservation of Historical Baseline Models
The nine historical baseline models located in `component3/models/baselines/` (e.g. `telco_churn_random_forest.joblib`) were fitted on the full 4,931 training partition. To prevent any destructive modification or version confusion:
* The nine baseline models will remain **strictly frozen and unmodified**.
* The adaptation experiment will train a separately serialized model: `telco_rf_experimental_source.joblib`, fitted exclusively on $D_{\text{source,train}}$ ($N=3,452$).

---

## 7. Comparison-Arm Architecture

To isolate the genuine contribution of synthetic data repair from confounding factors (such as authentic adaptation sample size, sample duplication, and filtering strategy), the experiment defines seven strictly matched comparison arms.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       COMPARISON ARMS ARCHITECTURE                         │
├────┬─────────────────────────────┬────────────────────────┬───────────────┤
│ Arm│ Description                 │ Training Composition   │ Research Role │
├────┼─────────────────────────────┼────────────────────────┼───────────────┤
│ A0 │ Frozen Source Model         │ D_source,train only    │ Degradation   │
│ A1 │ Authentic-Only Adaptation   │ + D_adapt (authentic)  │ True Control  │
│ A2 │ Authentic ROS / Reweighting │ + D_adapt oversampled  │ Weight Control│
│ A3 │ Random Synthetic Aug        │ + D_adapt + D_syn,rand │ Volume Control│
│ A4 │ Quality-Screened Synthetic  │ + D_adapt + D_syn,qual │ Main Synthetic│
│ A5 │ Slice-Targeted Synthetic    │ + D_adapt + D_syn,slice│ Slice Focus   │
│ A6 │ Alternative Generator Comp  │ CTGAN vs TVAE vs Diff  │ Architecture  │
└────┴─────────────────────────────┴────────────────────────┴───────────────┘
```

### 7.1 Detailed Arm Specifications

* **Arm A0 — Frozen Source Model (No-Repair Control):**
  * Training Data: $D_{\text{source,train}}$ ($N = 3,452$).
  * Adaptation Data: None ($N_{\text{adapt}} = 0$, $K = 0$).
  * Purpose: Establishes post-drift degradation baseline: $F1_{\text{target}}(A0)$.
* **Arm A1 — Authentic-Only Adaptation (Primary Scientific Benchmark):**
  * Training Data: $D_{\text{source,train}} \cup D_{\text{target,adapt}}$ ($N = 3,452 + N_{\text{adapt}}$).
  * Synthetic Data: None ($K = 0$).
  * Purpose: Measures what authentic target labels alone achieve without synthetic data. **Any valid synthetic repair claim must beat Arm A1.**
* **Arm A2 — Authentic Random OverSampling (ROS) / Reweighting Control:**
  * Training Data: $D_{\text{source,train}} \cup D_{\text{target,adapt}} \cup \text{ROS}(D_{\text{target,adapt}}, K)$.
  * Synthetic Data: None ($K$ authentic duplicates added).
  * Purpose: Isolates whether synthetic data provides novel distributional support or merely mimics sample weighting / duplication.
* **Arm A3 — Random Synthetic Augmentation:**
  * Training Data: $D_{\text{source,train}} \cup D_{\text{target,adapt}} \cup D_{\text{syn,random}}(K)$.
  * Synthetic Filtering: Tier 1 structural + Tier 2 hard safety only.
  * Purpose: Measures raw synthetic volume without quality screening.
* **Arm A4 — Quality-Screened Synthetic Augmentation (Primary Engine Arm):**
  * Training Data: $D_{\text{source,train}} \cup D_{\text{target,adapt}} \cup D_{\text{syn,screened}}(K)$.
  * Synthetic Filtering: Tier 1 + Tier 2 + Tier 3 empirical billing plausibility ($\text{TotalCharges} \approx \text{tenure} \times \text{MonthlyCharges}$).
  * Purpose: Tests whether plausibility filtering provides statistically significant lift over random generation ($A4 \text{ vs } A3$).
* **Arm A5 — Slice-Targeted Synthetic Augmentation:**
  * Training Data: $D_{\text{source,train}} \cup D_{\text{target,adapt}} \cup D_{\text{syn,slice}}(K)$.
  * Targeting Rule: Conditioned exclusively on the affected slice discovered from $D_{\text{target,adapt}}$ (e.g. `tenure > 29.0`).
  * Leakage Guard: Slice rule must be determined from pre-decision detector or $D_{\text{target,adapt}}$, NEVER from $D_{\text{target,eval}}$.
  * Purpose: Evaluates localized slice repair versus global distribution repair.
* **Arm A6 — Alternative Generator Architecture Comparison:**
  * Factorial comparison holding $K$, $D_{\text{target,adapt}}$, and screening rules identical across:
    * $A6a$: TVAE Quality-Screened
    * $A6b$: CTGAN Quality-Screened
    * $A6c$: TabDDPM Quality-Screened (Deduplicated)
  * Purpose: Isolates generator architecture (VAE vs. GAN vs. Diffusion).

### 7.2 Strict Confounding Controls
1. **Classifier Hyperparameter Lock:** All retrained models use identical hyperparameters:
   `RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', criterion='gini', min_samples_split=2, n_jobs=2, random_state=42)`.
2. **Fixed Random State:** Random seeds for synthetic sampling and model retraining are paired and held constant.
3. **Strict Sample Budget Parity:** Synthetic budget $K$ and authentic oversampling budget $K$ are identical.

---

## 8. Valid Recovery Metrics & Statistical Testing

### 8.1 Performance Metric Definitions

| Metric Name | Mathematical Definition | Evaluation Scope | Research Purpose |
| :--- | :--- | :--- | :--- |
| **Positive-Class F1** | $\frac{2 \cdot P \cdot R}{P + R}$ for $\text{Churn}=\text{'Yes'}$ | $D_{\text{target,eval}}$ | Primary performance metric for imbalanced minority class. |
| **Macro F1** | $\frac{1}{2} (F1_{\text{pos}} + F1_{\text{neg}})$ | $D_{\text{target,eval}}$ | Evaluates overall balanced classification capability. |
| **Balanced Accuracy** | $\frac{1}{2} (\text{TPR} + \text{TNR})$ | $D_{\text{target,eval}}$ | Threshold-agnostic classification accuracy under skew. |
| **PR-AUC** | Area under Precision-Recall curve | $D_{\text{target,eval}}$ | Evaluates ranking quality across all decision thresholds. |
| **Worst-Slice F1** | $F1(Y_{\text{slice}}, \hat{Y}_{\text{slice}})$ for $X \in S$ | $D_{\text{target,eval}} \cap S$ | Evaluates localized performance on diagnosed vulnerable subpopulation ($N_{\text{slice}} \ge 30$). |
| **Source Retention F1** | $F1(Y_{\text{source}}, \hat{Y}_{\text{source}})$ | $D_{\text{source,eval}}$ | Measures catastrophic forgetting on original pre-drift domain. |

### 8.2 Recovery Fractions and Comparative Differences

#### 1. Baseline Drift Degradation Denominator
$$\Delta F1_{\text{deg}} = F1_{\text{source}}(A0) - F1_{\text{target}}(A0)$$
* **Scientific Constraint:** The relative recovery fraction $\mathcal{R}$ is defined **if and only if $\Delta F1_{\text{deg}} \ge 0.05$**. If degradation is negligible, computing a fraction creates an unstable division by a near-zero denominator.

#### 2. Absolute Recovery over No-Repair Control ($A0$)
$$\Delta F1_{\text{no-repair}}(A_i) = F1_{\text{target}}(A_i) - F1_{\text{target}}(A0)$$

#### 3. Absolute Net Synthetic Advantage over Authentic Adaptation ($A1$)
$$\Delta F1_{\text{auth-only}}(A_i) = F1_{\text{target}}(A_i) - F1_{\text{target}}(A1)$$
* **Scientific Assertion:** If $\Delta F1_{\text{auth-only}} \le 0$, the synthetic repair engine provides **zero added value** over standard authentic data collection.

#### 4. Relative Recovery Fraction ($\mathcal{R}$)
$$\mathcal{R}(A_i) = \frac{F1_{\text{target}}(A_i) - F1_{\text{target}}(A0)}{\Delta F1_{\text{deg}}} \times 100\%$$

#### 5. Source-Domain Regression (Catastrophic Forgetting)
$$\Delta F1_{\text{forgetting}}(A_i) = F1_{\text{source,eval}}(A_i) - F1_{\text{source,eval}}(A0)$$
* An adapted model that recovers target-domain performance but severely drops on $D_{\text{source,eval}}$ ($\Delta F1_{\text{forgetting}} < -0.05$) suffers from unacceptable catastrophic forgetting.

### 8.3 Statistical Inference Standards
* **Paired Bootstrap Confidence Intervals:** 1,000 bootstrap resamples computed on aligned prediction vectors from $D_{\text{target,eval}}$ to yield 95% percentile CIs for $F1$ and $\Delta F1$.
* **Paired Permutation Tests:** 1,000 random sign-flip permutations of paired prediction residuals to determine exact one-sided $p$-values.
* **Significance Threshold:** An improvement is declared statistically significant only if $p < 0.05$ and the 95% CI of $\Delta F1$ excludes zero.

---

## 9. Minimum Effective Repair Quantity Design

A critical scientific question is: **how many synthetic records are necessary and sufficient to adapt the model without injecting excessive noise?**

```
Hypothesized Utility vs. Synthetic Budget K:
Performance (F1)
   ^
   │         Optimal K* (e.g. 100-200)
   │             ┌───┐
   │            ┌┘   └┐
   │           ┌┘     └── Mode Collapse / Noise Overfitting
   │          ┌┘
   │   ───────┘ Authentic Control (A1)
   │  ┌┘
   └──┴───────────────────────────────> Synthetic Budget K
      0   50   100   200   500
```

### 9.1 Candidate Budget Exploration
We propose exploring candidate budgets:
$$K \in \{0, 50, 100, 200, 500\}$$
* $K=0$: Unaugmented authentic adaptation ($A1$).
* $K=50$: Conservative repair ($~50\%$ of $N_{\text{adapt}}$).
* $K=100$: Balanced 1:1 synthetic-to-authentic ratio.
* $K=200$: High-volume augmentation (matches Notebook 05 pilot).
* $K=500$: Stress-testing saturation and noise breakdown.

### 9.2 Candidate Selection and Abstention Protocols
1. **Zero Quota Duplication Rule:** If a generator generates 1,000 candidates but only 140 pass Tier 2 safety and Tier 3 plausibility, **the engine MUST NOT duplicate accepted records to meet a budget of $K=200$**. Duplication corrupts sample diversity. The engine must either sample more candidates or report candidate exhaustion.
2. **Abstention Threshold:** If the generator acceptance rate falls below 10%, or if fewer than $K_{\text{min}} = 50$ unique valid records are produced after 5,000 sampling attempts, **Component 3 MUST ABSTAIN** and advise against synthetic retraining.
3. **Selection Protocol:** Optimal budget $K^*$ must be selected using cross-validation within $D_{\text{target,adapt}}$, and then independently evaluated on $D_{\text{target,eval}}$ to prevent budget-tuning leakage.

---

## 10. CPU-Safe and Memory-Bounded Execution Design

The development environment is constrained to a Windows laptop with CPU-only execution and limited RAM. A disciplined execution plan is required.

### 10.1 System Resource Budget
* **Max Active Processes:** 1 Python runner process.
* **Worker Threads:** Bounded at `n_jobs = 2` for scikit-learn models.
* **Generator Fitting:** Strictly sequential. Never fit multiple neural generators concurrently.
* **Batch Sampling:** Generate synthetic candidates in memory-bounded chunks of 500 records.
* **Disk Checkpointing:** Every retrained model, synthetic candidate batch, and evaluation manifest is immediately serialized to disk in `component3/results/adaptation/`.

### 10.2 Seven-Stage Sequential Execution Protocol

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEVEN-STAGE STAGED EXPERIMENT PLAN                   │
├───────┬──────────────────────────────────┬──────────────────────────────┤
│ Stage │ Objective                        │ Termination / Gate Condition │
├───────┼──────────────────────────────────┼──────────────────────────────┤
│ St. 1 │ Partition & Label Verification   │ Zero split overlap verified  │
│ St. 2 │ Frozen Degradation Audit         │ Delta F1_deg >= 0.05 verified│
│ St. 3 │ Authentic Adaptation Control (A1)│ Baseline A1 benchmark logged │
│ St. 4 │ Small Synthetic Pilot (K=50)     │ Pipeline smoke test passes   │
│ St. 5 │ Matched Comparison (A0-A6, K=200)│ Primary research hypothesis  │
│ St. 6 │ Repair Quantity Sweep (K in 50..)│ Identifies optimal K*        │
│ St. 7 │ Independent Scientific Assessment│ Final report & C4 governance │
└───────┴──────────────────────────────────┴──────────────────────────────┘
```

* **Stage 1 — Partition and Label-Validity Verification:** Verify disjoint indices across $D_{\text{source,train}}$, $D_{\text{source,eval}}$, $D_{\text{target,adapt}}$, and $D_{\text{target,eval}}$.
* **Stage 2 — Frozen-Model Degradation Verification:** Verify that $M_{\text{source}}$ suffers measurable degradation ($\Delta F1_{\text{deg}} \ge 0.05$) under the selected drift scenario on $D_{\text{target,eval}}$. If $\Delta F1_{\text{deg}} < 0.05$, halt and report unsuitability.
* **Stage 3 — Authentic-Only Adaptation Control:** Retrain $M_{\text{source}}$ on $D_{\text{source,train}} \cup D_{\text{target,adapt}}$ ($N_{\text{adapt}}=100$). Log authentic benchmark $F1(A1)$.
* **Stage 4 — Small Synthetic Augmentation Pilot:** Generate $K=50$ candidates using TVAE; verify validation filtering, retraining, and prediction pipeline without memory leaks.
* **Stage 5 — Matched Generator and Targeting Comparisons:** Execute Arms $A2$, $A3$, $A4$, $A5$, and $A6$ at fixed $K=200$. Run bootstrap CIs and permutation tests against $A0$ and $A1$.
* **Stage 6 — Repair-Size Evaluation:** Sweep $K \in \{50, 100, 200, 500\}$ to map the repair quantity curve.
* **Stage 7 — Independent Scientific Assessment:** Compile audit metrics, evaluate source retention on $D_{\text{source,eval}}$, check DCR privacy scores, and formulate the final deployment decision.

*(Note: None of these stages are executed during Batch 5C; this plan governs future implementation).*

---

## 11. Component Integration & Shared Contracts

Component 3 operates as the central repair engine within the 4-component platform architecture.

```
┌──────────────────────┐
│     Component 1      │  Drift Metrics: KS, TVD, Wasserstein, Severity Score
│   (Drift Detection)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     Component 2      │  Affected Slice Rule: (e.g. tenure > 29.0)
│   (Root Cause/Slice) │  Slice Support, Diagnosis Stability
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐  Checks Label Provenance & Adaptation Budget
│     Component 3      │  Generates Candidates (TVAE/CTGAN/TabDDPM)
│   (Repair Engine)    │  4-Stage Validation (Structural, Fidelity, Privacy, Utility)
└──────────┬───────────┘  Retrains Downstream Models (A0-A6)
           │
           ▼
┌──────────────────────┐  Evaluates Delta F1_auth-only, Source Retention
│     Component 4      │  Verifies DCR Privacy Gates
│(Deployment/Decision) │  Approves/Abstains Supervised Production Retraining
└──────────────────────┘
```

### 11.1 Interface Contracts and Compatibility
1. **Component 1 Inputs:** Provides scenario ID, detected drift family, statistical divergence metrics, and drift severity tier.
2. **Component 2 Inputs:** Provides safe AST slice rule string (evaluated via restricted AST parser) and affected subpopulation support.
3. **Component 3 Outputs:** Generates structured repair packages, validation logs, retrained candidate models, and prediction audit files.
4. **Component 4 Handoff:** Provides `PrescriptionExperimentRecord` with governance metadata and decision fields (`prescriptive_viability_label`, `repair_improvement_label`).

### 11.2 Backward-Compatible Contract Extension
To natively support post-drift adaptation without breaking existing team contracts, we propose extending `component3/src/contracts/prescription_contract.py` with optional, default-valued fields:

```python
# Backward-compatible extension to PrescriptionExperimentRecord
adaptation_sample_size: int = 0
adaptation_label_provenance: str = "none"
delta_target_f1_vs_no_repair: float = 0.0
delta_target_f1_vs_authentic_only: float = 0.0
source_domain_retention_f1: float = 0.0
```

---

## 12. Scientific Risks and Limitations

1. **Risk of Synthetic Hallucination under Concept Drift:** If adaptation budget $N_{\text{adapt}}$ is very small (e.g. $N_{\text{adapt}}=50$), deep generative models may fail to capture the new concept, generating noisy or mode-collapsed records.
2. **Realism of Programmatic Concept Drift:** `TELCO_CONCEPT` is a deterministic synthetic counterfactual rule. While mathematically clean, real-world concept drift is often stochastic, gradual, and noisy. Results on synthetic concept drift must be qualified as benchmark bounds.
3. **Catastrophic Forgetting Tradeoff:** Adapting to aggressive target drift may degrade performance on the original pre-drift domain ($D_{\text{source,eval}}$). Real-world applications must weigh target adaptation against source retention.
4. **Evaluation Leakage Risks:** Any contamination of $D_{\text{target,eval}}$ into generator training, slice discovery, or hyperparameter selection completely invalidates the research conclusions. The strict partition isolation defined in Section 6 must be programmatically enforced.

---

## 13. Explicit Decision: CONDITIONAL GO

Based on this comprehensive feasibility audit:

### **DECISION: CONDITIONAL GO**

Post-drift adaptation research is **scientifically feasible and methodologically defensible**, subject to strict adherence to five gating conditions.

### Gating Conditions:
1. **Condition 1 (Scenario Restriction):** The primary experiment must evaluate **`TELCO_CONCEPT`** (synthetic benchmark truth with full counterfactual disclosure) or a verified **Authentic Population-Selection Shift** (reweighting/filtering without feature mutation). Historical feature perturbation scenarios like `TELCO_COV` modify features while retaining original labels, are classified as `STRESS_TEST_UNCERTAIN`, and are strictly prohibited from supervised adaptation and claims of verified recovery.
2. **Condition 2 (Partition Quarantine):** The 4-way development partitioning scheme ($D_{\text{source,train}}$, $D_{\text{source,eval}}$, $D_{\text{target,adapt}}$, $D_{\text{target,eval}}$) must be implemented and verified with zero index overlap prior to any generator training.
3. **Condition 3 (Post-Drift Generator Fitting):** Generators must be trained strictly on post-drift adaptation records ($D_{\text{target,adapt}}$). Pre-drift generators cannot be used for concept adaptation.
4. **Condition 4 (Mandatory Authentic Control):** Arm $A1$ (authentic-only adaptation) must be executed and reported alongside all synthetic arms.
5. **Condition 5 (Test Partition Inviolability):** The canonical test set ($N=1,056$) remains locked and unread.

---

## 14. Remaining Prerequisites Before Implementation

Prior to launching implementation or retraining in future sprints, the following technical prerequisites must be completed:

1. **Partition Builder Module:** Create `component3/src/datasets/adaptation_splitter.py` to deterministically carve the 4-way development split with duplicate-group isolation.
2. **Post-Drift Generator Training Harness:** Implement a lightweight script to fit/fine-tune TVAE, CTGAN, and TabDDPM on small adaptation batches ($N_{\text{adapt}} \in \{50, 100, 200\}$).
3. **Experimental Source Model Serializer:** Train and save `telco_rf_experimental_source.joblib` on $D_{\text{source,train}}$ ($N=3,451$) without touching historical baseline models.
4. **Automated Protocol Tests:** Create unit tests verifying partition isolation, label provenance validation, and comparison arm contract validity.
5. **Formal Scientific Peer Review:** Submit this feasibility report for research team review and sign-off.
