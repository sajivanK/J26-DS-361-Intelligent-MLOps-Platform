# SPRINT 06 — BATCH 5F TECHNICAL REPORT
## Experimental Source Model Training and Frozen Baseline Evaluation

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan` | **Active Commit:** `9a3b2b9`  
**Execution Timestamp:** 2026-10-20T11:30:45+05:30  
**Model Name:** `telco_rf_experimental_source`  

---

## 1. Executive Summary

In accordance with the experimental partitioning protocol established in Batch 5E, we have trained and validated the **Experimental Random Forest Source Model** ($M_{\text{source}}$) on the quarantined source-training partition ($D_{\text{source,train}}$, $N=3,452$). This model provides the **frozen, unadapted baseline control ($A0$)** against which subsequent authentic-data adaptation ($A1$) and synthetic-data repair ($A2, A3, A4$) will be rigorously evaluated.

### Key Milestones Achieved:
1. **Zero Data Snooping / Complete Fit Isolation:** The preprocessor and the Random Forest classifier were fitted strictly on $D_{\text{source,train}}$ ($N=3,452$). Neither evaluation partition ($D_{\text{source,eval}}$, $D_{\text{target,eval}}$), adaptation pool ($D_{\text{target,adapt\_pool}}$), nor canonical validation/test partitions influenced preprocessing statistics, encoding dictionaries, or model parameters.
2. **Historical Artifact Inviolability:** Historical baseline models in `component3/models/baselines/` and canonical split manifests in `component3/results/split_manifests/` were cryptographically audited; zero bytes were overwritten or mutated.
3. **Reproducible Source-Domain Performance:** Evaluated on $D_{\text{source,eval}}$ ($N=493$), $M_{\text{source}}$ achieved an observational Positive-class F1 of **0.6075** and ROC-AUC of **0.8255**.
4. **Quantified Concept Degradation:** Evaluated under the frozen simulated `TELCO_CONCEPT` rule (`tenure > 29.0`) on $D_{\text{target,eval}}$ ($N=493$, 220 labels inverted), the frozen model's Positive-class F1 fell to **0.4139** ($\Delta F1 = \mathbf{-0.1936}$), with Balanced Accuracy dropping by $\mathbf{-0.2933}$.
5. **Reload Parity Verified:** Serialized model and preprocessor artifacts in `component3/models/experimental/` were reloaded from disk, demonstrating bitwise identical predictions and probability parity within $10^{-7}$.

---

## 2. Input Provenance and Partition Isolation

Prior to training, input manifests and dataset fingerprints were cryptographically verified and frozen:

| Artifact / Entity | File Path | Verified SHA-256 Hash | Row Count / Population | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Split Manifest** | `component3/results/split_manifests/telco_churn_split_manifest.json` | `e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68` | 4,931 (Train)<br>1,056 (Val)<br>1,056 (Test) | **LOCKED & UNTOUCHED** |
| **Experimental Split Manifest** | `component3/results/adaptation_splits/telco_churn_adaptation_split_manifest.json` | `34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59` | 3,452 (Source Train)<br>493 (Source Eval)<br>493 (Adapt Pool)<br>493 (Target Eval) | **FROZEN REFERENCE** |
| **Leakage Audit Artifact** | `component3/results/adaptation_splits/telco_churn_adaptation_group_overlap_audit.json` | `884d9d6983b1f589ff302749438f93bda84c5d4a2a7d5d90e7f793c0a2008518` | 4,931 rows total<br>23 duplicate groups | **VERIFIED PASS (0 overlap)** |
| **Historical Baseline RF** | `component3/models/baselines/telco_churn_random_forest.joblib` | `69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701` | Baseline Reference | **PRESERVED UNCHANGED** |
| **Historical Baseline Prep** | `component3/models/baselines/telco_churn_preprocessor.joblib` | `a9eb79cd6c519b99a1ef5372d5418afaacb01048498bb46617d1f6e012ed45c0` | Baseline Reference | **PRESERVED UNCHANGED** |

* **Dataset Fingerprint:** `0348ebce7a8c776d` (computed via SHA-256 over canonical training partition values and index order).
* **Row Conservation:** $\sum |D_i| = 3,452 + 493 + 493 + 493 = 4,931$ rows (100% accounted for).
* **Pairwise Index Overlap:** Zero overlapping row indices across all 6 partition pairs.
* **Feature Group Isolation:** All 23 duplicate feature groups remain strictly isolated within $D_{\text{source,train}}$ (21 groups, 47 rows) and $D_{\text{target,adapt\_pool}}$ (2 groups, 4 rows).

---

## 3. Dedicated Experimental Trainer Implementation

The trainer is encapsulated in `component3/src/experiments/train_experimental_source.py`.

### 3.1 Model Hyperparameters
Configured identically to the historical baseline Random Forest:
* `n_estimators`: 100
* `max_depth`: 10
* `class_weight`: `"balanced"` (computes inverse prevalence splits)
* `random_state`: 42
* `n_jobs`: 2 (strictly bounded CPU threads)
* `bootstrap`: True
* `criterion`: `"gini"`

### 3.2 Preprocessor Configuration
Implemented via `TabularDataPreprocessor.from_dataset_metadata(meta, scale_numerical=True)`:
* **Active Numerical Features (3):** `tenure`, `MonthlyCharges`, `TotalCharges` (imputed with median, standardized via `StandardScaler`).
* **Active Categorical Features (16):** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` (imputed with `"missing"`, encoded via `OrdinalEncoder` with unknown token handling).
* **Fitted Population:** Exactly 3,452 rows from $D_{\text{source,train}}$.
* **Computed Class Weights:** `No (0): 0.6806`, `Yes (1): 1.8843` ($3452 / (2 \times 2536)$ and $3452 / (2 \times 916)$).

### 3.3 Training Execution Telemetry
* **Hardware:** Laptop CPU (x86_64)
* **Training Runtime:** **0.36 seconds**
* **Peak Resident Set Memory (RSS):** **160.7 MB**
* **Dependencies:** `scikit-learn==1.9.1`, `numpy==2.4.6`, `pandas==2.2.3`, `joblib==1.6.0`

---

## 4. Source-Domain Evaluation on $D_{\text{source,eval}}$

$D_{\text{source,eval}}$ ($N=493$) evaluates the model under original observational ground truth (pre-drift distribution):

| Metric | Measured Value | Notes / Definition |
| :--- | :---: | :--- |
| **F1-Score (Positive Class)** | **0.6075** | Churn = 'Yes' ($N_{\text{pos}}=131$) |
| **Macro F1-Score** | **0.7208** | Unweighted average across classes |
| **Accuracy** | **0.7667** | Overall classification accuracy |
| **Balanced Accuracy** | **0.7389** | Average recall across classes |
| **Precision (Positive Class)** | **0.5494** | $\text{TP} / (\text{TP} + \text{FP}) = 89 / (89 + 73)$ |
| **Recall (Positive Class)** | **0.6794** | $\text{TP} / (\text{TP} + \text{FN}) = 89 / (89 + 42)$ |
| **ROC-AUC** | **0.8255** | Area under the ROC curve |
| **PR-AUC (Average Precision)**| **0.6225** | Area under Precision-Recall curve |
| **Total Support** | **493** | Observed Ground Truth |
| **Class Distribution** | No: 362 (73.43%)<br>Yes: 131 (26.57%) | Stratified sample matching canonical train |

### Confusion Matrix on $D_{\text{source,eval}}$:
$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 289 & 73 \\ 42 & 89 \end{pmatrix}$$

---

## 5. Target-Domain Frozen Baseline Evaluation on $D_{\text{target,eval}}$

$D_{\text{target,eval}}$ ($N=493$) evaluates the unadapted source model ($A0$) under the simulated `TELCO_CONCEPT` drift rule:
* **Inversion Rule:** For subscribers with $\text{tenure} > 29.0$, invert churn label (`Yes` $\to$ `No`, `No` $\to$ `Yes`). Subscribers with $\text{tenure} \le 29.0$ retain original labels.
* **Transformed Labels Inverted:** Exactly **220 of 493 rows (44.62%)** were inverted.
* **Transformed Class Distribution:** No: 190 (38.54%), Yes: 303 (61.46%).
* **Label Provenance:** Categorized strictly as $\mathbf{SYNTHETIC\_CONCEPT\_TRUTH}$.

| Metric | Measured Value | Degradation vs Source Domain ($\Delta$) |
| :--- | :---: | :---: |
| **F1-Score (Positive Class)** | **0.4139** | $\mathbf{-0.1936}$ (-19.36 pp) |
| **Macro F1-Score** | **0.4198** | $\mathbf{-0.3010}$ (-30.10 pp) |
| **Accuracy** | **0.4199** | $\mathbf{-0.3468}$ (-34.68 pp) |
| **Balanced Accuracy** | **0.4456** | $\mathbf{-0.2933}$ (-29.33 pp) |
| **Precision (Positive Class)** | **0.5459** | $-0.0035$ (-0.35 pp) |
| **Recall (Positive Class)** | **0.3333** | $\mathbf{-0.3461}$ (-34.61 pp) |
| **ROC-AUC** | **0.4063** | $\mathbf{-0.4192}$ (-41.92 pp) |
| **PR-AUC (Average Precision)**| **0.5976** | $-0.0249$ (-2.49 pp) |
| **Total Support** | **493** | Simulated Concept Truth |

### Confusion Matrix on $D_{\text{target,eval}}$ (Frozen Control $A0$):
$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 106 & 84 \\ 202 & 101 \end{pmatrix}$$

---

## 6. Performance Differences and Degradation Analysis

### 6.1 Scientific Explanation of Performance Gap
1. **Mechanism of Degradation:** Under the pre-drift concept, high tenure was strongly correlated with subscription loyalty (low churn). Under `TELCO_CONCEPT`, customers with $\text{tenure} > 29$ months invert their behavior to high churn (e.g., contractual expirations, competitive fiber rollout). Because the frozen source model expects high-tenure customers to remain, it predicts $P(\text{Churn} = \text{'Yes'}) \approx 0.10 - 0.25$, creating **202 False Negatives** out of 303 positive targets (Recall plunges from 67.94% to 33.33%).
2. **Sample Variance vs Concept Drift:** Source evaluation ($D_{\text{source,eval}}$) and target evaluation ($D_{\text{target,eval}}$) utilize disjoint rows ($N=493$ each). A small fraction of the performance gap arises from empirical sample variance across partitions, while the overwhelming majority ($\Delta F1 = -0.1936$, $\Delta \text{Acc} = -0.3468$) is directly driven by the concept inversion.
3. **No Premature Recovery Fraction Calculation:** Because no adapted model ($A1, A2, A3, A4$) has yet been trained, **no recovery fraction is calculated or claimed**. The measured F1 of 0.4139 represents the frozen denominator baseline ($F1_{\text{target,frozen}}$) for subsequent batches.

---

## 7. Artifact Integrity and Reload Parity Audit

The trained artifacts were serialized, reloaded, and audited:
* **Model Artifact:** `component3/models/experimental/telco_rf_experimental_source.joblib`  
  SHA-256: `256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0`
* **Preprocessor Artifact:** `component3/models/experimental/telco_preprocessor_experimental_source.joblib`  
  SHA-256: `a188f0285e6cf07741bbb69be8b3df44eece492e69166c78d92353f1f4d41224`
* **Experimental Manifest:** `component3/results/experimental_models/telco_rf_experimental_source_manifest.json`

### Cryptographic and Numerical Verification:
1. **Prediction Parity:** Exact bitwise equality on all 493 rows of $D_{\text{source,eval}}$ and 493 rows of $D_{\text{target,eval}}$ ($\text{np.array\_equal} = \mathbf{True}$).
2. **Probability Parity:** Maximum absolute difference in predicted class probabilities is $< 10^{-7}$ ($\text{np.allclose} = \mathbf{True}$).
3. **Baseline Model Invariance:** The SHA-256 hashes of `component3/models/baselines/telco_churn_random_forest.joblib` and `telco_churn_preprocessor.joblib` match their pre-experiment hashes identically.

---

## 8. Threat to Validity and Scientific Limitations

1. **Simulated Nature of Concept Shift:** The label inversion rule (`tenure > 29.0`) is deterministic and programmatic. While mathematically clean for benchmarking repair algorithms, real-world concept drift is often stochastic, noisy, and multivariate. Findings must be interpreted as benchmark capabilities under counterfactual shifts, not naturalistic churn drift.
2. **Evaluation Partition Fixedness:** To prevent evaluation contamination, $D_{\text{target,eval}}$ is locked. Hyperparameter tuning on adaptation models must be conducted on internal validation splits carved from $D_{\text{target,adapt\_pool}}$, never by probing $D_{\text{target,eval}}$.
3. **Subpopulation Imbalance Shift:** Inverting labels for $\text{tenure} > 29.0$ shifts the positive class prevalence from 26.6% to 61.5% in the target evaluation partition. An adapted model must recalibrate its decision threshold in addition to updating feature-target mappings.

---

## 9. Decision Recommendation for Authentic-Only Adaptation ($A1$)

$$\mathbf{CONDITIONAL\ GO}$$

**Condition:** Authorized to implement **Arm $A1$ (Authentic-Only Adaptation Control)** using adaptation budgets $B_{50}, B_{100}, B_{200}$ carved from $D_{\text{target,adapt\_pool}}$. 

**Mandatory Constraint:** Under no circumstances may synthetic data ($A2, A3, A4$) be claimed as beneficial unless it demonstrates statistically superior performance over Arm $A1$ on $D_{\text{target,eval}}$ under identical budget constraints.
