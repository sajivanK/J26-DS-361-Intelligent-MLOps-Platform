# SPRINT 06 — BATCH 5G TECHNICAL REPORT
## Labeled Simulated-Target Adaptation Control (Arm A1 Benchmark)

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan` | **Active Commit:** `9a3b2b9`  
**Execution Timestamp:** 2026-10-20T11:46:30+05:30  
**Experimental Scenario:** `TELCO_CONCEPT` (`tenure > 29.0` label inversion)  
**Reference Control Arm:** $A0$ (`telco_rf_experimental_source.joblib`)  

---

## 1. Executive Summary

In accordance with the pre-registered scientific protocol, we implemented and evaluated **Arm $A1$ (Authentic-Only Adaptation Control)** for the frozen `TELCO_CONCEPT` distribution shift. The primary research objective was to establish whether simply pooling small budgets of labeled, authentic target-domain data ($B_{50}, B_{100}, B_{200}$) with pre-drift training data restores classification performance without synthetic data generation.

### Key Empirical Findings:
1. **Inefficacy of Small Unweighted Authentic Adaptation Samples:** Pooling small labeled target batches with $D_{\text{source,train}}$ ($N=3,452$) yielded **no statistically significant improvement** over the frozen unadapted source model ($A0$):
   * $A0$ (Frozen Control, $N=3,452$): Target F1 = **0.4139**
   * $A1\text{-}B50$ ($N=3,502$): Target F1 = **0.4033** ($\Delta F1 = -0.0107$, 95% CI: $[-0.0243, +0.0015]$, $p_{\text{adj}} = 0.3645$)
   * $A1\text{-}B100$ ($N=3,552$): Target F1 = **0.4090** ($\Delta F1 = -0.0049$, 95% CI: $[-0.0200, +0.0104]$, $p_{\text{adj}} = 1.0000$)
   * $A1\text{-}B200$ ($N=3,652$): Target F1 = **0.4141** ($\Delta F1 = +0.0001$, 95% CI: $[-0.0121, +0.0119]$, $p_{\text{adj}} = 1.0000$)
2. **Root Cause Analysis (Pre-Drift Dominance):** In standard unweighted tree ensembles, 3,452 pre-drift records (including 1,719 subscribers with $\text{tenure} > 29$ displaying low churn) completely overwhelm the 25, 48, or 97 flipped adaptation records (representing only 1.4% to 5.8% of the training data). Decision trees continue to prioritize splits consistent with the dominant historical distribution.
3. **Source Retention Preserved (No Catastrophic Forgetting):** Source-domain F1 evaluated on $D_{\text{source,eval}}$ ($N=493$) remained stable across all arms: $A0 = 0.6075$, $A1\text{-}B50 = 0.6233$, $A1\text{-}B100 = 0.6093$, $A1\text{-}B200 = 0.5980$.
4. **Significance for Component 3:** This finding confirms the central thesis of Component 3: **small authentic target samples alone are insufficient for standard classifiers to overcome entrenched concept drift without targeted synthetic density augmentation or adaptive repair mechanisms.**

---

## 2. Pre-Registered Experimental Protocol

The machine-readable protocol was committed to [`batch05g_adaptation_protocol.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/batch05g_adaptation_protocol.json) prior to adaptation model training:

```json
{
  "protocol_name": "Batch_5G_Authentic_Adaptation_Control",
  "dataset_id": "telco_churn",
  "drift_scenario": "TELCO_CONCEPT",
  "reference_control_arm": "A0 (Frozen Experimental Source RF, Batch 5F)",
  "adaptation_arms": {
    "A1-B50": { "source_train_rows": 3452, "adaptation_budget": 50, "synthetic_rows": 0, "total_training_rows": 3502 },
    "A1-B100": { "source_train_rows": 3452, "adaptation_budget": 100, "synthetic_rows": 0, "total_training_rows": 3552 },
    "A1-B200": { "source_train_rows": 3452, "adaptation_budget": 200, "synthetic_rows": 0, "total_training_rows": 3652 }
  },
  "classifier_hyperparameters": {
    "n_estimators": 100, "max_depth": 10, "class_weight": "balanced",
    "random_state": 42, "n_jobs": 2, "bootstrap": true, "criterion": "gini"
  }
}
```

* **Preprocessing Policy:** A separate `TabularDataPreprocessor` was fitted strictly on each arm's permitted training composition ($D_{\text{source,train}} \cup B_k$). This guarantees complete self-containment, eliminates cross-arm contamination, and ensures zero data snooping from evaluation sets.
* **Evaluation Access Rule:** Single-pass confirmatory evaluation on $D_{\text{target,eval}}$ ($N=493$) and $D_{\text{source,eval}}$ ($N=493$). Zero iterative hyperparameter tuning, seed searching, or budget selection using evaluation sets.

---

## 3. Input Artifacts and Integrity Verification

| Artifact | File Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Canonical Split** | `component3/results/split_manifests/telco_churn_split_manifest.json` | `e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68` | **LOCKED & VERIFIED** |
| **Experimental Split** | `component3/results/adaptation_splits/telco_churn_adaptation_split_manifest.json` | `34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59` | **FROZEN & VERIFIED** |
| **Leakage Audit** | `component3/results/adaptation_splits/telco_churn_adaptation_group_overlap_audit.json` | `884d9d6983b1f589ff302749438f93bda84c5d4a2a7d5d90e7f793c0a2008518` | **PASS (0 Overlap)** |
| **Batch 5F Source Model** | `component3/models/experimental/telco_rf_experimental_source.joblib` | `256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0` | **INVIOLABLE & UNTOUCHED** |
| **Historical Baseline RF**| `component3/models/baselines/telco_churn_random_forest.joblib` | `69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701` | **PRESERVED UNCHANGED** |

* **Row Conservation:** $3,452 + 493 + 493 + 493 = 4,931$ rows.
* **Canonical Train Fingerprint:** `0348ebce7a8c776d`.

---

## 4. Adaptation Budget Composition & Concept Transformation

The nested adaptation budgets ($B_{50} \subset B_{100} \subset B_{200}$) carved from $D_{\text{target,adapt\_pool}}$ were transformed under the frozen `TELCO_CONCEPT` rule:

$$\text{If } \text{tenure} > 29.0 \implies \text{Churn} \gets \text{NOT}(\text{Churn});\quad \text{Else } \text{Churn} \gets \text{Churn}$$

| Dimension | Budget $B_{50}$ | Budget $B_{100}$ | Budget $B_{200}$ | Target Eval ($D_{\text{target,eval}}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Rows** | 50 | 100 | 200 | 493 |
| **Affected Slice ($\text{tenure} > 29$)** | 25 (50.0%) | 48 (48.0%) | 97 (48.5%) | 220 (44.62%) |
| **Labels Inverted** | Exactly 25 | Exactly 48 | Exactly 97 | Exactly 220 |
| **Original Observational Churn** | No: 40, Yes: 10 | No: 76, Yes: 24 | No: 147, Yes: 53 | No: 362, Yes: 131 |
| **Transformed Target Churn** | No: 19, Yes: 31 | No: 40, Yes: 60 | No: 76, Yes: 124 | No: 190, Yes: 303 |
| **Target Positive Class Rate** | 62.0% | 60.0% | 62.0% | 61.46% |
| **Label Provenance** | `SYNTHETIC_CONCEPT_TRUTH` | `SYNTHETIC_CONCEPT_TRUTH` | `SYNTHETIC_CONCEPT_TRUTH` | `SYNTHETIC_CONCEPT_TRUTH` |
| **Feature Modification** | None (0 features mutated) | None (0 features mutated) | None (0 features mutated) | None (0 features mutated) |

---

## 5. Model Training Telemetry

All models were trained strictly on CPU with $n\_jobs=2$:

| Experimental Arm | Training Composition | Fit Runtime | Peak RAM (RSS) | Serialized Artifact Path |
| :--- | :--- | :---: | :---: | :--- |
| **$A0$ (Frozen Control)** | $D_{\text{source,train}}$ (3,452 rows) | 0.36s | 160.7 MB | `component3/models/experimental/telco_rf_experimental_source.joblib` |
| **$A1\text{-}B50$** | 3,452 Source + 50 Target Adapt = 3,502 | 0.61s | 170.2 MB | `component3/models/experimental/telco_rf_a1_b50.joblib` |
| **$A1\text{-}B100$** | 3,452 Source + 100 Target Adapt = 3,552 | 0.64s | 174.6 MB | `component3/models/experimental/telco_rf_a1_b100.joblib` |
| **$A1\text{-}B200$** | 3,452 Source + 200 Target Adapt = 3,652 | 0.65s | 180.3 MB | `component3/models/experimental/telco_rf_a1_b200.joblib` |

---

## 6. Target-Domain Evaluation Performance

Evaluated on $D_{\text{target,eval}}$ ($N=493$, 303 positive targets, 190 negative targets):

| Metric | $A0$ (Frozen Reference) | $A1\text{-}B50$ | $A1\text{-}B100$ | $A1\text{-}B200$ |
| :--- | :---: | :---: | :---: | :---: |
| **Positive F1-Score** | **0.4139** | **0.4033** | **0.4090** | **0.4141** |
| **Macro F1-Score** | 0.4198 | 0.4057 | 0.4138 | 0.4257 |
| **Accuracy** | 0.4199 | 0.4057 | 0.4138 | 0.4260 |
| **Balanced Accuracy** | 0.4456 | 0.4292 | 0.4387 | 0.4545 |
| **Positive Precision** | 0.5459 | 0.5266 | 0.5376 | 0.5556 |
| **Positive Recall** | 0.3333 | 0.3267 | 0.3300 | 0.3300 |
| **ROC-AUC** | 0.4063 | 0.4104 | 0.4208 | 0.4493 |
| **PR-AUC (Average Precision)**| 0.5976 | 0.5928 | 0.6031 | 0.6115 |
| **Confusion Matrix** $\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix}$ | $\begin{pmatrix} 106 & 84 \\ 202 & 101 \end{pmatrix}$ | $\begin{pmatrix} 101 & 89 \\ 204 & 99 \end{pmatrix}$ | $\begin{pmatrix} 104 & 86 \\ 203 & 100 \end{pmatrix}$ | $\begin{pmatrix} 110 & 80 \\ 203 & 100 \end{pmatrix}$ |

---

## 7. Paired Statistical Analysis & Multiplicity Audit

Evaluated via paired prediction-level comparisons relative to $A0$ on identical $D_{\text{target,eval}}$ records:

| Adaptation Arm | $\Delta F1$ Point Est. | 95% Bootstrap CI ($B=2,000$) | CI Spans Zero? | Permutation $p_{\text{raw}}$ ($R=2,000$) | Bonferroni $p_{\text{adj}}$ | Benjamini-Hochberg FDR $p_{\text{adj}}$ | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$A1\text{-}B50$ vs $A0$** | **-0.0107** | $[-0.0243, +0.0015]$ | Yes | 0.1215 | 0.3645 | 0.3645 | **No** |
| **$A1\text{-}B100$ vs $A0$** | **-0.0049** | $[-0.0200, +0.0104]$ | Yes | 0.6090 | 1.0000 | 0.9135 | **No** |
| **$A1\text{-}B200$ vs $A0$** | **+0.0001** | $[-0.0121, +0.0119]$ | Yes | 1.0000 | 1.0000 | 1.0000 | **No** |

### Statistical Conclusions:
* **Confidence Intervals:** All three 95% bootstrap confidence intervals span zero, confirming that neither small positive nor small negative observed differences deviate from random variation.
* **Permutation Tests:** Permutation testing confirms no departure from the null hypothesis of equal performance ($p_{\text{raw}} \ge 0.1215$; adjusted $p \ge 0.3645$).
* **Recovery Fraction ($\mathcal{R}$):** $\mathcal{R}(A1\text{-}B200) = \frac{+0.0001}{0.1936} \times 100\% = \mathbf{+0.05\%}$. For all practical and statistical purposes, unweighted authentic pooling achieves **0% recovery** under small sample budgets.

---

## 8. Source-Domain Retention Assessment

Evaluated on $D_{\text{source,eval}}$ ($N=493$, pre-drift observational truth):

| Arm | Source F1 | $\Delta F1$ vs $A0$ | Source Macro F1 | Source Balanced Acc | Catastrophic Forgetting? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$A0$ (Reference)** | **0.6075** | Baseline | 0.7208 | 0.7389 | None |
| **$A1\text{-}B50$** | **0.6233** | $+0.0158$ | 0.7324 | 0.7506 | **No** (Preserved) |
| **$A1\text{-}B100$** | **0.6093** | $+0.0018$ | 0.7184 | 0.7420 | **No** (Preserved) |
| **$A1\text{-}B200$** | **0.5980** | $-0.0095$ | 0.7107 | 0.7330 | **No** (Preserved) |

* **Zero Catastrophic Forgetting:** Retention on the historical pre-drift distribution remained high ($\Delta F1 \ge -0.0095$). Because pre-drift data heavily outnumbered adaptation data, the historical decision boundaries remained virtually intact.

---

## 9. Reload Parity and Anonymized Prediction Audit

* **Reload Parity:** Reloading serialized models and preprocessors from disk yielded **100% bitwise prediction parity** and float probability parity within $10^{-7}$ across all arms.
* **Privacy Assurance:** The prediction audit artifact [`telco_adaptation_predictions_audit.csv`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_adaptation_predictions_audit.csv) contains only anonymized row identifiers (`eval_row_000` to `eval_row_492`) with predictions, probabilities, and slice flags; zero raw customer IDs are stored.
* **Baseline Preservation:** Cryptographic hash verification confirmed that historical baseline models in `component3/models/baselines/` and the Batch 5F source model in `component3/models/experimental/` were never modified.

---

## 10. Threat to Validity & Scientific Limitations

1. **Unweighted Pooling Constraint:** Arm $A1$ tested standard unweighted data concatenation ($D_{\text{source,train}} \cup B_k$). It does not evaluate recency weighting, sample reweighting, or slice-targeted fine-tuning, which might yield better authentic utilization.
2. **Budget Scale Bound:** The maximum tested budget ($B_{200}$) represented only 5.8% of the source training data. A much larger adaptation sample (e.g., $N \ge 1,000$) would likely begin to shift decision trees, but acquiring large labeled samples under rapid drift is typically infeasible in operational environments.
3. **Exploratory Confirmatory Demarcation:** While the protocol was frozen and single-pass confirmatory on $D_{\text{target,eval}}$, subsequent experiments must avoid using these numerical results to post-hoc adjust drift parameters or evaluation thresholds.

---

## 11. Decision Recommendation for Subsequent Synthetic Repair Experiments

$$\mathbf{CONDITIONAL\ GO}$$

### Scientific Justification:
The experimental results from Batch 5G provide the **essential empirical baseline for Component 3**:
* We have proved that authentic labeled data alone at realistic operational budgets ($N \le 200$) is **statistically incapable of repairing model degradation under concept drift**.
* This formally establishes the scientific necessity for synthetic data repair (Track B).
* Authorized to proceed to synthetic repair investigation, under the **strict gating condition** that any synthetic repair arm ($A2, A3, A4$) must be statistically benchmarked against Arm $A1$ under identical budget constraints.

---

## 12. Scientific Qualification & Scope Clarification (Sprint 06 — Batch 5H Addendum)

During the scientific audit of Batch 5G conducted in Sprint 06 — Batch 5H, the following essential qualifications and methodological boundary conditions were established to prevent over-generalization:

1. **Scope of the Inefficacy Finding:** The empirical finding that authentic adaptation produced no statistically significant target F1 improvement ($\Delta F1 \le +0.0001$, $p_{\text{adj}} \ge 0.3645$) applies strictly to **naive unweighted pooling** ($D_{\text{source,train}} \cup B_k$). Because 3,452 pre-drift records (including 1,719 records with `tenure > 29.0`) outnumber the 25–97 post-drift adaptation records in that slice by $\approx 17.7 : 1$, standard tree splitting splits predominantly on historical correlations.
2. **Non-Synthetic Alternatives Not Yet Precluded:** The inefficacy of naive unweighted pooling **does not prove that synthetic repair is uniquely necessary**. Stronger non-synthetic baselines—specifically sample-weighted adaptation (e.g., $w_{\text{target}} \gg 1.0$), authentic target oversampling (ROS), and target-only training—have not yet been evaluated. These authentic methods might also overcome the pre-drift dominance without generative modeling.
3. **Synthetic Rule Boundary:** The `TELCO_CONCEPT` ground-truth labels are generated by a known, deterministic programmatic rule ($\text{tenure} > 29.0 \implies \text{invert Churn}$). This constitutes a controlled synthetic benchmark (`SYNTHETIC_CONCEPT_TRUTH`) for simulation and stress testing, not observed empirical consumer behavior.
4. **Pre-Drift Generator Checkpoint Invalidation:** Existing generator checkpoints (`telco_churn_ctgan_model.joblib`, `telco_churn_tvae_model.joblib`, `telco_churn_tabddpm_model.joblib`) were fitted exclusively on pre-drift positive records. They model $P_{\text{source}}(X, Y)$ where `tenure > 29.0` has low churn propensity. They cannot automatically represent or generate post-drift concept relationships and cannot repair concept drift without post-drift retraining or adaptation.
5. **Prior Exposure of Target Evaluation Partition:** The target evaluation partition $D_{\text{target,eval}}$ ($N=493$) was evaluated in Batch 5F (for $A0$) and Batch 5G (for $A0, A1\text{-}B50, A1\text{-}B100, A1\text{-}B200$). Because its empirical metrics are now known, any iterative tuning of hyperparameters, sample weights, synthetic quantities $K$, or generator checkpoints against $D_{\text{target,eval}}$ would constitute severe data snooping.
6. **Development Isolation Mandate:** Future method development, hyperparameter selection, and synthetic quantity tuning must be conducted exclusively on an isolated development validation protocol drawn from the permitted training-side records ($D_{\text{target,adapt\_pool}}$), leaving $D_{\text{target,eval}}$ strictly locked as a final, single-pass comparison whose results must be documented as exploratory-confirmatory given prior exposure history.
