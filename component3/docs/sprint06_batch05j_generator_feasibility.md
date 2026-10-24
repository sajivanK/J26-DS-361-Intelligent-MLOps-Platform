# SPRINT 06 — BATCH 5J / BATCH 5J-AUDIT
# Post-Drift Synthetic Generator Feasibility & Candidate Validation Audit

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan`  
**Audit Date:** October 20, 2026  
**Execution Environment:** Windows Laptop, Python 3.11, PyTorch 2.10.0+cpu, CPU-Only Execution  

---

## EXECUTIVE SUMMARY & AUDIT RECONCILIATION

This audited report formalizes the scientific evaluation of **CTGAN**, **TVAE**, and **TabDDPM** under strict small-sample, CPU-only constraints using **only the 200 labeled post-drift adaptation records** ($B_{200}$, $N=200$) under the frozen `TELCO_CONCEPT` distribution shift.

In accordance with the Batch 5J-Audit mandate:
1. **Zero Generator or Model Retraining:** All candidate metrics and validation funnels were audited from the existing frozen candidate CSV artifacts. No models were refitted.
2. **Reconciled Concept Inversions:** Verified that **all 97 affected records** with $\text{tenure} > 29.0$ are inverted under `TELCO_CONCEPT` (84 $\text{No} \to \text{Yes}$ and 13 $\text{Yes} \to \text{No}$), resulting in exactly 124 `Yes` (62.0%) and 76 `No` (38.0%). Previous ambiguous documentation suggesting "84 records flipped" has been corrected.
3. **Corrected Synthetic Concept Metrics:** The previous "pure deterministic rule agreement" metric was removed as an acceptance criterion because synthetic records lack authentic pre-drift counterfactual identities. Replaced with purely descriptive conditional feature–label comparisons against $B_{200}$ and independent $D_{\text{dev\_val}}$ references.
4. **Reconciled Frozen Benchmark References:** Reconciled historical benchmark references against authoritative Batch 5I saved JSON files:
   - $B0$ (Frozen Source Model): $F1_{\text{dev}} = \mathbf{0.4055}$ on clean $D_{\text{dev\_val}}$ ($N=292$); $F1_{\text{source}} = \mathbf{0.6075}$ on $D_{\text{source,eval}}$ ($N=493$).
   - $B2\_B200\_K200$ (Authentic ROS): $F1_{\text{dev}} = \mathbf{0.4497}$; $F1_{\text{source}} = \mathbf{0.5890}$.
   - $B4\_B200$ (Target-Only Model): $F1_{\text{dev}} = \mathbf{0.7660}$; $F1_{\text{source}} = \mathbf{0.3492}$ (reconciled from a 0.3089 reporting transcription error).
5. **Audited Sequential Funnel vs. Independent Pass Rates:** Confirmed that Tier 3 plausible unique counts (TVAE: 65, TabDDPM: 42, CTGAN: 30) are calculated sequentially among Tier 2 survivors after deduplication, and that invalid records are rejected rather than silently clamped.
6. **Feasibility Verdict:** **CONDITIONAL GO for Batch 5K**. Candidate-quality metrics reflect structural and distributional properties but **do not evaluate downstream utility or prove successful drift repair**.

---

## 1. VERIFIED $B_{200}$ TRAINING-DATA PROVENANCE & ISOLATION

### 1.1 Dataset Provenance and Cryptographic Fingerprints

The training partition $B_{200}$ is nested within the adaptation budget container of $D_{\text{target,adapt\_pool}}$ ($N=493$), derived strictly from the canonical training partition ($N=4,938$) of the Telco Customer Churn dataset:

| Dimension | Observed / Verified Value | Integrity Status |
| :--- | :--- | :--- |
| **Dataset ID** | `telco_churn` | Verified Registry Key |
| **Canonical Split Manifest SHA-256** | `e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68` | Cryptographically Matched |
| **Adaptation Split Manifest SHA-256** | `34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59` | Cryptographically Matched |
| **Adaptation Leakage Audit SHA-256** | `884d9d6983b1f589ff302749438f93bda84c5d4a2a7d5d90e7f793c0a2008518` | Cryptographically Matched |
| **Source Preprocessor Path** | `component3/models/experimental/telco_preprocessor_experimental_source.joblib` | Verified Unrefitted |
| **$B_{200}$ Training Dataset Fingerprint** | `4434765ddf207253` | Frozen Reference Input |
| **Total Realized $B_{200}$ Rows** | Exactly 200 records | Verified |
| **Total Features Modeled** | 20 columns (3 numerical, 16 categorical predictors + 1 post-drift `Churn`) | Verified |

### 1.2 Isolation from Development and Evaluation Partitions

Feature-signature audits across all 19 predictor columns confirmed **zero cross-partition leakage**:

| Comparison Partition | Nominal Size ($N$) | Overlapping Signatures with $B_{200}$ | Audit Status |
| :--- | :---: | :---: | :---: |
| **$D_{\text{source,train}}$** | 3,452 | 0 | **PASS (Zero Leakage)** |
| **$D_{\text{source,eval}}$** | 493 | 0 | **PASS (Zero Leakage)** |
| **$D_{\text{dev\_val}}$ (Clean)** | 292 | 0 | **PASS (Zero Leakage)** |
| **$D_{\text{target,eval}}$ (Locked)** | 493 | 0 | **PASS (Zero Quarantine)** |

### 1.3 Preprocessor Integrity Check

The tabular preprocessor was loaded from frozen `telco_preprocessor_experimental_source.joblib`. Only `.transform()` was executed to coerce strings (e.g., whitespace in `TotalCharges`) using pre-drift statistics. **The preprocessor was never refitted on target adaptation data.**

---

## 2. RECONCILED POST-DRIFT CLASS COUNTS UNDER `TELCO_CONCEPT`

Under the frozen deterministic rule:
$$\text{tenure} > 29.0 \implies \text{Churn} \gets \begin{cases} \text{"Yes"}, & \text{if Churn} = \text{"No"} \\ \text{"No"}, & \text{if Churn} = \text{"Yes"} \end{cases}$$

The empirical shift realized in $B_{200}$ ($N=200$) is verified as follows:

| Slice | Raw Pre-Drift $B_{200}$ | Post-Drift $B_{200}$ (`TELCO_CONCEPT`) | Shift Details |
| :--- | :--- | :--- | :--- |
| **Affected Slice ($\text{tenure} > 29.0$)** | 97 rows<br>• `No`: 84 (86.60%)<br>• `Yes`: 13 (13.40%) | 97 rows<br>• `No`: 13 (13.40%)<br>• `Yes`: 84 (86.60%) | **All 97 Affected Records Inverted**<br>• 84 inverted from `No` to `Yes`<br>• 13 inverted from `Yes` to `No` |
| **Preserved Slice ($\text{tenure} \le 29.0$)** | 103 rows<br>• `No`: 63 (61.17%)<br>• `Yes`: 40 (38.83%) | 103 rows<br>• `No`: 63 (61.17%)<br>• `Yes`: 40 (38.83%) | **Preserved Baseline Truth**<br>0 records modified |
| **Total Partition ($B_{200}$)** | 200 rows<br>• `No`: 147 (73.50%)<br>• `Yes`: 53 (26.50%) | 200 rows<br>• `No`: 76 (38.00%)<br>• `Yes`: 124 (62.00%) | **Class Inversion**<br>Minority class becomes `No` (38.00%) |

*Correction Note:* Previous draft documentation stating that "84 labels were flipped" has been reconciled: exactly 84 records flipped $\text{No} \to \text{Yes}$ and 13 records flipped $\text{Yes} \to \text{No}$, totaling **97 inverted records** (100% of the affected slice).

---

## 3. GENERATION STRATEGY: JOINT GENERATION RATIONALE

- **Strategy A (Joint Generation):** Models full joint distribution $P(X, Y)$ on all 200 records, treating `Churn` as a discrete variable.
- **Strategy B (Class-Specific Generation):** Separate models for `No` ($N=76$) and `Yes` ($N=124$).

**Defensible Choice: Strategy A (Joint Generation).**  
*Rationale:* Splitting $N=200$ records across 19 features into class-specific pools leaves only 76 records for the `No` class. Neural generators (CTGAN, TVAE, TabDDPM) undergo catastrophic mode collapse and singular covariance matrices when trained on 76 tabular rows. Strategy A models the full dataset and enables empirical evaluation of macroscopic conditional associations. Pre-drift models and labels were not reused.

---

## 4. CPU-ONLY GENERATOR FEASIBILITY & ARTIFACT REPRODUCIBILITY

Pilot smoke tests on CPU demonstrated execution safety without memory exhaustion:

| Generator | Architecture | Epochs | Fit Time | Sampling Time ($K=200$) | Peak RSS Memory | Candidate CSV SHA-256 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **TVAE** | Variational Autoencoder | 150 | 2.74 s | 0.04 s | 403.93 MB | `ce22c1adb3d7a0cc332d6b1c993e739b4b17b25156f45129e84667b74b40ddf9` |
| **CTGAN** | Conditional WGAN-GP | 150 | 7.01 s | 0.06 s | 418.76 MB | `1d8b1f4c79c81041d66de1986f5559d505b469a9c5da0a85d531b09e8e021c63` |
| **TabDDPM** | Multinomial Diffusion | 20 (50 steps) | 1.13 s | 0.35 s | 420.72 MB | `14b3112d0ec92f037128c7e30a1c9969e8231f66377e6817e7564d997877a6ec` |

---

## 5. AUDITED CANDIDATE FILTERING: SEQUENTIAL FUNNEL VS. INDEPENDENT PASS RATES

Candidate yields were audited directly from the frozen CSVs. We report both **Independent Pass Rates** (evaluating isolated criteria across all 200 rows) and the **Sequential Filter Funnel** (cascading pipeline):

### 5.1 Independent Criteria Pass Rates ($N=200$)

| Evaluation Criterion | TVAE | CTGAN | TabDDPM | Description / Rule |
| :--- | :---: | :---: | :---: | :--- |
| **Tier 1: Structural Valid** | **200 (100.0%)** | **200 (100.0%)** | **200 (100.0%)** | Correct columns, zero NaNs, zero Infs, valid discrete domains |
| **Tier 2: Hard Physical Valid** | **167 (83.5%)** | **115 (57.5%)** | **200 (100.0%)** | $0 \le \text{tenure} \le 72$, $\text{MonthlyCharges} \ge 0$, $\text{TotalCharges} \ge 0$ |
| — Tenure $\ge 0$ | 191 (95.5%) | 127 (63.5%) | 200 (100.0%) | CTGAN generated 73 negative tenure records |
| — Tenure $\le 72$ | 194 (97.0%) | 197 (98.5%) | 200 (100.0%) | TVAE generated 6 excessive tenure records |
| — Total Charges $\ge 0$ | 179 (89.5%) | 185 (92.5%) | 200 (100.0%) | Unbounded Gaussian tails produce negative billing |
| **Tier 2: Within-Sample Unique** | **200 (100.0%)** | **200 (100.0%)** | **169 (84.5%)** | TabDDPM generated 31 repeated records |
| **Tier 2: $B_{200}$ Training Non-Match** | **200 (100.0%)** | **200 (100.0%)** | **200 (100.0%)** | Zero exact copies of authentic training rows |
| **Tier 2: Evaluation Leakage-Free** | **200 (100.0%)** | **200 (100.0%)** | **200 (100.0%)** | Zero feature overlap with $D_{\text{source,train}}, D_{\text{source,eval}}, D_{\text{dev\_val}}, D_{\text{target,eval}}$ |
| **Tier 3: Billing Ratio Valid** | **70 (35.0%)** | **32 (16.0%)** | **49 (24.5%)** | $0.5 \le \frac{\text{TotalCharges}}{\max(1, \text{tenure}) \times \text{MonthlyCharges}} \le 1.5$ across all 200 rows |

### 5.2 Sequential Filter Funnel (Cascading Pipeline)

```
[Stage 0: Candidates Requested & Returned] ──► 200 (100.0%) for all models
       │
       ▼
[Stage 1: Tier 1 Structural Survivors]    ──► 200 (100.0%) for all models
       │
       ▼
[Stage 2: Tier 2 Hard Physical Survivors] ──► TVAE: 167 (83.5%), CTGAN: 115 (57.5%), TabDDPM: 200 (100.0%)
       │
       ▼
[Stage 3: Tier 2 Unique Hard-Valid]       ──► TVAE: 167 (83.5%), CTGAN: 115 (57.5%), TabDDPM: 169 (84.5%)
       │
       ▼
[Stage 4: Tier 3 Plausible Unique Yield]  ──► TVAE: 65 (32.5%), TabDDPM: 42 (21.0%), CTGAN: 30 (15.0%)
```

### 5.3 Billing Ratio Treatment for $\text{tenure} = 0$
In Telco Customer Churn, new customers with $\text{tenure} = 0$ have expected $\text{TotalCharges} \approx \text{MonthlyCharges}$. The billing ratio screen clamps $\text{tenure}$ to a minimum of 1.0 ($\text{safe\_tenure} = \max(1.0, \text{tenure})$).  
- In **TabDDPM**, 104 candidates had $\text{tenure} = 0.0$. Of these, **only 34 satisfied $\text{TotalCharges} \approx \text{MonthlyCharges}$**, while 70 rows generated unphysical total charges up to $\$8,337.45$ for brand new customers. These unphysical rows were properly rejected by the filter funnel and **never silently clamped**.

---

## 6. CORRECTED CONCEPT-RELATIONSHIP INTERPRETATION

### 6.1 Removal of Deterministic Rule Gating
The previously reported "pure deterministic rule agreement" metric was removed. In synthetic generation, individual candidates are newly synthesized; they have no authentic pre-drift counterfactual identity. Testing whether a synthetic record's label equals what a rule would dictate is conceptually flawed.

### 6.2 Descriptive Feature–Label Conditional Distributions

We report macroscopic conditional distributions compared against the permitted training reference ($B_{200}$, $N=200$) and independent development reference ($D_{\text{dev\_val}}$, $N=292$):

| Dataset / Candidate Source | Overall Churn ($P(\text{Yes})$) | Affected Slice ($\text{tenure} > 29$) $P(\text{Yes})$ | Preserved Slice ($\text{tenure} \le 29$) $P(\text{Yes})$ | Conditional Difference ($\Delta P$) | Class Balance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Authentic $B_{200}$ (Training Ref.)** | 62.00% (124/200) | **86.60%** (84/97) | **38.83%** (40/103) | **+47.76%** | 0.6129 |
| **Authentic $D_{\text{dev\_val}}$ (Independent Ref.)** | 66.44% (194/292) | **89.19%** (132/148) | **43.06%** (62/144) | **+46.13%** | 0.5052 |
| **TabDDPM Candidates** | 63.00% (126/200) | **86.46%** (83/96) | **41.35%** (43/104) | **+45.11%** | 0.5873 |
| **CTGAN Candidates** | 60.50% (121/200) | **62.50%** (30/48) | **59.87%** (91/152) | **+2.63%** | 0.6529 |
| **TVAE Candidates** | 76.50% (153/200) | **100.00%** (44/44) | **69.87%** (109/156) | **+30.13%** | 0.3072 |

### 6.3 Scientific Findings on Concept Representation:
1. **TabDDPM captured the macroscopic conditional distribution:** TabDDPM produced an 86.46% churn rate in the affected slice and 41.35% in the preserved slice ($\Delta P = +45.11\%$), closely matching authentic post-drift reality ($B_{200}: +47.76\%$, $D_{\text{dev\_val}}: +46.13\%$).  
   *Caveat:* Matching conditional rates reflects distribution learning; it does **not** prove that individual synthetic labels are counterfactually correct.
2. **CTGAN failed to capture the conditional shift:** CTGAN produced virtually identical churn rates across both tenure slices (62.50% vs 59.87%, $\Delta P = +2.63\%$), demonstrating that its generator treated tenure and churn as conditionally independent.
3. **TVAE suffered from majority-class mode collapse:** TVAE generated high churn rates across both slices (100.0% and 69.87%), compressing the minority class down to 23.5%.

---

## 7. AUDITED PRIVACY-RISK SCREENING & MEMORIZATION

| Screening Metric | TVAE | CTGAN | TabDDPM | Technical Definition & Risk Level |
| :--- | :---: | :---: | :---: | :--- |
| **Exact Matches with $B_{200}$** | 0 / 200 | 0 / 200 | 0 / 200 | Full 20-column exact matches against training data |
| **Evaluation Partition Overlap** | 0 / 200 | 0 / 200 | 0 / 200 | 19-feature signature matches against evaluation splits |
| **Within-Sample Duplicates** | 0 | 0 | **31 (15.5%)** | Repeated identical records within synthetic candidate set |
| **DCR 5th Percentile ($DCR_{p05}$)** | 0.0246 | 0.0551 | **0.0000** | Euclidean distance on Min-Max normalized continuous features |
| **DCR Median** | 0.0736 | 0.2624 | 0.1706 | Median distance to closest authentic $B_{200}$ record |
| **Near-Duplicates ($\text{DCR} < 0.01$)** | 0 | 0 | **15** | Continuous distance $< 0.01$ to a training record |

### Privacy Disclaimers & Clarifications:
- **DCR is an empirical heuristic screening metric, not a mathematical privacy guarantee.** A non-zero DCR does not establish differential privacy ($\epsilon, \delta$-DP), nor does it guarantee absence of attribute disclosure.
- **TabDDPM exhibited a severe small-sample memorization signal:** $DCR_{p05} = 0.0000$ and 15 near-duplicates indicate that under small training budgets ($N=200$), diffusion models tend to collapse continuous trajectories onto empirical training coordinates.

---

## 8. RECONCILED FROZEN BENCHMARK REFERENCES

We audited the authoritative Batch 5I prediction artifacts ([`telco_authentic_controls_dev_metrics_corrected.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_authentic_controls_dev_metrics_corrected.json) and [`telco_authentic_controls_source_retention.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_authentic_controls_source_retention.json)).

### 8.1 Benchmark Reconciliation Table

| Benchmark Arm | Training Composition | Reconciled $F1_{\text{dev}}$ ($D_{\text{dev\_val}}, N=292$) | Reconciled $F1_{\text{source}}$ ($D_{\text{source,eval}}, N=493$) | Batch 5J Reporting Discrepancy & Resolution |
| :--- | :--- | :---: | :---: | :--- |
| **$B0$ (Frozen Source Model)** | 3,452 Source records | **0.4055** | **0.6075** | **Resolved:** Batch 5J listed 0.4139 (which was historical $D_{\text{target,eval}}$ baseline). Clean development validation F1 on $D_{\text{dev\_val}}$ is **0.4055**. |
| **$B2\_B200\_K200$ (Authentic ROS)** | 3,452 Source + $B_{200}$ + 200 Duplicated $B_{200}$ | **0.4497** | **0.5890** | **Verified:** Unchanged across both audit files. |
| **$B4\_B200$ (Target-Only Model)** | Exactly 200 $B_{200}$ records (0 Source) | **0.7660** | **0.3492** | **Resolved:** Batch 5J listed 0.3089 due to a reporting transcription error. Authoritative source retention file confirms **0.3492**. |

---

## 9. SCIENTIFIC DISTINCTION OF EVALUATION DIMENSIONS

To prevent premature conclusions, we distinguish the five scientific dimensions evaluated:

| Dimension | Observed Finding | Downstream Implication |
| :--- | :--- | :--- |
| **A. CPU Feasibility** | All models fit in $< 8\text{ s}$ with RAM $< 425\text{ MB}$. | Feasible for laptop execution in Batch 5K. |
| **B. Candidate Quality** | Plausible unique yields: TVAE 32.5%, TabDDPM 21.0%, CTGAN 15.0%. | Raw candidates require filtering before downstream use. |
| **C. Distributional Similarity** | CTGAN best categorical TVD (0.0565); TabDDPM best conditional shift ($\Delta P = +45.11\%$). | Different models capture different distribution slices. |
| **D. Privacy Indicators** | CTGAN/TVAE maintain continuous spread; TabDDPM shows memorization ($DCR_{p05}=0.0$). | Memorization must be factored into TabDDPM utility claims. |
| **E. Downstream Utility** | **NOT EVALUATED IN THIS BATCH.** | **Zero claims of drift repair or generator superiority.** |

---

## 10. DECISION FOR BATCH 5K

### Verdict: **CONDITIONAL GO**

**Mandatory Conditions for Batch 5K:**
1. **Candidate Filtering Enforcement:** Raw generator outputs must undergo Tier 2 hard physical filtering and Tier 3 plausibility screening prior to pooling into downstream retraining.
2. **Bounded Augmentation Budgets ($K=50$ and $K=200$):** $K=50$ is readily supported by plausible unique survivors; $K=200$ requires sampling larger raw pools to survive filtering.
3. **TabDDPM Memorization Caution:** Any downstream performance of TabDDPM must be interpreted alongside its near-duplicate rate.
4. **Benchmarking Against Authentic ROS ($B2$):** Synthetic repair models must be evaluated against $B2\_B200\_K200$ ($F1_{\text{dev}} = 0.4497, F1_{\text{source}} = 0.5890$).
5. **Absolute Quarantine of Locked Evaluation:** $D_{\text{target,eval}}$ ($N=493$) remains quarantined; evaluation is conducted strictly on development validation $D_{\text{dev\_val}}$ ($N=292$).
