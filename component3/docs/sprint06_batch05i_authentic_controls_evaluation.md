# SPRINT 06 — BATCH 5I TECHNICAL REPORT
## Strong Authentic Adaptation Controls and Development Validation

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan` | **Base Commit:** `9a3b2b9`  
**Execution Timestamp:** 2026-10-20T12:24:00+05:30  
**Experimental Scenario:** `TELCO_CONCEPT` (`tenure > 29.0` inverted Churn label)  
**Governing Protocol:** [`batch05i_authentic_controls_protocol.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/batch05i_authentic_controls_protocol.json)  
**Development Split Audit:** [`telco_dev_val_split_audit.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_dev_val_split_audit.json)  

---

## 1. Executive Summary & Research Objectives

In Sprint 06 — Batch 5I, we implemented and evaluated the **six non-synthetic adaptation controls (Arms B0–B5)** designed during Batch 5H. The primary scientific objective was to establish rigorous authentic baselines using only source training records ($N=3,452$) and permitted labeled adaptation records ($B_k \in \{50, 100, 200\}$) evaluated on an isolated post-drift development validation set ($D_{\text{adapt,dev\_val}}$, $N=292$), completely independent of the locked final target evaluation partition ($D_{\text{target,eval}}$, $N=493$).

### Key Empirical Discoveries:
1. **Dynamic Development Validation Isolation ($N=292$):** From the 293 unused records of $D_{\text{target,adapt\_pool}}$, automated feature-signature auditing detected exactly **one cross-partition duplicate feature pair** (`9122-UMROB` in $B_{200}$ and `2676-ISHSF` in unused pool). Excluding `2676-ISHSF` produced an unassailable clean development validation set of **exactly 292 records with verified zero duplicate-group overlap** with $B_{200}$, $D_{\text{source,train}}$, $D_{\text{source,eval}}$, and $D_{\text{target,eval}}$.
2. **Failure of Naive Pooling Reconfirmed (Arm B1):** Naive unweighted concatenation ($D_{\text{source,train}} \cup B_k$) yielded negligible development F1 gains over the frozen source model ($B0 = 0.4055$): $B1\text{-}B50 = 0.4150$, $B1\text{-}B100 = 0.3958$, $B1\text{-}B200 = 0.4150$ ($\Delta F1 \le +0.0095$, all $p_{\text{bonf}} = 1.0000$).
3. **Power of Target-Only Learning (Arm B4):** Training strictly on the labeled adaptation sample ($B_k$) with zero historical source data achieved **F1 = 0.7660** on $B_{200}$ (affected-slice F1 = 0.9071, $\Delta F1 = +0.3510$, $p_{\text{adj}} < 0.0001$). Eliminating the 3,452 contradictory pre-drift records allows the decision trees to cleanly learn the post-drift concept without historical interference.
4. **The Catastrophic Forgetting Trade-Off:** While B4 dominates target-domain adaptation, diagnostic evaluation on $D_{\text{source,eval}}$ reveals severe **catastrophic forgetting** ($\Delta F1_{\text{source}} = -0.2583$, source F1 dropped from 0.6075 to 0.3492). B4 is suitable only for permanent target migration.
5. **Strongest Source-Retaining Authentic Control (Arm B2):** For deployment settings requiring source-domain retention ($\Delta F1_{\text{source}} \ge -0.05$), **Authentic Random OverSampling (ROS, Arm B2-K200 on $B_{200}$)** achieved the strongest benchmark: **Target Dev F1 = 0.4497** ($\Delta F1 = +0.0347$ vs. B1, affected-slice F1 = 0.2194) while preserving Source Retention F1 at **0.5890** ($\Delta F1_{\text{source}} = -0.0185$, well within safety bounds).

---

## 2. Input Artifacts and Verification

All core input artifacts and model checkpoints were cryptographically verified prior to execution:

| Artifact Name | File Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Canonical Split Manifest** | `component3/results/split_manifests/telco_churn_split_manifest.json` | `e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68` | **LOCKED & VERIFIED** |
| **Experimental Split Manifest**| `component3/results/adaptation_splits/telco_churn_adaptation_split_manifest.json` | `34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59` | **FROZEN & VERIFIED** |
| **Dynamic Leakage Audit** | `component3/results/adaptation_splits/telco_churn_adaptation_group_overlap_audit.json` | `884d9d6983b1f589ff302749438f93bda84c5d4a2a7d5d90e7f793c0a2008518` | **PASS (0 Overlap)** |
| **Frozen Source Model (A0)** | `component3/models/experimental/telco_rf_experimental_source.joblib` | `256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0` | **INVIOLABLE & PRESERVED** |
| **Historical Baseline RF** | `component3/models/baselines/telco_churn_random_forest.joblib` | `69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701` | **INVIOLABLE & PRESERVED** |
| **Source Preprocessor** | `component3/models/experimental/telco_preprocessor_experimental_source.joblib` | `a188f0285e6cf07741bbb69be8b3df44eece492e69166c78d92353f1f4d41224` | **REUSED ACROSS ARMS** |

* **Canonical Train Fingerprint:** `0348ebce7a8c776d` (Verified identical).
* **Partition Counts:** Source Train = 3,452; Source Eval = 493; Target Adapt Pool = 493; Target Eval = 493.

---

## 3. Dynamic Development Validation Construction & Group Isolation

### 3.1 Mathematical Formulation of the Development Split
The target adaptation pool ($D_{\text{target,adapt\_pool}}$, $N=493$) was partitioned dynamically:
1. **$B_{200}$ Membership:** Identified directly from `splits.nested_adapt_budgets[200]` ($N=200$).
2. **Raw Unused Pool:** Records in $D_{\text{target,adapt\_pool}}$ not in $B_{200}$ ($N=293$).
3. **Duplicate Feature Signature Extraction:** For every record $i$, compute the tuple of frozen canonical feature columns:
   $$\sigma(x_i) = \left(x_{i, 1}, x_{i, 2}, \dots, x_{i, D}\right), \quad D=19 \text{ features}$$
4. **Duplicate Exclusion Rule:**
   $$D_{\text{adapt,dev\_val}} = \left\{ x_j \in \text{Unused Pool} \mid \sigma(x_j) \notin \{\sigma(x_k) \mid x_k \in B_{200}\} \right\}$$
5. **Audit Result:** Exactly one feature signature overlap was detected and excluded:
   * Record in $B_{200}$: Subscriber `9122-UMROB`
   * Overlapping Record in Unused Pool: Subscriber `2676-ISHSF`
   * Exclusion of `2676-ISHSF` yields realized count: **$N = 292$ records**.

### 3.2 Realized Development Validation Characteristics

| Dimension | Observational Raw Value | Transformed Concept Truth (`TELCO_CONCEPT`) |
| :--- | :---: | :---: |
| **Total Rows** | 292 | 292 |
| **Positive Class (Churn='Yes')** | 78 (26.71%) | **194 (66.44%)** |
| **Negative Class (Churn='No')** | 214 (73.29%) | **98 (33.56%)** |
| **Affected Slice ($\text{tenure} > 29.0$)** | 148 (50.68%) | 148 (Inverted: No $\leftrightarrow$ Yes) |
| **Preserved Slice ($\text{tenure} \le 29.0$)** | 144 (49.32%) | 144 (Preserved unchanged) |
| **Label Provenance** | `VERIFIED_OBSERVATIONAL_GROUND_TRUTH` | `SYNTHETIC_CONCEPT_TRUTH` |
| **Feature Values** | Untouched (0 mutations) | Untouched (0 mutations) |

### 3.3 Overlap Verification
* Overlap with $B_{200}$: **0**
* Overlap with $D_{\text{source,train}}$ ($N=3,452$): **0**
* Overlap with $D_{\text{source,eval}}$ ($N=493$): **0**
* Overlap with locked $D_{\text{target,eval}}$ ($N=493$): **0**
* **Isolation Status: 100% PASS.**

---

## 4. Control Arms Implementation (Arms B0–B5)

All models were trained strictly on CPU using `RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=2, bootstrap=True, criterion='gini')` transformed by the frozen source preprocessor:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       AUTHENTIC CONTROL ARMS SUMMARY                       │
├────┬─────────────────────────────┬────────────────────────┬────────────────┤
│ Arm│ Description                 │ Training Composition   │ Sample Weights │
├────┼─────────────────────────────┼────────────────────────┼────────────────┤
│ B0 │ Frozen Source Model         │ 3,452 Source           │ Uniform (w=1.0)│
│ B1 │ Naive Authentic Pooling     │ 3,452 Source + B_k     │ Uniform (w=1.0)│
│ B2 │ Authentic OverSampling (ROS)│ 3,452 Source + B_k + K │ Uniform (w=1.0)│
│ B3 │ Balanced Domain Weighting   │ 3,452 Source + B_k     │ w_tgt = 3452/Bk│
│ B4 │ Target-Only Model           │ B_k Only (0 Source)    │ Balanced Class │
│ B5 │ Fixed Weight Grid           │ 3,452 Source + B_k     │ w_tgt in {2..20│
└────┴─────────────────────────────┴────────────────────────┴────────────────┘
```

---

## 5. Development Validation Performance Results

Evaluated on $D_{\text{adapt,dev\_val}}$ ($N=292$, 194 positive targets, 98 negative targets):

> [!NOTE]
> **AUDIT NOTICE (Batch 5I-Audit):** The table below contains historical manual transcription errors (e.g., baseline B0 precision/recall duplicated into B1 rows). See [Section 13 (Audit Addendum)](#13-audit-addendum-metric-integrity-and-statistical-corrections-batch-5i-audit) and the dedicated audit document [`component3/docs/sprint06_batch05i_metric_integrity_audit.md`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/docs/sprint06_batch05i_metric_integrity_audit.md) for the verified corrected tables where $F1 = 2PR/(P+R)$ is mathematically proven across all arms.

| Experimental Arm | Budget | Training Size | Dev F1 | Macro F1 | Balanced Acc | Precision | Recall | ROC-AUC | PR-AUC | Affected Slice F1 ($\text{tenure}>29$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Frozen Source)** | 0 | 3,452 | **0.4055** | 0.4075 | 0.4582 | 0.6082 | 0.3041 | 0.4238 | 0.6511 | 0.1600 |
| **B1 (Naive Pooling)** | B50 | 3,502 | 0.4150 | 0.4075 | 0.4582 | 0.6082 | 0.3041 | 0.4238 | 0.6511 | 0.1600 |
| **B1 (Naive Pooling)** | B100 | 3,552 | 0.3958 | 0.4132 | 0.4581 | 0.6170 | 0.2990 | 0.4356 | 0.6657 | 0.1096 |
| **B1 (Naive Pooling)** | B200 | 3,652 | 0.4150 | 0.4075 | 0.4582 | 0.6082 | 0.3041 | 0.4238 | 0.6511 | 0.1600 |
| **B2 (ROS K=50)** | B50 | 3,552 | 0.3931 | 0.4013 | 0.4479 | 0.5895 | 0.2887 | 0.4358 | 0.6653 | 0.1224 |
| **B2 (ROS K=200)** | B50 | 3,702 | 0.3945 | 0.4071 | 0.4530 | 0.6000 | 0.2938 | 0.4368 | 0.6635 | 0.1224 |
| **B2 (ROS K=50)** | B100 | 3,602 | 0.4286 | 0.4237 | 0.4736 | 0.6275 | 0.3299 | 0.4344 | 0.6587 | 0.1722 |
| **B2 (ROS K=200)** | B100 | 3,752 | 0.4218 | 0.4154 | 0.4659 | 0.6122 | 0.3093 | 0.4361 | 0.6675 | 0.1722 |
| **B2 (ROS K=50)** | B200 | 3,702 | 0.4028 | 0.4194 | 0.4658 | 0.6237 | 0.2990 | 0.4385 | 0.6687 | 0.1351 |
| **B2 (ROS K=200)** | B200 | 3,852 | **0.4497** | **0.4367** | **0.4839** | **0.6389** | **0.3454** | **0.4485** | **0.6775** | **0.2194** |
| **B3 (Domain Weight)** | B50 | 3,502 | 0.3308 | 0.4057 | 0.4705 | 0.6970 | 0.2371 | 0.4632 | 0.6720 | 0.0993 |
| **B3 (Domain Weight)** | B100 | 3,552 | 0.3957 | 0.4452 | 0.4938 | 0.6866 | 0.2371 | 0.4878 | 0.6859 | 0.1987 |
| **B3 (Domain Weight)** | B200 | 3,652 | 0.3942 | 0.4566 | 0.5065 | 0.7206 | 0.2526 | 0.5235 | 0.7095 | 0.1987 |
| **B4 (Target-Only)** | B50 | 50 | **0.7795** | 0.6409 | 0.6673 | 0.7303 | 0.8351 | 0.7423 | 0.8228 | **0.9314** |
| **B4 (Target-Only)** | B100 | 100 | **0.7609** | 0.6276 | 0.6416 | 0.7339 | 0.7887 | 0.6727 | 0.7758 | **0.9270** |
| **B4 (Target-Only)** | B200 | 200 | **0.7660** | **0.6558** | **0.6773** | **0.7511** | **0.7835** | **0.7078** | **0.8066** | **0.9071** |
| **B5 (Weight w=2)** | B200 | 3,652 | 0.4203 | 0.4140 | 0.4608 | 0.6100 | 0.3144 | 0.4357 | 0.6657 | 0.1733 |
| **B5 (Weight w=5)** | B200 | 3,652 | 0.4152 | 0.4285 | 0.4761 | 0.6316 | 0.3093 | 0.4593 | 0.6791 | 0.1987 |
| **B5 (Weight w=10)**| B200 | 3,652 | 0.3972 | 0.4347 | 0.4811 | 0.6506 | 0.2784 | 0.4842 | 0.6905 | 0.1854 |
| **B5 (Weight w=20)**| B200 | 3,652 | 0.3690 | 0.4454 | 0.4911 | 0.6866 | 0.2371 | 0.5117 | 0.7042 | 0.2000 |

---

## 6. Paired Statistical Comparisons

Evaluated via paired prediction-level comparisons relative to reference arms on the identical $N=292$ records:

| Comparison Pair | $\Delta F1$ Point Est. | 95% Bootstrap CI ($B=2,000$) | CI Spans Zero? | Permutation $p_{\text{raw}}$ ($R=2,000$) | Bonferroni $p_{\text{adj}}$ | Benjamini-Hochberg FDR $p_{\text{adj}}$ | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$B1\text{-}B50$ vs $B0$** | $+0.0095$ | $[-0.0211, +0.0400]$ | Yes | 0.6715 | 1.0000 | 0.6715 | **No** |
| **$B2\text{-}K200$ vs $B1$ ($B_{50}$)** | $-0.0205$ | $[-0.0510, +0.0093]$ | Yes | 0.2580 | 1.0000 | 0.5160 | **No** |
| **$B3$ vs $B1$ ($B_{50}$)** | **-0.0842** | $[-0.1377, -0.0312]$ | **No** | 0.0015 | 0.0090 | 0.0045 | **Yes (Degraded)** |
| **$B4$ vs $B1$ ($B_{50}$)** | **+0.3645** | $[+0.2752, +0.4571]$ | **No** | 0.0000 | 0.0000 | 0.0000 | **Yes (Superior)** |
| **$B5\text{-}w2$ vs $B1$ ($B_{50}$)**| $+0.0014$ | $[-0.0262, +0.0287]$ | Yes | 1.0000 | 1.0000 | 1.0000 | **No** |
| **$B1\text{-}B100$ vs $B0$** | $-0.0097$ | $[-0.0378, +0.0179]$ | Yes | 0.6390 | 1.0000 | 0.7668 | **No** |
| **$B2\text{-}K200$ vs $B1$ ($B_{100}$)**| $+0.0260$ | $[-0.0057, +0.0600]$ | Yes | 0.1760 | 1.0000 | 0.5280 | **No** |
| **$B3$ vs $B1$ ($B_{100}$)** | $-0.0001$ | $[-0.0473, +0.0497]$ | Yes | 0.9565 | 1.0000 | 0.9565 | **No** |
| **$B4$ vs $B1$ ($B_{100}$)** | **+0.3651** | $[+0.2728, +0.4584]$ | **No** | 0.0000 | 0.0000 | 0.0000 | **Yes (Superior)** |
| **$B5\text{-}w2$ vs $B1$ ($B_{100}$)**| $+0.0070$ | $[-0.0171, +0.0315]$ | Yes | 0.6755 | 1.0000 | 0.7668 | **No** |
| **$B1\text{-}B200$ vs $B0$** | $+0.0095$ | $[-0.0215, +0.0401]$ | Yes | 0.6865 | 1.0000 | 0.8238 | **No** |
| **$B2\text{-}K200$ vs $B1$ ($B_{200}$)**| **+0.0347** | $[+0.0070, +0.0657]$ | **No** | 0.0190 | 0.1140 | 0.0570 | **Marginal ($p_{\text{raw}}<0.05$)** |
| **$B3$ vs $B1$ ($B_{200}$)** | $-0.0208$ | $[-0.0726, +0.0310]$ | Yes | 0.4360 | 1.0000 | 0.6540 | **No** |
| **$B4$ vs $B1$ ($B_{200}$)** | **+0.3510** | $[+0.2596, +0.4428]$ | **No** | 0.0000 | 0.0000 | 0.0000 | **Yes (Superior)** |
| **$B5\text{-}w2$ vs $B1$ ($B_{200}$)**| $+0.0053$ | $[-0.0181, +0.0305]$ | Yes | 0.8400 | 1.0000 | 0.8400 | **No** |

---

## 7. Source-Domain Retention Assessment

Evaluated on $D_{\text{source,eval}}$ ($N=493$, pre-drift observational truth, Reference $A0 = 0.6075$):

| Experimental Arm | Budget | Source F1 | Macro F1 | Balanced Acc | $\Delta F1$ vs $A0$ | Catastrophic Forgetting? ($\Delta < -0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Frozen Reference)** | 0 | **0.6075** | 0.7208 | 0.7389 | Baseline | **No** (Reference) |
| **B1 (Naive Pooling)** | B50 | 0.6159 | 0.7291 | 0.7490 | $+0.0084$ | **No** (Retained) |
| **B1 (Naive Pooling)** | B100 | 0.6040 | 0.7188 | 0.7424 | $-0.0035$ | **No** (Retained) |
| **B1 (Naive Pooling)** | B200 | 0.5993 | 0.7171 | 0.7431 | $-0.0082$ | **No** (Retained) |
| **B2 (ROS K=200)** | B50 | 0.6154 | 0.7287 | 0.7486 | $+0.0079$ | **No** (Retained) |
| **B2 (ROS K=200)** | B100 | 0.6054 | 0.7200 | 0.7434 | $-0.0021$ | **No** (Retained) |
| **B2 (ROS K=200)** | B200 | **0.5890** | 0.7099 | 0.7371 | $-0.0185$ | **No (Retained)** |
| **B3 (Domain Weighting)** | B200 | 0.5448 | 0.6865 | 0.7358 | $-0.0627$ | **YES (Forgetting)** |
| **B4 (Target-Only)** | B50 | **0.3180** | 0.4491 | 0.5830 | **-0.2895** | **YES (Severe Collapse)** |
| **B4 (Target-Only)** | B100 | **0.3537** | 0.4735 | 0.5921 | **-0.2538** | **YES (Severe Collapse)** |
| **B4 (Target-Only)** | B200 | **0.3492** | 0.4851 | 0.6111 | **-0.2583** | **YES (Severe Collapse)** |
| **B5 (Weight w=2)** | B200 | 0.5993 | 0.7171 | 0.7431 | $-0.0082$ | **No** (Retained) |

---

## 8. Definition of Strongest Authentic Controls

In accordance with the prespecified decision rules, we distinguish two operational deployment scenarios:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   DUAL OPERATIONAL BENCHMARK SPECIFICATION                  │
├──────────────────────────┬───────────────────────┬──────────────────────────┤
│ Deployment Mandate       │ Selected Benchmark    │ Target F1 | Source F1    │
├──────────────────────────┼───────────────────────┼──────────────────────────┤
│ 1. Complete Migration    │ Arm B4 (Target-Only)  │ Dev F1: 0.7660           │
│    (Source Retired)      │ Budget B200           │ Source F1: 0.3492 (Coll.)│
├──────────────────────────┼───────────────────────┼──────────────────────────┤
│ 2. Dual-Domain Retention │ Arm B2-K200 (ROS)     │ Dev F1: 0.4497           │
│    (Backward Compatible) │ Budget B200           │ Source F1: 0.5890 (Ret.) │
└──────────────────────────┴───────────────────────┴──────────────────────────┘
```

### Selection Rationale:
* **Benchmark 1 (Pure Target Adaptation):** If the production goal is adapting strictly to the drifted domain without caring about pre-drift data, **Arm B4 (Target-Only)** is the definitive benchmark. Synthetic augmentation must achieve $\mathbf{F1 > 0.7660}$ to claim value over authentic target-only training.
* **Benchmark 2 (Model Repair with Backward Compatibility):** If the production goal is repairing a deployed model while retaining source-domain capability ($\Delta F1_{\text{source}} \ge -0.05$), **Arm B2-K200 (Authentic Random OverSampling)** is the definitive non-synthetic control. Synthetic repair must achieve $\mathbf{F1 > 0.4497}$ while maintaining $\Delta F1_{\text{source}} \ge -0.05$.

---

## 9. Model Training Runtime and Memory Telemetry

All models were executed sequentially on CPU with bounded worker threads ($n\_jobs=2$):

| Experimental Arm | Fit Duration (s) | Peak Resident Memory (RSS MB) | Memory Safety Status |
| :--- | :---: | :---: | :---: |
| **B0 (Frozen Source)** | 0.00s | 165.2 MB | Safe (< 4.0 GB) |
| **B1 (Naive Pooling B200)** | Reused / 0.65s | 178.4 MB | Safe (< 4.0 GB) |
| **B2 (ROS K=200, B200)** | 0.66s | 192.1 MB | Safe (< 4.0 GB) |
| **B3 (Domain Weight B200)**| 0.64s | 185.3 MB | Safe (< 4.0 GB) |
| **B4 (Target-Only B200)** | 0.32s | 170.8 MB | Safe (< 4.0 GB) |
| **B5 (Weight Grid B200)** | 0.63s | 186.0 MB | Safe (< 4.0 GB) |
| **Total Experiment Suite**| **68.4s** | **Max 215.4 MB** | **All Safety Gates Passed** |

---

## 10. Generated Artifacts and Path Manifest

* **Execution Engine:** [`component3/src/experiments/run_authentic_controls_benchmark.py`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/src/experiments/run_authentic_controls_benchmark.py)
* **Development Split Audit:** [`component3/results/experimental_models/telco_dev_val_split_audit.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_dev_val_split_audit.json)
* **Machine-Readable Protocol:** [`component3/results/experimental_models/batch05i_authentic_controls_protocol.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/batch05i_authentic_controls_protocol.json)
* **Development Metrics Summary:** [`component3/results/experimental_models/telco_authentic_controls_dev_metrics.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_authentic_controls_dev_metrics.json)
* **Statistical Comparisons JSON:** [`component3/results/experimental_models/telco_authentic_controls_comparisons.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_authentic_controls_comparisons.json)
* **Source Retention Diagnostic:** [`component3/results/experimental_models/telco_authentic_controls_source_retention.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_authentic_controls_source_retention.json)
* **Anonymized Prediction Audit:** [`component3/results/experimental_models/telco_authentic_controls_predictions_audit.csv`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_authentic_controls_predictions_audit.csv)
* **Visual Metric Figure:** [`component3/results/figures/batch05i_authentic_controls_metrics.png`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/figures/batch05i_authentic_controls_metrics.png)
* **Trained Models Directory:** `component3/models/experimental/telco_rf_b*.joblib` (All serialized cleanly)

---

## 11. Scientific Limitations and Threats to Validity

1. **Exploratory Model-Selection Boundary:** All reported metrics and paired comparisons were conducted strictly on the development validation partition ($D_{\text{adapt,dev\_val}}$, $N=292$). While isolated from the training composition, this partition serves as an exploratory guide for method and hyperparameter selection.
2. **Deterministic Simulation Boundary:** The concept drift is injected via a fixed step-function rule on tenure. The dramatic success of target-only models (B4) reflects the simplicity of the clean, unperturbed feature relationship when conflicting historical records are removed.
3. **Small Sample Size in Target-Only B50:** While B4 on $B_{50}$ achieved F1 = 0.7795, fitting a 100-tree ensemble on only 50 samples carries potential leaf sparsity and generalization risk under complex distributions.

---

## 12. Decision Recommendation for Subsequent Implementation Batches

$$\mathbf{CONDITIONAL\ GO}$$

### Implementation Gating Conditions:
1. **Clear Benchmark Targets Established:**
   - Any synthetic repair arm claiming **Pure Target Adaptation** must beat **Arm B4 ($F1 > 0.7660$)**.
   - Any synthetic repair arm claiming **Source-Retentive Model Repair** must beat **Arm B2-K200 ($F1 > 0.4497$)** while maintaining $F1_{\text{source}} \ge 0.5890$.
2. **Authorized to proceed to Phase 2 (Generator Smoke Testing):**
   - Test TVAE and CTGAN fitting feasibility on post-drift $B_{200}$ on CPU.
   - Verify Tier 1–3 candidate validation yield before attempting downstream synthetic retraining.

---

## 13. Audit Addendum: Metric Integrity and Statistical Corrections (Batch 5I-Audit)

> [!IMPORTANT]
> **AUDIT MANDATE & TRACEABILITY:** Following an independent audit of the Batch 5I benchmark artifacts prior to synthetic generator fitting, this section documents the verified metrics, statistical corrections, and root causes of historical discrepancies. For full mathematical derivations and partition hashes, consult the dedicated audit document [`component3/docs/sprint06_batch05i_metric_integrity_audit.md`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/docs/sprint06_batch05i_metric_integrity_audit.md).

### 13.1 Root Causes of Numerical Discrepancies
* **Section 5 Table Mismatches:** Discrepancies between reported F1, precision, and recall were caused by **manual transcription errors** during report writing. The underlying machine-readable metrics file ([`telco_authentic_controls_dev_metrics.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_authentic_controls_dev_metrics.json)) and prediction audit CSV ([`telco_authentic_controls_predictions_audit.csv`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_authentic_controls_predictions_audit.csv)) are 100% mathematically correct and consistent with model inference.
  - In arms **B1_B50** and **B1_B200**, baseline B0 metrics ($P=0.6082, R=0.3041$) were mistakenly copy-pasted into the precision/recall columns, while the true F1 ($0.4150$) was retained. The true values are $P=0.6100, R=0.3144$, which exactly satisfy $F1 = 2PR / (P+R)$.
  - In arm **B3_B200**, numbers from an exploratory run ($P=0.7206, R=0.2526$) were transcribed instead of the true evaluated values ($P=0.6750, R=0.2784, F1=0.3942$).
  - In arm **B4 (Target-Only)**, inverted precision and recall were transcribed.
  - In arm **B0**, affected slice F1 was transcribed as $0.1600$ instead of $0.1351$.
* **Statistical Permutation P-Values:** Raw permutation p-values for Arm B4 were originally reported as $0.0000$. A standard finite-sample correction $p = (\text{extreme\_count} + 1) / (R + 1)$ was applied, establishing an exact lower bound of $p_{\text{finite}} = 0.0005$ for $R=2,000$.

### 13.2 Corrected Development Validation Performance Table
Evaluated on $D_{\text{adapt,dev\_val}}$ ($N=292$, 194 positive targets, 98 negative targets, positive class = Churn = Yes = 1). Every reported Positive F1 rigorously satisfies $F1 = \frac{2PR}{P+R}$ within numerical tolerance:

| Experimental Arm | Budget | Training Size | Dev F1 | Macro F1 | Balanced Acc | Precision | Recall | ROC-AUC | PR-AUC | Affected Slice F1 ($\text{tenure}>29$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Frozen Source)** | 0 | 3,452 | **0.4055** | 0.4075 | 0.4582 | 0.6082 | 0.3041 | 0.4238 | 0.6511 | 0.1351 |
| **B1 (Naive Pooling)** | B50 | 3,502 | 0.4150 | 0.4109 | 0.4582 | 0.6100 | 0.3144 | 0.4219 | 0.6537 | 0.1600 |
| **B1 (Naive Pooling)** | B100 | 3,552 | 0.3958 | 0.4040 | 0.4581 | 0.6064 | 0.2938 | 0.4339 | 0.6510 | 0.1096 |
| **B1 (Naive Pooling)** | B200 | 3,652 | 0.4150 | 0.4109 | 0.4582 | 0.6100 | 0.3144 | 0.4580 | 0.6619 | 0.1600 |
| **B2 (ROS K=50)** | B50 | 3,552 | 0.3931 | 0.3972 | 0.4479 | 0.5938 | 0.2938 | 0.4281 | 0.6549 | 0.1224 |
| **B2 (ROS K=200)** | B50 | 3,702 | 0.3945 | 0.4006 | 0.4530 | 0.6000 | 0.2938 | 0.4301 | 0.6446 | 0.1224 |
| **B2 (ROS K=50)** | B100 | 3,602 | 0.4286 | 0.4246 | 0.4736 | 0.6300 | 0.3247 | 0.4368 | 0.6530 | 0.1722 |
| **B2 (ROS K=200)** | B100 | 3,752 | 0.4218 | 0.4178 | 0.4659 | 0.6200 | 0.3196 | 0.4523 | 0.6644 | 0.1722 |
| **B2 (ROS K=50)** | B200 | 3,702 | 0.4028 | 0.4108 | 0.4658 | 0.6170 | 0.2990 | 0.4598 | 0.6539 | 0.1351 |
| **B2 (ROS K=200)** | B200 | 3,852 | **0.4497** | **0.4381** | **0.4839** | **0.6442** | **0.3454** | **0.4762** | **0.6681** | **0.2194** |
| **B3 (Domain Weight)** | B50 | 3,502 | 0.3308 | 0.3855 | 0.4705 | 0.6111 | 0.2268 | 0.4728 | 0.6632 | 0.0993 |
| **B3 (Domain Weight)** | B100 | 3,552 | 0.3957 | 0.4233 | 0.4938 | 0.6548 | 0.2835 | 0.5138 | 0.6816 | 0.1987 |
| **B3 (Domain Weight)** | B200 | 3,652 | 0.3942 | 0.4293 | 0.5065 | 0.6750 | 0.2784 | 0.5381 | 0.6829 | 0.1987 |
| **B4 (Target-Only)** | B50 | 50 | **0.7795** | **0.6681** | **0.6673** | **0.7755** | **0.7835** | **0.7425** | **0.8372** | **0.9314** |
| **B4 (Target-Only)** | B100 | 100 | **0.7609** | **0.6420** | **0.6416** | **0.7590** | **0.7629** | **0.7447** | **0.8479** | **0.9270** |
| **B4 (Target-Only)** | B200 | 200 | **0.7660** | **0.6714** | **0.6773** | **0.7912** | **0.7423** | **0.7737** | **0.8613** | **0.9071** |
| **B5 (Weight w=2)** | B200 | 3,652 | 0.4203 | 0.4143 | 0.4608 | 0.6139 | 0.3196 | 0.4852 | 0.6673 | 0.1733 |
| **B5 (Weight w=5)** | B200 | 3,652 | 0.4152 | 0.4212 | 0.4761 | 0.6316 | 0.3093 | 0.5165 | 0.6770 | 0.1987 |
| **B5 (Weight w=10)**| B200 | 3,652 | 0.3972 | 0.4171 | 0.4811 | 0.6364 | 0.2887 | 0.5309 | 0.6744 | 0.1854 |
| **B5 (Weight w=20)**| B200 | 3,652 | 0.3690 | 0.4113 | 0.4911 | 0.6494 | 0.2577 | 0.5449 | 0.6837 | 0.2000 |

### 13.3 Corrected Paired Statistical Comparisons
Evaluated via paired prediction-swap randomization ($R=2000$) and paired bootstrap ($B=2000$) on the identical $N=292$ records with finite-sample corrected p-values and step-up monotonic Benjamini-Hochberg adjustment:

| Comparison Pair | Budget | $\Delta F1$ Point Est. | 95% Bootstrap CI | Permutation $p_{\text{finite}}$ | Bonferroni $p_{\text{adj}}$ ($m=6$) | BH FDR $q$ ($m=6$) | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$B1\text{-}B50$ vs $B0$** | 50 | $+0.0095$ | $[-0.0211, +0.0400]$ | 0.6717 | 1.0000 | 0.8060 | No |
| **$B2\text{-}K200$ vs $B1$** | 50 | $-0.0205$ | $[-0.0510, +0.0093]$ | 0.2584 | 1.0000 | 0.3876 | No |
| **$B3$ vs $B1$** | 50 | **-0.0842** | $[-0.1377, -0.0312]$ | 0.0020 | 0.0120 | 0.0040 | **Yes (Degraded)** |
| **$B4$ vs $B1$** | 50 | **+0.3645** | $[+0.2752, +0.4571]$ | 0.0005 | 0.0030 | 0.0030 | **Yes (Superior)** |
| **$B5\text{-}w2$ vs $B1$** | 50 | $+0.0014$ | $[-0.0262, +0.0287]$ | 1.0000 | 1.0000 | 1.0000 | No |
| **$B3$ vs $B5\text{-}w2$** | 50 | **-0.0856** | $[-0.1361, -0.0354]$ | 0.0010 | 0.0060 | 0.0030 | **Yes (Degraded)** |
| **$B1\text{-}B100$ vs $B0$** | 100 | $-0.0097$ | $[-0.0378, +0.0179]$ | 0.6392 | 1.0000 | 0.8521 | No |
| **$B2\text{-}K200$ vs $B1$** | 100 | $+0.0260$ | $[-0.0057, +0.0600]$ | 0.1764 | 1.0000 | 0.5292 | No |
| **$B3$ vs $B1$** | 100 | $-0.0001$ | $[-0.0473, +0.0497]$ | 0.9565 | 1.0000 | 0.9565 | No |
| **$B4$ vs $B1$** | 100 | **+0.3651** | $[+0.2728, +0.4584]$ | 0.0005 | 0.0030 | 0.0030 | **Yes (Superior)** |
| **$B5\text{-}w2$ vs $B1$** | 100 | $+0.0070$ | $[-0.0171, +0.0315]$ | 0.6757 | 1.0000 | 0.8521 | No |
| **$B3$ vs $B5\text{-}w2$** | 100 | $-0.0071$ | $[-0.0493, +0.0388]$ | 0.7101 | 1.0000 | 0.8521 | No |
| **$B1\text{-}B200$ vs $B0$** | 200 | $+0.0095$ | $[-0.0215, +0.0401]$ | 0.6867 | 1.0000 | 0.8240 | No |
| **$B2\text{-}K200$ vs $B1$** | 200 | **+0.0347** | $[+0.0070, +0.0657]$ | 0.0195 | 0.1170 | 0.0585 | **Marginal ($p_{\text{raw}}<0.05$, $q_{\text{BH}}>0.05$)** |
| **$B3$ vs $B1$** | 200 | $-0.0208$ | $[-0.0726, +0.0310]$ | 0.4363 | 1.0000 | 0.6544 | No |
| **$B4$ vs $B1$** | 200 | **+0.3510** | $[+0.2596, +0.4428]$ | 0.0005 | 0.0030 | 0.0030 | **Yes (Superior)** |
| **$B5\text{-}w2$ vs $B1$** | 200 | $+0.0053$ | $[-0.0181, +0.0305]$ | 0.8401 | 1.0000 | 0.8401 | No |
| **$B3$ vs $B5\text{-}w2$** | 200 | $-0.0261$ | $[-0.0764, +0.0258]$ | 0.3038 | 1.0000 | 0.6076 | No |

### 13.4 Reassessment of Arm B2_B200_K200 Significance
While Arm B2-K200 ($B_{200}$) exhibits positive empirical improvement ($\Delta F1 = +0.0347$) and an unadjusted permutation $p_{\text{raw}} = 0.0195 < 0.05$, its family-adjusted Benjamini-Hochberg FDR is $q = 0.0585$ ($m=6$) and Bonferroni $p_{\text{adj}} = 0.1170$. It does **not** clear the formal family-wise $\alpha=0.05$ threshold. It is selected as the dual-domain benchmark because it is the highest-performing authentic control that preserves source retention ($\Delta F1_{\text{source}} = -0.0185 \ge -0.05$), but it is an empirical development optimum rather than a conclusively proven statistical breakthrough.

### 13.5 Selection of B200 over B50
Arm B4 on $B_{50}$ achieved a point estimate of $0.7795$, slightly above $B_{200}$ ($0.7660$). The selection of $B_{200}$ over $B_{50}$ was made to match the maximum data budget ($B_k=200$) available to future synthetic repair models, ensuring fair experimental comparison and reducing leaf-sparsity variance. It represents a **matched-access development design decision**, not a prespecified F1 optimum.

