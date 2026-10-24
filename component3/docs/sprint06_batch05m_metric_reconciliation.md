# SPRINT 06 — BATCH 5M
# Historical Metric Reconciliation and Partition Identity Audit

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine (`c3-mathusan`)  
**Date:** October 20, 2026  
**Status:** AUDITED, RECONCILED & CRYPTOGRAPHICALLY VERIFIED  

---

## 1. Executive Summary

Batch 5M consolidates all empirical evidence from Batches 5I–5L to establish an auditable, verified foundation for the future intelligent repair prescription engine. This report delivers:
1. Complete mathematical and forensic reconciliation of two historical metric discrepancies.
2. Forensic resolution of the historical $B_{50} / B_{100}$ class-count discrepancy between Batch 5E and Batch 5G.
3. Cryptographic and partition identity verification for all nested budgets ($B_{50} \subset B_{100} \subset B_{200}$) and development validation records ($D_{\text{dev\_val}}$, $N=292$).
4. Strict evaluation isolation confirmation: zero access to locked target evaluation data ($D_{\text{target,eval}}$, $N=493$) or canonical validation/test partitions.

---

## 2. Investigation and Resolution of Metric Discrepancies (Step 1)

### 2.1 Discrepancy 1: $B2\_B200\_K200$ Source Positive F1 (0.5890 vs 0.6034)

#### Forensic Summary:
- **Earlier Audited Value:** $0.5890$ (Batch 5I, Batch 5J, Batch 5K).
- **Batch 5L Narrative Table Value:** $0.6034$ (reported in markdown narrative table of `sprint06_batch05l_targeted_weighted_repair.md`).
- **Actual Saved Artifact Value in Batch 5L:** $\mathbf{0.5890}$ (stored in `batch05l_weighted_repair_results.json`).

#### Forensic Trace:
1. **Model Artifact:** [`component3/models/experimental/telco_rf_b2_b200_k200.joblib`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/models/experimental/telco_rf_b2_b200_k200.joblib) (SHA-256: `95e0c524b0a3c20067675f9227f2f18aa002a281729452b4dc25ee229dbe07e2`).
2. **Evaluation Partition:** Independent observational source evaluation partition $D_{\text{source,eval}}$ ($N=493$, pre-drift ground truth labels, 131 positive, 362 negative).
3. **Execution Pipeline:**
   ```python
   # Evaluated in run_targeted_weighted_repair.py:
   bundle_seval = source_prep.transform(df_source_eval)
   y_pred_seval = m_b2_k200.predict(bundle_seval.X)
   f1_pos = f1_score(bundle_seval.y, y_pred_seval, pos_label=1, zero_division=0)
   # Result: exactly 0.5890
   ```
4. **Root Cause Analysis:**
   - The saved JSON artifact [`batch05l_weighted_repair_results.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/batch05l_weighted_repair_results.json) under `source_retention_metrics["B2_B200_K200"]["positive_f1"]` correctly records `0.589`.
   - In that same JSON artifact, arm **$H2\_TVAE\_K50\_w2\_disc$** exhibits a source positive F1 of **$0.6034$**.
   - During the manual synthesis of the markdown summary table in `sprint06_batch05l_targeted_weighted_repair.md`, the value `0.6034` from $H2\_w2\_disc$ was inadvertently transcribed into the $B2\_B200\_K200$ row.
5. **Reconciliation Verdict:** **RESOLVED & VERIFIED**. The true, reproducible source retention F1 for $B2\_B200\_K200$ across all batches is **$0.5890$**. No underlying model or data discrepancy exists; the mismatch was strictly a narrative typographical transposition.

---

### 2.2 Discrepancy 2: Affected Development Slice with `tenure > 29` (148 vs 133)

#### Forensic Summary:
- **Earlier Reported Count:** 148 (Batch 5I, Batch 5J-Audit, Batch 5K).
- **Batch 5L Narrative Count:** 133 (reported in Section 6.2 narrative of `sprint06_batch05l_targeted_weighted_repair.md`).
- **Actual Partition Count:** $\mathbf{148}$ affected records, $\mathbf{144}$ preserved records ($148 + 144 = 292$).

#### Forensic Trace:
1. **Partition Specification:** Clean development validation partition $D_{\text{dev\_val}}$ ($N=292$), constructed by isolating unused records from $D_{\text{target,adapt\_pool}}$ ($N=493$) that share no customer ID or feature tuple with $B200$ ($N=200$).
2. **Ground Truth Predicate:** $\text{tenure} > 29.0$.
3. **Partition Row Audit:**
   - Exact query: `(clean_dev_val["tenure"] > 29.0).sum()` yields exactly **148** rows.
   - Exact query: `(clean_dev_val["tenure"] <= 29.0).sum()` yields exactly **144** rows.
   - Sum: $148 + 144 = 292$ rows.
4. **Stored Artifacts:**
   - In [`telco_dev_val_split_audit.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/telco_dev_val_split_audit.json):
     `"affected_slice_tenure_gt_29_count": 148`
     `"preserved_slice_tenure_le_29_count": 144`
   - In [`batch05l_weighted_repair_results.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/batch05l_weighted_repair_results.json):
     `"affected_slice_metrics": { "B0": { "total_support": 148 } }`
     `"preserved_slice_metrics": { "B0": { "total_support": 144 } }`
5. **Origin of the Number 133:**
   - In Batch 5K and Batch 5L, the overall confusion matrix on $D_{\text{dev\_val}}$ for $B1\_B200$ and $S2\_CTGAN\_K50$ is `[[59, 39], [133, 61]]`.
   - The False Negative ($FN$) count for these models is **133**, and the remaining instances are $292 - 133 = \mathbf{159}$.
   - During the drafting of Section 6.2 in `sprint06_batch05l_targeted_weighted_repair.md`, the model's False Negative count (133) was erroneously copied into the text description as the slice support count.
6. **Reconciliation Verdict:** **RESOLVED & VERIFIED**. The physical development validation partition $D_{\text{dev\_val}}$ has always contained exactly **148** affected slice records and **144** preserved slice records.

---

## 3. Investigation and Resolution of Historical $B_{50} / B_{100}$ Class-Count Discrepancies (Step 2)

### 3.1 The Manifest Discrepancy
In `component3/results/adaptation_splits/telco_churn_adaptation_split_manifest.json` (created in Batch 5E), the serialized budget metadata reported:
```json
"nested_budget_counts": {
  "K_50": { "positive_count": 0, "negative_count": 50, "positive_rate": 0.0 },
  "K_100": { "positive_count": 0, "negative_count": 100, "positive_rate": 0.0 },
  "K_200": { "positive_count": 0, "negative_count": 200, "positive_rate": 0.0 }
}
```
However, Batch 5G and Batch 5I reported positive churn counts of 10 in $B_{50}$, 24 in $B_{100}$, and 53 in $B_{200}$.

### 3.2 Root Cause Analysis
An inspection of `component3/src/splitting/experimental_splitter.py` revealed:
1. In the dataset registry metadata for `telco_churn`, `target_mapping` is defined as `{"No": 0, "Yes": 1}`.
2. In `ExperimentalPartitionBuilder.build_splits` (line 299):
   ```python
   pos_c = int((b_df[target_col] == (metadata.target_mapping.get("Yes", "Yes") if metadata else "Yes")).sum())
   ```
3. Because `metadata.target_mapping.get("Yes", "Yes")` returned the integer `1`, the expression compared string column values (`"Yes"`, `"No"`) against integer `1`:
   ```python
   "Yes" == 1  # Evaluates to False in Python
   ```
4. As a result of this type mismatch, `pos_c` evaluated to `0` for all budgets during manifest generation in Batch 5E.
5. Crucially, the actual row partitioning logic (lines 268–275) did not use this flawed comparison for record extraction; it performed standard stratified permutation.
6. When the actual dataframes were inspected directly in Batch 5G and Batch 5I, the true observational class counts were correctly extracted:
   - $B_{50}$: No: 40, Yes: 10
   - $B_{100}$: No: 76, Yes: 24
   - $B_{200}$: No: 147, Yes: 53

### 3.3 Post-Drift Transformation Re-Verification
Under the frozen `TELCO_CONCEPT` ground truth rule ($\text{tenure} > 29.0 \implies \text{Churn} \gets \text{NOT}(\text{Churn})$):
- **$B_{50}$ (50 rows):** 25 records with $\text{tenure} > 29$ inverted $\rightarrow$ Post-Drift: **Yes: 31, No: 19** (62.0% positive).
- **$B_{100}$ (100 rows):** 48 records with $\text{tenure} > 29$ inverted $\rightarrow$ Post-Drift: **Yes: 60, No: 40** (60.0% positive).
- **$B_{200}$ (200 rows):** 97 records with $\text{tenure} > 29$ inverted $\rightarrow$ Post-Drift: **Yes: 124, No: 76** (62.0% positive).

---

## 4. Verification of Stable Partition and Budget Identity

Deterministic reconstruction using `ExperimentalPartitionBuilder.build_splits(seed=42)` confirms exact cryptographic row membership:

### 4.1 Partition Sizes & Fingerprints
| Partition Name | Realized Rows | Feature Signature Set Size | SHA-256 Customer ID Membership Hash |
| :--- | :---: | :---: | :--- |
| **$D_{\text{source,train}}$** | 3,452 | 3,431 | `733cb0a5445fc6f76c5b9f7a77e5d1ec5a457493a776c57f0bc62a373111f71a` |
| **$D_{\text{source,eval}}$** | 493 | 493 | `81d4e410f9d92b95c25608bfa846757b1bc1851e44c2075727e46ea76d2994eb` |
| **$D_{\text{target,adapt\_pool}}$** | 493 | 493 | `5f0b5aa2df954a26df1f4d99c4c16a19f6a5b6cb14842d3adad21074e2dbe655` |
| **$D_{\text{dev\_val}}$ (Clean)** | 292 | 292 | `22acbcf6b38b0add69a2a44d6076d1d46bcdd67757a44f891581ff1a0e8182bb` |
| **$D_{\text{target,eval}}$ (Locked)**| 493 | 493 | `[LOCKED — NEVER LOADED OR QUERIED]` |

### 4.2 Nested Budget Containment
- $B_{50} \subset B_{100}$: **TRUE** ($50 / 50$ customer IDs present in $B_{100}$).
- $B_{100} \subset B_{200}$: **TRUE** ($100 / 100$ customer IDs present in $B_{200}$).
- Budget Hashes:
  - $B_{50}$: `ed25342b6389314548e0559681dfaddef8123bd43605d17ef368f24fc0736cbb`
  - $B_{100}$: `b505b428877bcde01b390393b84ba59e4f543db238c1eab3076b0d81ea4a03d7`
  - $B_{200}$: `f8029325865a423583b43e3b7799cc197110f7f7c0768a8b7bb7c1a3e3131194`

### 4.3 Evaluation Isolation Summary
- Customer ID Overlap between $B200$ and $D_{\text{dev\_val}}$: **0**.
- Exact Feature Signature Overlap between $B200$ and $D_{\text{dev\_val}}$: **0**.
- Evaluation Contamination Status: **STRICT PASS**.
