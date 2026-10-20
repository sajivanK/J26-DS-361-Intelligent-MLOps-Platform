# SPRINT 06 — BATCH 5L EXPERIMENTAL PROTOCOL
## Targeted and Domain-Weighted Synthetic Repair: Controlled Pilot Protocol

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan` | **Environment:** Windows, CPU-Only, Host Memory Constrained  
**Execution Timestamp:** 2026-10-20T14:10:00+05:30  
**Status:** PRE-REGISTERED EXPERIMENTAL PROTOCOL (Locked Prior to Model Training)  
**Governing Document:** `component3/docs/sprint06_batch05l_targeted_weighted_repair_protocol.md`  
**Associated Results Manifest:** `component3/results/experimental_models/batch05l_weighted_repair_protocol.json`  

---

## 1. Executive Summary & Research Objectives

Sprint 06 Batch 5L investigates whether **targeted synthetic augmentation** or **domain-weighted hybrid repair** can improve post-drift model performance on $D_{\text{dev\_val}}$ ($N=292$) while preserving pre-drift competence on $D_{\text{source,eval}}$ ($N=493$).

This is a **controlled, exploratory pilot**. In accordance with MLOps scientific integrity principles:
1. **Primary Comparison against Matched Controls:** All evaluations are conducted against budget-matched authentic controls ($K=50$), specifically $B2\_B200\_K50$, preventing the sample-size asymmetry identified in the Batch 5K audit.
2. **Strict Evaluation Isolation:** The locked evaluation partition $D_{\text{target,eval}}$ ($N=493$) is **neither loaded, inspected, hashed, nor queried**. Candidate screening uses exclusively authorized training information.
3. **Targeted Repair Integrity:** Non-oracle targeted repair requires legitimate diagnostic inputs from Components 1 and 2. Because no non-oracle diagnostic contract was generated for `telco_churn`, non-oracle targeted repair is formally declared **BLOCKED**. The simulated drift rule (`tenure > 29.0`) is analyzed strictly as an oracle upper-bound reference, avoiding fabricated diagnostic claims.
4. **Honest Reporting:** If weighted hybrid repair fails to outperform unweighted synthetic augmentation or authentic weighted controls, this negative result will be reported without post-hoc manipulation.

---

## 2. Partition Governance & Absolute Evaluation Isolation

| Partition Name | Record Count ($N$) | Role in Batch 5L | Access & Isolation Rules |
| :--- | :--- | :--- | :--- |
| **$D_{\text{source,train}}$** | 3,452 | Source domain training base | Authorized in all combined models ($w_{\text{source}} = 1.0$). |
| **$B200$** | 200 | Post-drift authentic adaptation sample | Authorized in adaptation training ($w_{\text{target}} \in \{1.0, 2.0, 5.0\}$). |
| **$D_{\text{dev\_val}}$** | 292 | Clean development validation split | **Primary post-drift evaluation split**. Strictly unobserved during training. |
| **$D_{\text{source,eval}}$** | 493 | Clean observational source split | **Secondary source retention evaluation**. Strictly unobserved during training. |
| **$D_{\text{target,eval}}$** | 493 | Locked final target evaluation split | **STRICTLY FORBIDDEN. Never loaded into memory or accessed.** |

### Architectural Isolation Policy
In contrast to Batch 5K (where $D_{\text{target,eval}}$ signatures were included in candidate screening), Batch 5L candidate screening derives exclusion sets **strictly from authorized training data** ($D_{\text{source,train}}$ and $B200$). Partition builder calls omit `target_eval` references entirely.

---

## 3. Targeted Repair Feasibility & Status

### Diagnostic Information Boundaries
To maintain scientific validity:
- **Genuinely Detected Slice:** Requires an upstream Component 1/2 diagnostic payload (`c2_output_to_c3_schema.json`). None exists for `telco_churn`.
- **Predefined Stress-Test Slice:** `tenure > 29.0` is the hidden ground-truth simulation rule. Using it as a generator conditioning rule constitutes an **Oracle Slice**, not an automated repair discovery.
- **Candidate Starvation in Oracle Slice:** Screening existing plausible TVAE candidates reveals that only 19 candidates exhibit `tenure > 29.0`. Meeting $K=50$ within the oracle slice would require extensive oversampling, violating CPU limits and plausibility screening.

### Formal Status: BLOCKED
Non-oracle targeted synthetic repair is formally designated as **BLOCKED** for Batch 5L. We do not invent an ad-hoc heuristic slice.

---

## 4. Domain-Weighted Hybrid Repair Architecture

Weighted hybrid repair combines sample weighting with synthetic augmentation at fixed sample budget $K=50$:

```
                       BATCH 5L COMPARISON ARMS AND SAMPLE WEIGHTS
┌────────────────────────┬─────────────┬──────────┬──────────┬───────────┬────────────────────────────────┐
│ Arm Identifier         │ Total Rows  │ w_source │ w_target │ w_synth   │ Description / Scientific Role  │
├────────────────────────┼─────────────┼──────────┼──────────┼───────────┼────────────────────────────────┤
│ B0 (Frozen Source)     │ 3,452       │ 1.0      │ -        │ -         │ Unadapted pre-drift baseline   │
│ B1_B200 (Naive Pool)   │ 3,652       │ 1.0      │ 1.0      │ -         │ Unweighted authentic pooling   │
│ B2_B200_K50 (ROS)      │ 3,702       │ 1.0      │ 1.0      │ - (dup=1) │ Primary matched-budget control │
│ B2_B200_K200 (ROS)     │ 3,852       │ 1.0      │ 1.0      │ - (dup=1) │ Extended authentic reference   │
│ B5_B200_w2 (Auth W2)   │ 3,652       │ 1.0      │ 2.0      │ -         │ Strongest authentic weighted   │
│ B5_B200_w5 (Auth W5)   │ 3,652       │ 1.0      │ 5.0      │ -         │ Authentic weighted (w=5)       │
│ B2_B200_K50_w2 (ROS W2)│ 3,702       │ 1.0      │ 2.0      │ - (dup=2) │ Matched ROS weighted (w=2)     │
│ B2_B200_K50_w5 (ROS W5)│ 3,702       │ 1.0      │ 5.0      │ - (dup=5) │ Matched ROS weighted (w=5)     │
├────────────────────────┼─────────────┼──────────┼──────────┼───────────┼────────────────────────────────┤
│ S1_TVAE_K50 (Unweighted│ 3,702       │ 1.0      │ 1.0      │ 1.0       │ Unweighted synthetic TVAE arm  │
│ H1_TVAE_K50_w2 (Joint) │ 3,702       │ 1.0      │ 2.0      │ 2.0       │ Joint hybrid weighting (w=2)   │
│ H1_TVAE_K50_w5 (Joint) │ 3,702       │ 1.0      │ 5.0      │ 5.0       │ Joint hybrid weighting (w=5)   │
│ H2_TVAE_K50_w2_disc    │ 3,702       │ 1.0      │ 2.0      │ 1.0       │ Discounted synthetic weight    │
│ H2_TVAE_K50_w5_disc    │ 3,702       │ 1.0      │ 5.0      │ 2.5       │ Discounted synthetic weight    │
└────────────────────────┴─────────────┴──────────┴──────────┴───────────┴────────────────────────────────┘
```

---

## 5. Model Architecture & Preprocessing Policy

- **Downstream Model:** `RandomForestClassifier` with invariant hyperparameters:
  `n_estimators=100`, `max_depth=10`, `class_weight='balanced'`, `random_state=42`, `n_jobs=2`, `bootstrap=True`, `criterion='gini'`.
- **Preprocessing:** `TabularDataPreprocessor(scale_numeric=True, one_hot_encode=True)` using the frozen feature mapping from `telco_preprocessor_experimental_source.joblib` (46-dimensional input space).
- **Classification Threshold:** Fixed at 0.5 across all models.

---

## 6. Pre-Registered Hypotheses & Multiple Testing Controls

### Primary Hypothesis Family
The primary hypothesis family tests whether any new weighted hybrid arm achieves statistically significant positive F1 improvement on $D_{\text{dev\_val}}$ over the primary matched control $B2\_B200\_K50$:
1. $H_1: \text{F1}(H1\_w2) > \text{F1}(B2\_K50)$
2. $H_2: \text{F1}(H1\_w5) > \text{F1}(B2\_K50)$
3. $H_3: \text{F1}(H2\_w2\_disc) > \text{F1}(B2\_K50)$
4. $H_4: \text{F1}(H2\_w5\_disc) > \text{F1}(B2\_K50)$
5. $H_5: \text{F1}(S1\_TVAE\_K50) > \text{F1}(B2\_K50)$

### Secondary Comparisons
Each weighted hybrid arm is compared against its exact authentic weighted counterpart ($H1\_w2$ vs. $B2\_K50\_w2$ and $B5\_w2$; $H1\_w5$ vs. $B2\_K50\_w5$ and $B5\_w5$).

### Statistical Testing Protocol
- Paired bootstrap 95% confidence intervals ($B=2,000$, seed 42).
- Paired two-sided permutation tests ($R=2,000$, seed 42) with finite-sample correction:
  $$p = \frac{\sum_{r=1}^R \mathbb{I}(|\Delta^{(r)}| \ge |\Delta_{\text{obs}}|) + 1}{R + 1}$$
- Step-down Holm-Bonferroni correction applied across the family of 5 primary hypotheses.

---

## 7. Success Criteria & Tolerance Thresholds

1. **Practical Adaptation Gain:** $\Delta \text{F1} \ge +0.0200$ over matched control $B2\_B200\_K50$ ($F1_{\text{dev}} \ge 0.4228$).
2. **Statistical Significance:** Multiplicity-adjusted $p_{\text{adj}} < 0.05$.
3. **Source Retention Tolerance:** Source F1 degradation on $D_{\text{source,eval}}$ must not exceed $0.0200$ relative to $B0$ ($F1_{\text{source}} \ge 0.5875$, since $B0 = 0.6075$).
4. **Advantage over Authentic Weighted Counterpart:** A hybrid arm is considered successful if and only if it outperforms both the unweighted matched control ($B2\_K50$) **and** the corresponding authentic weighted control ($B2\_K50\_w$ and $B5\_w$).
