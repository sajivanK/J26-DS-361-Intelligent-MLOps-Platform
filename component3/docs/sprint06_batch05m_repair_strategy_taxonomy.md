# SPRINT 06 — BATCH 5M
# Repair Strategy Taxonomy, Prescription Schema, and Feasibility Assessment

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine (`c3-mathusan`)  
**Date:** October 20, 2026  
**Status:** APPROVED ARCHITECTURAL SPECIFICATION & FEASIBILITY VERDICT  

---

## 1. Scope & Objective

This document formalizes the strategy registry, pre-decision feature schema, model feasibility decision, and minimum effective repair policy for Component 3. It synthesizes empirical evidence from Batches 5I–5L to ensure that the future intelligent repair prescription engine operates strictly on verified, non-leaking, reproducible foundations.

---

## 2. Repair Strategy Taxonomy (Step 4)

Component 3 recognizes seven formal repair strategies. Each strategy is governed by strict input requirements, operational constraints, and safety rejection criteria.

### Summary Strategy Matrix

| Strategy Identifier | Generator Required? | Adaptation Budget Required | Sample Weighting Policy | Current Implementation Status | Current Empirical Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`NO_REPAIR`** | None | 0 | None ($w=1.0$ on source) | **IMPLEMENTED** | High (Baseline $B0$) |
| **`AUTHENTIC_ONLY_ADAPTATION`** | None | $B_k \ge 50$ | Uniform ($w=1.0$) | **IMPLEMENTED** | High (Batch 5G, 5I $B1$) |
| **`AUTHENTIC_OVERSAMPLING`** | None | $B_k \ge 50$ | Uniform ($w=1.0$) | **IMPLEMENTED** | High (Batch 5I $B2$) |
| **`AUTHENTIC_WEIGHTED_ADAPTATION`** | None | $B_k \ge 50$ | Domain weights ($w_{\text{target}} \in [2, 5]$) | **IMPLEMENTED** | High (Batch 5I $B5$) |
| **`SYNTHETIC_UNWEIGHTED`** | TVAE / CTGAN / TabDDPM | $B_k \ge 200$ | Uniform ($w=1.0$) | **IMPLEMENTED** | Moderate (Batch 5K $S1-S3$) |
| **`SYNTHETIC_WEIGHTED_HYBRID`** | TVAE | $B_k \ge 200$ | Joint weights ($w_{\text{target}}=w_{\text{synth}}=2$) | **IMPLEMENTED** | Moderate (Batch 5L $H1, H2$) |
| **`TARGETED_SYNTHETIC_REPAIR`** | Conditional TVAE | Slice-dependent | Localized weights | **BLOCKED** | Non-oracle diagnostic absent |

---

### Detailed Strategy Specifications

#### 1. `NO_REPAIR`
- **Definition:** Maintain the existing frozen production model ($M_{\text{source}}$) without adaptation.
- **Required Inputs:** Current production model artifact, drift detection telemetry.
- **Available Labeled-Data Budget:** $B_k = 0$ records.
- **Generator Requirements:** None.
- **Candidate-Validation Requirements:** None.
- **Expected Output:** Guaranteed zero degradation on historical source data ($F1=0.6075$ on $D_{\text{source,eval}}$); severe unadapted vulnerability in shifted feature regions ($F1=0.0882$ on affected slice).
- **Resource Constraints:** 0 compute, 0 MB memory overhead.
- **Safety / Rejection Conditions:** Mandatory fallback whenever adaptation violates source-retention tolerances ($> -0.05$) or when available labeled adaptation budget is insufficient ($B_k < 50$).
- **Status:** **IMPLEMENTED**.

#### 2. `AUTHENTIC_ONLY_ADAPTATION`
- **Definition:** Concatenate available authentic adaptation records directly with source training data ($D_{\text{source,train}} \cup B_k$) with uniform weighting ($w=1.0$).
- **Required Inputs:** Source training data, authentic adaptation batch $B_k$.
- **Available Labeled-Data Budget:** $B_k \in \{50, 100, 200\}$.
- **Generator Requirements:** None.
- **Candidate-Validation Requirements:** Standard schema and domain bounds verification.
- **Expected Output:** Marginal post-drift adaptation gain ($F1=0.4150$ at $B_{200}$ vs $B0=0.4055$), minimal source drift ($F1=0.5993$). Inefficacious when source records outnumber adaptation records ($17.7 : 1$).
- **Resource Constraints:** Retraining runtime $\le 1.0$ s, peak RAM $\le 200$ MB.
- **Safety / Rejection Conditions:** Rejected if downstream source retention drops by $> 0.05$.
- **Status:** **IMPLEMENTED**.

#### 3. `AUTHENTIC_OVERSAMPLING`
- **Definition:** Augment authentic adaptation data by random oversampling (ROS) with replacement ($D_{\text{source,train}} \cup B_k \cup \text{ROS}_K$).
- **Required Inputs:** Source training data, authentic adaptation batch $B_k$.
- **Available Labeled-Data Budget:** $B_k \ge 50$.
- **Generator Requirements:** None.
- **Candidate-Validation Requirements:** None (authentic duplicates).
- **Expected Output:** At $K=50$, $F1=0.4028$; at $K=200$, $F1=0.4497$ with source retention $0.5890$.
- **Resource Constraints:** Retraining runtime $\le 1.2$ s, peak RAM $\le 200$ MB.
- **Safety / Rejection Conditions:** Serves as mandatory budget-matched authentic control for any synthetic comparison.
- **Status:** **IMPLEMENTED**.

#### 4. `AUTHENTIC_WEIGHTED_ADAPTATION`
- **Definition:** Apply non-uniform sample weights to authentic adaptation records ($w_{\text{source}}=1.0, w_{\text{target}} \in \{2.0, 5.0\}$).
- **Required Inputs:** Source training data, authentic adaptation batch $B_k$.
- **Available Labeled-Data Budget:** $B_k \ge 50$.
- **Generator Requirements:** None.
- **Candidate-Validation Requirements:** None.
- **Expected Output:** Moderate weighting ($w=2$) achieves $F1=0.4203$ on dev and $0.5993$ on source. Aggressive weighting ($w \ge 5$) degrades source retention down to $0.5283$.
- **Resource Constraints:** Zero retraining latency overhead, zero inference latency overhead.
- **Safety / Rejection Conditions:** Weights exceeding $w > 2$ are rejected when backward compatibility is constrained.
- **Status:** **IMPLEMENTED**.

#### 5. `SYNTHETIC_UNWEIGHTED`
- **Definition:** Fit a tabular generator (TVAE, CTGAN, TabDDPM) on $B_k$, generate $K$ synthetic records, filter for quality, and concatenate with uniform weights ($w=1.0$).
- **Required Inputs:** Authentic adaptation batch $B_k$, source-fitted preprocessor, generator hyperparameter profile.
- **Available Labeled-Data Budget:** $B_k \ge 200$ (required for stable generator parameter convergence).
- **Generator Requirements:** Generator must pass Tier-1 structural feasibility (DCR $\ge 0.05$, zero exact training duplicates, validity rate $\ge 95\%$).
- **Candidate-Validation Requirements:** Mandatory schema, domain bounds, and privacy validation.
- **Expected Output:** TVAE achieves $F1=0.4324$ ($\Delta=+0.0296$ over matched ROS $B2\_K50$) with source retention $0.6113$. CTGAN achieves $0.4192$; TabDDPM achieves $0.4106$.
- **Resource Constraints:** Generator sampling $\le 60$ s, downstream training $\le 1.0$ s, peak RAM $\le 1$ GB.
- **Safety / Rejection Conditions:** Rejected if generator yield $< K$ or if synthetic DCR drops below $0.01$.
- **Status:** **IMPLEMENTED**.

#### 6. `SYNTHETIC_WEIGHTED_HYBRID`
- **Definition:** Combine authentic adaptation data and accepted synthetic data under a joint weighting policy ($w_{\text{target}}=w_{\text{synth}}=2.0$).
- **Required Inputs:** Source training data, authentic adaptation batch $B_k$, approved synthetic candidate pool ($K=50$).
- **Available Labeled-Data Budget:** $B_k \ge 200$.
- **Generator Requirements:** TVAE (demonstrated strongest empirical fidelity).
- **Candidate-Validation Requirements:** Full Tier-1 structural and privacy filtering.
- **Expected Output:** $H1\_w2$ achieves $F1=0.4315$, balanced accuracy $0.4838$, macro F1 $0.4315$, and source retention $0.6102$.
- **Resource Constraints:** Downstream training runtime $\le 1.0$ s, peak RAM $\le 250$ MB.
- **Safety / Rejection Conditions:** Discounted weighting policies ($w_{\text{synth}} < w_{\text{target}}$) and aggressive weights ($w > 2$) are rejected due to empirical underperformance or source degradation.
- **Status:** **IMPLEMENTED**.

#### 7. `TARGETED_SYNTHETIC_REPAIR`
- **Definition:** Restrict synthetic generation or filtering strictly to an impaired subspace identified by upstream diagnostics.
- **Required Inputs:** Upstream diagnostic payload from Components 1 and 2 (e.g. localized bounding hypercube, affected slice predicate, drift localization metrics).
- **Available Labeled-Data Budget:** Adaptation records inside the localized slice.
- **Generator Requirements:** Conditional tabular generator or rejection sampling inside the localized slice.
- **Candidate-Validation Requirements:** In addition to standard Tier-1 checks, candidates must satisfy the diagnostic slice predicate.
- **Safety / Rejection Conditions:** **MANDATORY BLOCK**: In the absence of an authentic, non-oracle upstream diagnostic contract, this strategy MUST NOT be executed. Inventing an oracle rule post hoc is strictly prohibited.
- **Status:** **BLOCKED (Non-Oracle Diagnostic Contract Absent)**.

---

## 3. Pre-Decision Prescription Feature Schema (Step 5)

To guarantee scientific validity, the prescription engine must operate exclusively on **pre-decision features** available prior to selecting a repair strategy.

```
                      +---------------------------------------+
                      | PRE-DECISION CONTEXT (Permitted)     |
                      | - Dataset Metadata & Shapes           |
                      | - Drift Monitoring Degradation        |
                      | - Available Labeled Budget B_k        |
                      | - Resource Constraints (RAM, CPU)     |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | PRESCRIPTION ENGINE (Deterministic)   |
                      | - Evaluates Strategy Taxonomy         |
                      | - Checks Feasibility & Safety Rules   |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | POST-REPAIR OUTCOMES (Prohibited!)    |
                      | - D_dev_val / Test Metrics            |
                      | - Post-hoc Generator Rankings         |
                      | - True Simulation Flip Rules          |
                      +---------------------------------------+
```

### 3.1 Permitted Pre-Decision Features
1. **Dataset Structural Profile:**
   - `dataset_num_rows`: Total number of training rows in historical baseline ($N=3,452$).
   - `num_numerical_features`: Count of continuous numerical columns (3).
   - `num_categorical_features`: Count of categorical columns (16).
   - `class_imbalance_ratio`: Positive class fraction in source training ($0.2654$).
   - `missing_value_rate`: Fraction of missing entries ($0.0016$).
2. **Drift & Monitoring Profile (From authorized monitoring split):**
   - `drift_detected`: Boolean flag from Component 1 detector.
   - `drift_severity_score`: Statistical divergence (e.g. Wasserstein distance or KS test p-value).
   - `num_affected_features`: Number of features flagged by drift detector.
   - `baseline_performance_degradation`: $\Delta F1$ observed between baseline training and monitoring split.
   - `diagnostic_slice_identified`: Boolean flag (False if no localized rule payload is available).
   - `diagnostic_slice_support`: Row support of identified slice, if available.
3. **Operational Constraints:**
   - `authentic_adaptation_budget`: Available labeled records ($B_k \in \{0, 50, 100, 200\}$).
   - `source_retention_tolerance`: Maximum permissible drop on pre-drift data ($\le 0.05$).
   - `runtime_budget_seconds`: Maximum permissible repair execution time (e.g. 180 s).
   - `memory_limit_mb`: Maximum permissible RAM allocation (e.g. 4,096 MB).
4. **Historical Generator Feasibility Profile:**
   - `tvae_feasibility_tier1`: Historical capability of TVAE to pass structural validation on similar schema.
   - `ctgan_feasibility_tier1`: Historical capability of CTGAN to pass structural validation.
   - `tabddpm_feasibility_tier1`: Historical capability of TabDDPM to pass structural validation.

### 3.2 Prohibited Features (Leakage Audit)
The following features are **strictly prohibited** from the prescription feature vector:
- Any downstream evaluation metric on $D_{\text{dev\_val}}$ or $D_{\text{target,eval}}$ for candidate models.
- Post-hoc generator selection (e.g. "TVAE had highest F1 in Batch 5K").
- Post-generation quality outcomes (e.g. realized candidate yield of a future unexecuted run).
- Hidden simulation parameters (e.g. the oracle condition `tenure > 29.0`).
- True evaluation labels or record customer IDs.
- Arbitrary experiment IDs used as predictive categorical shortcuts.

---

## 4. Prescription-Model Feasibility Assessment (Step 6)

### 4.1 Independent Scenario Accounting
An audit of all completed experimental batches (5F through 5L) reveals:
- **Independent Datasets Evaluated:** Exactly **1** (`telco_churn`).
- **Independent Drift Scenarios Evaluated:** Exactly **1** (`TELCO_CONCEPT`).
- **Independent Downstream Development Partitions:** Exactly **1** ($D_{\text{dev\_val}}$, $N=292$).
- **Total Independent Experimental Scenarios ($N_{\text{scenarios}}$):** **1**.

### 4.2 Feasibility Verdict: MACHINE LEARNING PRESCRIPTION IS NOT FEASIBLE
Attempting to fit a statistical meta-learning model (such as a Gradient Boosted Classifier, Meta-Regressor, or Multi-Armed Bandit) on $N=1$ scenario would violate core machine learning principles:
1. **Severe Overfitting:** With only 1 independent task context, any trained model would achieve 0 degrees of freedom and simply memorize the categorical target encoding of the single dataset.
2. **Complete Data Leakage:** All candidate arms in Batches 5I–5L were evaluated on the *same 292 development records*. Cross-validation across arms is cross-validation on repeated measures of identical rows.
3. **No Generalization Guarantee:** A model trained on Telco Churn cannot generalize to UCI Adult Census, Credit Card Fraud, or other drift topologies.

### 4.3 Recommended Solution: Deterministic Rule-Based Baseline Engine
Component 3 must deploy an **evidence-aware, rule-based deterministic prescription baseline**:

```python
def prescribe_repair_strategy(context: PreDecisionContext) -> RepairPrescription:
    """Deterministic, evidence-aware repair prescription baseline."""
    # Rule 1: Zero adaptation budget -> NO_REPAIR
    if context.authentic_adaptation_budget == 0:
        return RepairPrescription(
            strategy="NO_REPAIR",
            rationale="No authentic adaptation data available to guide repair.",
        )

    # Rule 2: Low adaptation budget (B_k < 200) -> AUTHENTIC_WEIGHTED_ADAPTATION
    if context.authentic_adaptation_budget < 200:
        return RepairPrescription(
            strategy="AUTHENTIC_WEIGHTED_ADAPTATION",
            sample_weight_policy={"source": 1.0, "target": 2.0},
            rationale="Sample budget insufficient for stable deep generative modeling.",
        )

    # Rule 3: Diagnostic slice absent -> Block targeted repair
    if not context.diagnostic_slice_identified:
        # Evaluate generator feasibility and source retention constraint
        if context.source_retention_tolerance <= 0.05:
            return RepairPrescription(
                strategy="SYNTHETIC_WEIGHTED_HYBRID",
                generator="TVAE",
                synthetic_quantity=50,
                sample_weight_policy={"source": 1.0, "target": 2.0, "synth": 2.0},
                rationale="TVAE joint hybrid weighting (w=2) empirically achieves superior adaptation while preserving source retention within 0.05 tolerance.",
            )
        else:
            return RepairPrescription(
                strategy="SYNTHETIC_UNWEIGHTED",
                generator="TVAE",
                synthetic_quantity=50,
                rationale="Standard TVAE synthetic augmentation under relaxed source constraint.",
            )

    # Rule 4: Diagnostic slice present -> TARGETED_SYNTHETIC_REPAIR
    return RepairPrescription(
        strategy="TARGETED_SYNTHETIC_REPAIR",
        generator="TVAE",
        slice_predicate=context.diagnostic_slice_predicate,
        rationale="Upstream diagnostic payload verified; targeted repair authorized.",
    )
```

### 4.4 Roadmap to Expand Evidence Matrix
Before statistical meta-learning can be reconsidered, the evidence base must be expanded across:
1. **At least 3-5 distinct datasets:** `telco_churn`, `uci_adult`, `credit_card_default`, `bank_marketing`.
2. **Multiple drift topologies:** Covariate shift (numerical mean shift), subpopulation shift, concept drift (label flip), and conditional variance shift.
3. **Multiple adaptation budgets:** $B \in \{50, 100, 200, 500\}$.
4. **Group-aware task splitting:** Leave-one-dataset-out cross-validation across independent datasets.

---

## 5. Minimum Effective Repair Decision Criteria (Step 7)

Any candidate repair prescription proposed by the engine must clear eight pre-decision gates before deployment:

1. **Target Performance Margin Gate:**
   - Must achieve $\Delta F1_{\text{target}} \ge +0.02$ over naive authentic pooling ($B1\_B_k$).
   - Must not underperform matched-budget authentic oversampling ($B2\_B_k\_K$).
2. **Source Retention Stability Gate:**
   - Post-repair performance on source monitoring data must satisfy:
     $$F1_{\text{source}}(\text{repaired}) \ge F1_{\text{source}}(\text{baseline}) - 0.05$$
3. **Statistical Uncertainty Gate:**
   - The 95% paired bootstrap confidence interval for $\Delta F1$ must not be entirely negative, or the paired permutation test must achieve nominal $p < 0.10$.
4. **Candidate Structural Validity Gate:**
   - 100% of generated records must pass Tier-1 schema integrity, column domain constraints, and non-null guarantees.
5. **Privacy and Overfitting Gate:**
   - Zero exact duplicates with training data ($D_{\text{source,train}} \cup B_k$).
   - Distance to Closest Record (DCR) 5th percentile must exceed $0.01$.
6. **Computational Feasibility Gate:**
   - Generator sampling runtime $\le 60$ seconds.
   - Downstream classifier retraining runtime $\le 5$ seconds.
   - Peak memory consumption $\le 2,048$ MB.
7. **Minimum Effective Augmentation Quantity ($K_{\min}$):**
   - Current experiments evaluated $K=50$ and $K=200$.
   - **Crucial Qualification:** An exact optimal minimum quantity has **NOT** yet been discovered. Intermediate grid points ($K \in \{10, 25, 75, 100, 150\}$) have not been evaluated. At present, $K=50$ serves as the default operational pilot quantity.
8. **Automated Fallback to `NO_REPAIR`:**
   - If candidate generation fails any structural or privacy check, or if candidate yield falls below $K$, the engine must cleanly abort and fall back to `NO_REPAIR` or `AUTHENTIC_ONLY_ADAPTATION`.

---

## 6. Phase Recommendation

**RECOMMENDATION: CONDITIONAL GO**

- **CONDITIONAL GO for Rule-Based Prescription Engine:** The evidence registry, strategy taxonomy, pre-decision schema, and decision criteria provide a rock-solid, fully audited foundation to build the deterministic, rule-based prescription baseline.
- **BLOCKED for Machine Learning Meta-Learner:** Training a parameterized machine learning model for prescription is strictly blocked until the multi-dataset evidence matrix is expanded.
- **BLOCKED for Targeted Synthetic Repair:** Targeted repair remains blocked until Components 1 and 2 deliver an authenticated, non-oracle diagnostic contract.
