# SPRINT 06 — BATCH 5H TECHNICAL SPECIFICATION
## Fair Drift Repair Benchmark Design, Evaluation Isolation, and Matched Control Protocol

**Project:** J26-DS-361 Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Branch:** `c3-mathusan` | **Base Commit:** `9a3b2b9`  
**Execution Timestamp:** 2026-10-20T12:05:00+05:30  
**Status:** BENCHMARK DESIGN AND EVALUATION ISOLATION SPECIFICATION (No Model Training or Generator Fitting Executed in this Batch)  
**Governing Protocol:** [`batch05h_benchmark_protocol.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/batch05h_benchmark_protocol.json)

---

## 1. Executive Summary & Audit of Completed Experiments

### 1.1 Context and Repository Audit
In Sprint 06 Batches 5E, 5F, and 5G, Component 3 established the experimental foundation for post-drift adaptation benchmarking on the Telco Customer Churn dataset under the frozen `TELCO_CONCEPT` simulation:
* **Batch 5E:** Established the four-way partition scheme ($D_{\text{source,train}}$, $D_{\text{source,eval}}$, $D_{\text{target,adapt\_pool}}$, $D_{\text{target,eval}}$) with 0 cross-partition index or feature-group overlap, verified canonical training fingerprint `0348ebce7a8c776d`, and corrected `TELCO_COV` label provenance to `STRESS_TEST_UNCERTAIN`.
* **Batch 5F:** Trained and frozen the experimental source Random Forest ($M_{\text{source}}$, Arm $A0$) on $D_{\text{source,train}}$ ($N=3,452$), demonstrating substantial degradation under `TELCO_CONCEPT` on $D_{\text{target,eval}}$ ($N=493$): F1 decreased from 0.6075 to 0.4139 ($\Delta F1_{\text{deg}} = -0.1936$).
* **Batch 5G:** Pre-registered and evaluated Arm $A1$ (Authentic-Only Adaptation Control) across nested budgets $B_{50}, B_{100}, B_{200}$ carved from $D_{\text{target,adapt\_pool}}$, observing that naive unweighted data concatenation yielded no statistically significant target F1 improvement over $A0$ ($\Delta F1(A1\text{-}B200) = +0.0001$, 95% CI: $[-0.0121, +0.0119]$, $p_{\text{adj}} = 1.0000$).

### 1.2 Verification of Existing Artifacts and Hashes
All existing model artifacts, preprocessors, split manifests, and baseline models were audited and verified to match their inviolable cryptographic fingerprints:

| Artifact Name | File Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Canonical Train Manifest** | `component3/results/split_manifests/telco_churn_split_manifest.json` | `e80e253f91e8712143e15e05dbb4d5c8d2039a69beb23423eec5a87ca217de68` | **LOCKED & VERIFIED** |
| **Experimental Split Manifest**| `component3/results/adaptation_splits/telco_churn_adaptation_split_manifest.json` | `34fa2bb2bcfad28961287343368ceeb87803f3f04542a57595fa89bc444d9c59` | **FROZEN & VERIFIED** |
| **Pairwise Leakage Audit** | `component3/results/adaptation_splits/telco_churn_adaptation_group_overlap_audit.json` | `884d9d6983b1f589ff302749438f93bda84c5d4a2a7d5d90e7f793c0a2008518` | **PASS (0 Overlap)** |
| **Historical Baseline RF** | `component3/models/baselines/telco_churn_random_forest.joblib` | `69dc82cfa7dd8f3f3ca0bf5cbd60cbbd86bafebd4db4a8db282f21543bde4701` | **INVIOLABLE & UNTOUCHED** |
| **Historical Preprocessor** | `component3/models/baselines/telco_churn_preprocessor.joblib` | `a9eb79cd6c519b99a1ef5372d5418afaacb01048498bb46617d1f6e012ed45c0` | **INVIOLABLE & UNTOUCHED** |
| **Batch 5F Source Model ($A0$)**| `component3/models/experimental/telco_rf_experimental_source.joblib` | `256a692be95a0ea840d9afe03891131bc15c93090d72e3c62f815ac337b75bd0` | **FROZEN & UNTOUCHED** |
| **Batch 5F Source Preprocessor**| `component3/models/experimental/telco_preprocessor_experimental_source.joblib` | `d5aa24bbfeae92942b04f145f8a004523c9ce057f848148b8fe1ad742337d1d2` | **FROZEN & UNTOUCHED** |
| **Batch 5G Model $A1\text{-}B50$** | `component3/models/experimental/telco_rf_a1_b50.joblib` | `3e382d61d120a16b233a7fca5477d9cfa3a0ae2326771e8431940989f66679b4` | **FROZEN & UNTOUCHED** |
| **Batch 5G Model $A1\text{-}B100$**| `component3/models/experimental/telco_rf_a1_b100.joblib` | `6c10eb83e60fc31c80f4f9f7ba372d8e404b998cfb6140ccf8c3e80df046bc37` | **FROZEN & UNTOUCHED** |
| **Batch 5G Model $A1\text{-}B200$**| `component3/models/experimental/telco_rf_a1_b200.joblib` | `7524ce8df8ecbeea5aa75be4116499806fe9a9bc44d9f67a27cb46e036e6ba68` | **FROZEN & UNTOUCHED** |

---

## 2. Scientific Corrections and Qualification of Prior Claims

The scientific findings of Batch 5G provided critical empirical insight, but preliminary narrative statements in the Batch 5G technical report required essential scientific qualification. In Batch 5H, we formally establish six methodological corrections:

### Correction 1: Scope of the Authentic Inefficacy Finding
* **Historical Statement:** *"We have proved that authentic labeled data alone at realistic operational budgets ($N \le 200$) is statistically incapable of repairing model degradation under concept drift."*
* **Scientific Reality:** Batch 5G proved strictly that **naive unweighted pooling** ($D_{\text{source,train}} \cup B_k$) is inefficacious when 3,452 pre-drift records overwhelm 50–200 adaptation records. In the vulnerable slice ($\text{tenure} > 29.0$), 1,719 pre-drift records displaying low historical churn outnumber the 25–97 inverted target records by a ratio of $\mathbf{17.7 : 1}$. Decision tree bootstrap samples and Gini impurity calculations remain heavily dominated by the pre-drift records.

### Correction 2: Non-Synthetic Alternatives Not Yet Precluded
* **Historical Statement:** *"This formally establishes the scientific necessity for synthetic data repair (Track B)."*
* **Scientific Reality:** This conclusion was premature. Standard machine learning literature provides powerful non-synthetic mechanisms to handle distribution imbalance and domain adaptation:
  1. *Sample Weighting:* Assigning higher sample weight to target records ($w_{\text{target}} \gg 1.0$) directly counterbalances the $17.7 : 1$ historical record dominance.
  2. *Target Random OverSampling (ROS):* Duplicating authentic target records expands representation in tree leaves without requiring neural generative approximations.
  3. *Target-Only Training:* Training strictly on the labeled adaptation sample ($B_k$) discards contradictory pre-drift records entirely.
  If any of these non-synthetic methods achieve performance recovery, the assertion that synthetic generation is "necessary" is disproved. Synthetic data repair is only justified if it provides measurable, statistically verified improvement over **both** naive pooling and strong non-synthetic controls.

### Correction 3: Synthetic Nature of the Concept Shift Ground Truth
* The `TELCO_CONCEPT` ground-truth labels are generated programmatically via:
  $$\text{If } \text{tenure} > 29.0 \implies \text{Churn} \gets \text{NOT}(\text{Churn});\quad \text{Else } \text{Churn} \gets \text{Churn}$$
* This transformation defines a valid, reproducible synthetic benchmark (`SYNTHETIC_CONCEPT_TRUTH`) designed to stress-test adaptation algorithms. However, it represents an artificial stress simulation, not observed market consumer behavior. All findings must be contextualized as synthetic concept-shift benchmarks.

### Correction 4: Pre-Drift Generator Invalidation for Concept Shift
* Existing pre-drift generator checkpoints (`telco_churn_ctgan_model.joblib`, `telco_churn_tvae_model.joblib`, `telco_churn_tabddpm_model.joblib`) were fitted exclusively on pre-drift positive records.
* Because they learned $P_{\text{source}}(X, Y)$, they generate samples where `tenure > 29.0` exhibits low churn. Sampling from pre-drift generators to "repair" a model under `TELCO_CONCEPT` merely amplifies the outdated distribution, exacerbating degradation. Pre-drift generators cannot automatically learn or repair shifted concept rules without post-drift data access.

### Correction 5: Exposure History of $D_{\text{target,eval}}$
* The target evaluation partition $D_{\text{target,eval}}$ ($N=493$) has been accessed during Batch 5F (to establish $A0$ degradation) and Batch 5G (to evaluate $A0, A1\text{-}B50, A1\text{-}B100, A1\text{-}B200$).
* Because its metrics and difficulty are now empirically known to the research team, $D_{\text{target,eval}}$ can no longer serve as an unexposed, blind test partition during method development. Using it to tune sample weights, select synthetic budgets $K$, or prune generator checkpoints would introduce test-set snooping.

### Correction 6: Development Evaluation Isolation Mandate
* Future method development, hyperparameter selection, and synthetic quantity tuning must be conducted exclusively on an isolated development validation protocol drawn from the permitted training-side records ($D_{\text{target,adapt\_pool}}$).
* Future evaluations on $D_{\text{target,eval}}$ must be preserved as a single-pass locked comparison and formally documented as exploratory-confirmatory given prior exposure history.

---

## 3. Evaluation Isolation Architecture and Development Protocol

To eliminate data snooping while honoring existing partition boundaries, we define a rigorous three-tier data governance protocol:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PERMITTED CANONICAL DEVELOPMENT SCOPE (N = 4,931)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [1. Pre-Drift Source Domain]                                               │
│  ├── D_source,train (N = 3,452) ──► Fits M_source and training base         │
│  └── D_source,eval  (N = 493)   ──► Evaluates Source-Domain Retention       │
│                                                                             │
│  [2. Permitted Training-Side Adaptation Sandbox: D_target,adapt_pool N = 493]│
│  ├── Adaptation Budgets (B_k)   ──► B50 (N=50), B100 (N=100), B200 (N=200)   │
│  │     └── Used for: Generator fitting, authentic retraining, weights       │
│  └── D_adapt,dev_val (N = 293)  ──► Residual records (indices 200..492)     │
│        └── Used for: Tuning weights, selecting K*, screening filters        │
│        └── Audit: Exactly 0 duplicate-group overlap with B200               │
│                                                                             │
│  [3. Locked Post-Drift Evaluation Domain]                                   │
│  └── D_target,eval (N = 493)    ──► SINGLE-PASS LOCKED COMPARATIVE REPORT   │
│        └── NEVER touched during development or hyperparameter tuning        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
  [4. SEALED INVIOLABLE PARTITIONS: Canonical Val (1,056), Canonical Test (1,056)]
  └── ZERO ACCESS PERMITTED UNDER ANY CIRCUMSTANCES
```

### 3.1 Development Validation Split Verification and Group Isolation
* We investigated the structure of $D_{\text{target,adapt\_pool}}$ ($N=493$):
  * **Nested adaptation budgets:** $B_{50} \subset B_{100} \subset B_{200}$ are drawn from the pool ($N=200$).
  * **Residual adaptation pool:** The remaining unused pool contains $N=293$ records.
* **Granular Duplicate-Group Overlap Audit:**
  * In the canonical Telco population, identical feature rows exist across distinct subscriber IDs.
  * In our automated pairwise cross-partition audit between $B_{200}$ and the raw unused pool ($N=293$), we discovered **exactly one cross-partition duplicate feature pair**: subscriber `9122-UMROB` in $B_{200}$ has an identical feature vector to subscriber `2676-ISHSF` in the raw unused pool.
  * **Leakage-Safe Resolution:** To achieve strict group-safe isolation with **zero duplicate-group leakage**, subscriber `2676-ISHSF` is explicitly excluded from the development validation set.
  * **Clean Development Validation Partition ($D_{\text{adapt,dev\_val}}$):** Yields $N = 292$ records with **verified 0 feature-group overlap** with $B_{200}$.
* **Development Ground Truth:** The clean $N=292$ records are transformed under the identical frozen `TELCO_CONCEPT` rule, providing an independent post-drift development validation set.
* **Role:** All algorithm tuning (selecting optimal sample weights $w^*$, finding optimal synthetic quantity $K^*$, comparing generator architectures, evaluating Tier 1–3 screening filters) must be executed strictly against $D_{\text{adapt,dev\_val}}$ ($N=292$), keeping $D_{\text{target,eval}}$ completely locked.

---

## 4. Strong Non-Synthetic Adaptation Controls (Arms B0–B5)

To establish whether synthetic data generation provides genuine added value beyond standard data-centric and cost-sensitive learning techniques, we formulate six strictly matched authentic control arms:

| Control Arm | Identifier | Source Rows | Target Rows | Synthetic Rows | Sample Weighting Policy | Research Objective |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Frozen Source Model** | **B0** | 3,452 | 0 | 0 | Uniform ($w=1.0$) | Quantifies baseline degradation under drift ($\Delta F1_{\text{deg}}$). |
| **Naive Authentic Pooling** | **B1** | 3,452 | $B_k \in \{50, 100, 200\}$ | 0 | Uniform ($w=1.0$) | Evaluates standard concatenation ($D_{\text{source,train}} \cup B_k$). Historical Batch 5G benchmark. |
| **Authentic OverSampling (ROS)**| **B2** | 3,452 | $B_k$ | 0 ($+K$ ROS) | Uniform ($w=1.0$) on duplicated rows | Disentangles raw sample size / row duplication from synthetic diversity. |
| **Balanced Domain Weighting** | **B3** | 3,452 | $B_k$ | 0 | $w_{\text{target}} = N_{\text{source}} / B_k$; $w_{\text{source}} = 1.0$ | Tests cost-sensitive rebalancing without adding records. |
| **Target-Only Training** | **B4** | 0 | $B_k$ | 0 | Balanced class weights | Eliminates contradictory pre-drift records entirely. Tests whether small authentic data alone is superior. |
| **Prespecified Target Weighting**| **B5** | 3,452 | $B_k$ | 0 | $w_{\text{target}} \in \{2, 5, 10, 20\}$; $w_{\text{source}} = 1.0$ | Fixed heuristic weighting grid avoiding tuning leakage. |

### 4.1 Resolving Label Conflict in the Vulnerable Slice
* Under `TELCO_CONCEPT`, customers with $\text{tenure} > 29.0$ have opposite churn labels in source vs. target.
* In $D_{\text{source,train}}$, there are 1,719 records in this slice (predominantly `Churn='No'`).
* In $B_k$, there are 25 ($B_{50}$), 48 ($B_{100}$), or 97 ($B_{200}$) records in this slice (predominantly `Churn='Yes'`).
* **Why B1 Fails:** In unweighted pooling (B1), the split criterion evaluates Gini impurity on all records. The 1,719 historical records outvote the 25–97 target records by $\approx 18 : 1$.
* **How B2 & B3 Address This:** 
  * In B2 (ROS), duplicating target records $K$ times increases their frequency in bootstrap draws.
  * In B3 (Domain Weighting), setting $w_{\text{target}} = \frac{3,452}{B_k}$ gives the target records aggregate weight equal to the entire source dataset, ensuring target records have substantial voting power in conflicting leaf nodes.
* **How B4 Addresses This:** B4 eliminates the conflicting 3,452 source records completely. For $B_{200}$ ($N=200$, 124 positive churn, 76 negative churn), a balanced Random Forest can learn the post-drift concept without interference.

---

## 5. Synthetic Repair Arms (Arms S1–S4)

Synthetic repair is only scientifically justified if it outperforms **both** naive pooling (B1) and the strongest authentic control (B2, B3, B4, or B5). We define four structured synthetic repair arms:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       SYNTHETIC REPAIR ARMS MATRIX                         │
├────┬─────────────────────────────┬────────────────────────┬────────────────┤
│ Arm│ Description                 │ Training Composition   │ Screening Tier │
├────┼─────────────────────────────┼────────────────────────┼────────────────┤
│ S1 │ Raw Synthetic Augmentation  │ Source + B_k + Syn(K)  │ Tier 1 (Schema)│
│ S2 │ Quality-Screened Synthetic  │ Source + B_k + Syn(K)  │ Tier 1 + 2 + 3 │
│ S3 │ Slice-Targeted Synthetic    │ Source + B_k + Syn(K)  │ Tier 1-3+Slice │
│ S4 │ Target-Only + Synthetic     │ B_k + Syn(K)           │ Tier 1 + 2 + 3 │
└────┴─────────────────────────────┴────────────────────────┴────────────────┘
```

### 5.1 Generator Selection and Feasibility Analysis

#### 1. TVAE (Tabular Variational Autoencoder) — PRIMARY RECOMMENDED
* **Feasibility:** High. TVAE trains in $< 15$ seconds on CPU for $N=200$.
* **Small-Sample Behavior:** VAE latent space provides smooth regularization. To prevent batch dimension failure on small samples, batch size must be bounded: $\text{batch\_size} = \min(N_{\text{adapt}}, 32)$.
* **Conditioning:** Joint distribution $P(X, Y)$ trained by passing features and target label together.

#### 2. CTGAN (Conditional Tabular GAN) — SECONDARY BENCHMARK
* **Feasibility:** Moderate. CTGAN trains in $\approx 45$–$60$ seconds on CPU for $N=200$.
* **Small-Sample Behavior:** PacGAN architecture groups pac samples (default $\text{pac}=10$). For $N=50$ or $N=100$, batch size must be a multiple of 10 (e.g., $\text{batch\_size} = 30$). Discriminator instability is a known risk on small samples.
* **Conditioning:** Joint distribution $P(X, Y)$ using discrete column specification.

#### 3. TabDDPM (Diffusion Model) — EXPLORATORY PILOT ONLY
* **Feasibility:** Computationally heavy on CPU. Training 10 epochs with 100 diffusion timesteps on CPU requires substantial memory and step latency.
* **Protocol Restriction:** TabDDPM must be evaluated only in a focused pilot at $B_{200}, K=50$, never across large combinatorial grids.

### 5.2 Synthetic Label Assignment and Benchmark Assumptions
1. **Joint Distribution Modeling ($P(X, Y)$):**
   * The generator fits on $(X, Y)$ from the post-drift adaptation sample $B_k$.
   * During generation, the model samples both synthetic features $\hat{X}$ and synthetic label $\hat{Y}$.
   * **Advantage:** Unsupervised generative modeling; no oracle access assumed.
2. **Oracle Benchmark Assumption (Simulated Rule Labeling):**
   * If a generator models only features $P(X)$ and synthetic labels are assigned via the frozen concept rule ($\text{tenure} > 29.0 \implies \text{invert Churn}$), this assumes an active oracle exists in the environment.
   * **Scientific Constraint:** If rule-based synthetic labeling is tested, it must be explicitly designated as an **Oracle-Assisted Upper Bound**, not an autonomous synthetic repair.

### 5.3 Deduplication and Privacy Protection (Near-Duplicate Guard)
* To ensure the generator does not merely memorize $B_k$, we compute the Distance to Closest Record (DCR) in standardized Gower space.
* Any synthetic candidate with $\text{DCR} < 10^{-4}$ relative to $B_k$ is classified as an **authentic near-duplicate** and rejected. If $> 50\%$ of generated records are duplicates of $B_k$, the generator has collapsed into empirical oversampling (ROS).

---

## 6. Strictly Matched Comparison Design

To isolate the precise mechanism responsible for any observed performance change, comparisons must hold information access, sample sizes, and computational budgets strictly constant:

| Evaluation Question | Comparison Pair | Controlled Variables | Isolated Factor |
| :--- | :--- | :--- | :--- |
| **Row Volume vs. Synthetic Diversity** | **S1 vs. B2** | Identical training rows ($3,452 + B + K$), identical authentic rows ($3,452 + B$), identical classifier | Whether $K$ novel synthetic records outperform $K$ duplicated authentic records (ROS). |
| **Synthetic Density vs. Domain Weighting** | **S2 vs. B3** | Identical authentic records ($3,452 + B$), identical information access | Whether generating synthetic density provides value beyond cost-sensitive sample weighting. |
| **Quality Screening Value** | **S2 vs. S1** | Identical generator, identical $K$, identical authentic records | Whether Tier 2 hard safety and Tier 3 billing plausibility filtering improves downstream utility. |
| **Drift-Targeted vs. Global Generation** | **S3 vs. S2** | Identical generator, identical $K$, identical screening | Whether conditioning generation on the vulnerable slice ($\text{tenure} > 29.0$) outperforms global sampling. |
| **Target-Only Synthetic Amplification** | **S4 vs. B4** | 0 source records, identical authentic target records ($B_k$) | Whether synthetic expansion improves models trained exclusively on target data. |

---

## 7. Synthetic Repair Quantity Study ($K$) and Minimum Effective Repair

### 7.1 Quantity Grid
We explore five prespecified synthetic quantities:
$$K \in \{0, 50, 100, 200, 500\}$$
* $K=0$: Unaugmented authentic baseline.
* $K=50$: Conservative repair ($0.25 \times$ to $1.0 \times$ adaptation budget).
* $K=100$: Balanced 1:1 synthetic-to-authentic ratio for $B_{100}$.
* $K=200$: High-volume augmentation (matching Notebook 05 pilot).
* $K=500$: Stress-testing saturation, noise injection, and distribution drift.

### 7.2 Minimum Effective Repair Criterion
A synthetic repair configuration ($S_i, K^*$) is declared **scientifically effective** if and only if all five pre-registered conditions are met simultaneously on $D_{\text{adapt,dev\_val}}$:

$$\text{Effective Repair} \iff \begin{cases}
1.\ \Delta F1_{\text{dev\_val}} = F1(S_i, K^*) - \max_{j} F1(B_j) > 0.0 \\
2.\ \text{Lower bound of paired bootstrap 95\% CI for } \Delta F1_{\text{dev\_val}} > 0.0 \\
3.\ \text{Source retention: } \Delta F1_{\text{source}} \ge -0.05 \text{ on } D_{\text{source,eval}} \\
4.\ \text{Validation acceptance rate: } \ge 50\% \text{ without quota duplication} \\
5.\ \text{Total CPU runtime: } \le 900 \text{ seconds (15 minutes)}
\end{cases}$$

If no tested configuration satisfies all five criteria, **Component 3 MUST ABSTAIN** and report that non-synthetic adaptation controls are superior.

---

## 8. Primary and Secondary Evaluation Metrics

### 8.1 Metric Definitions

| Metric Category | Metric Name | Evaluation Partition | Mathematical Formula / Scope | Research Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Primary** | **Positive-Class F1** | $D_{\text{adapt,dev\_val}}$ / $D_{\text{target,eval}}$ | $\frac{2 \cdot P \cdot R}{P + R}$ for $\text{Churn}=\text{'Yes'}$ | Primary target metric under class imbalance. |
| **Secondary** | **Macro F1** | $D_{\text{adapt,dev\_val}}$ / $D_{\text{target,eval}}$ | $\frac{1}{2}(F1_{\text{pos}} + F1_{\text{neg}})$ | Overall balance across both classes. |
| **Secondary** | **Balanced Accuracy** | $D_{\text{adapt,dev\_val}}$ / $D_{\text{target,eval}}$ | $\frac{1}{2}(\text{TPR} + \text{TNR})$ | Threshold-independent classification accuracy. |
| **Secondary** | **Precision & Recall** | $D_{\text{adapt,dev\_val}}$ / $D_{\text{target,eval}}$ | $P = \frac{\text{TP}}{\text{TP}+\text{FP}},\ R = \frac{\text{TP}}{\text{TP}+\text{FN}}$ | Diagnoses false alarm vs. churn detection trade-offs. |
| **Secondary** | **ROC-AUC & PR-AUC** | $D_{\text{adapt,dev\_val}}$ / $D_{\text{target,eval}}$ | Trapezoidal integral of ROC and PR curves | Evaluates continuous probability calibration. |
| **Slice Diagnostic** | **Worst-Slice F1** | Subset: $\text{tenure} > 29.0$ | $F1(Y_{\text{slice}}, \hat{Y}_{\text{slice}})$ | Measures direct recovery on the drifted concept slice. |
| **Safety Retention** | **Source F1 Retention** | $D_{\text{source,eval}}$ ($N=493$) | $F1_{\text{source}}(S_i) - F1_{\text{source}}(A0)$ | Guards against catastrophic forgetting on pre-drift domain. |
| **Generator Quality**| **Validation Yield Rate**| Synthetic Candidate Pool | $\frac{N_{\text{valid}}}{N_{\text{drawn}}} \times 100\%$ | Measures efficiency and structural fidelity of generator. |
| **Telemetry** | **Runtime & RSS Peak** | System OS Process | Elapsed CPU seconds and peak resident memory | Monitors laptop resource bounds. |

### 8.2 Statistical Testing & Multiple Testing Corrections
* **Paired Bootstrap:** $B=2,000$ paired bootstrap iterations on aligned predictions.
* **Paired Permutation Tests:** $R=2,000$ sign-flip permutations of paired residuals.
* **Multiplicity Adjustments:** For the prespecified comparisons within each budget, both Bonferroni and Benjamini-Hochberg False Discovery Rate (FDR) adjustments will be reported.
* **Deterministic Analysis Seed:** Set strictly to `seed=42`.

---

## 9. Computational Feasibility and CPU Execution Stages

The execution environment is a Windows laptop with CPU-only compute and bounded available memory. We organize the benchmark into four sequential, fail-fast phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FOUR-PHASE SEQUENTIAL BENCHMARK                       │
├─────────┬───────────────────────────────────┬───────────────────────────────┤
│ Phase   │ Objective                         │ Resource Budget & Stop Rule   │
├─────────┼───────────────────────────────────┼───────────────────────────────┤
│ Phase 1 │ Authentic Controls (B0–B5)        │ Runtime: < 2 min. RAM: < 300MB│
│         │ Run B0, B1, B2, B3, B4, B5 on     │ Stop if any split overlap or  │
│         │ dev_val for budgets 50, 100, 200  │ preprocessor leak is detected │
├─────────┼───────────────────────────────────┼───────────────────────────────┤
│ Phase 2 │ Generator Smoke Test ($B_{200}, K=50$) │ Runtime: < 3 min. RAM: < 1GB │
│         │ Test TVAE, CTGAN, TabDDPM on CPU  │ Stop if loss is NaN or valid  │
│         │ Verify Tier 1-3 validation yield  │ yield rate < 10%              │
├─────────┼───────────────────────────────────┼───────────────────────────────┤
│ Phase 3 │ Focused Quantity Study (Best Gen) │ Runtime: < 8 min. RAM: < 2GB  │
│         │ Test K in {50, 100, 200, 500} on  │ Select K* maximizing dev_val  │
│         │ B200 across Arms S1, S2, S3, S4   │ F1 over strongest control     │
├─────────┼───────────────────────────────────┼───────────────────────────────┤
│ Phase 4 │ Multi-Budget Benchmark & Locked   │ Runtime: < 10 min. RAM: < 2GB │
│         │ Comparative Evaluation on D_target│ Single-pass locked evaluation │
│         │ Evaluate B50, B100, B200 at K*    │ Produce final audit artifacts │
└─────────┴───────────────────────────────────┴───────────────────────────────┘
```

### 9.1 Measurable Stopping Conditions
1. **Memory Ceiling:** If process memory exceeds 4.0 GB RSS at any point $\implies$ **ABORT IMMEDIATELY**.
2. **Generator Divergence:** If training loss produces `NaN` or `Inf` $\implies$ **FAIL GENERATOR ARM**.
3. **Mode Collapse:** If duplicate synthetic row rate exceeds 90% $\implies$ **FAIL GENERATOR ARM**.
4. **Validation Yield Failure:** If valid candidate yield is $< 10\%$ after 5,000 sampling draws $\implies$ **ABSTAIN FROM RETRAINING**.
5. **Partition Leakage:** If any feature group or index overlap is detected between training data and validation/evaluation data $\implies$ **HARD SYSTEM HALT**.
6. **Provenance Violation:** If unverified feature mutations are labeled as ground truth $\implies$ **HARD SYSTEM HALT**.

---

## 10. Explicit Decision Gates for Implementation

To govern subsequent experimental batches, we establish three unambiguous gating decisions:

$$\begin{aligned}
&\mathbf{GO:} \quad \text{Authorized to implement Phase 1 (Authentic Controls B0–B5) and Phase 2 (Generator Smoke Test).} \\
&\mathbf{CONDITIONAL\ GO:} \quad \text{Proceeding to Phase 3 (Synthetic Repair Study) is conditional on:}\\
&\qquad \text{1. Demonstrating TVAE or CTGAN can train on } B_{200} \text{ within } 3 \text{ minutes on CPU.}\\
&\qquad \text{2. Achieving } \ge 50\% \text{ valid candidate yield under Tier 1–3 screening.}\\
&\qquad \text{3. Identifying a non-trivial authentic control benchmark (max of B2, B3, B4, B5).}\\
&\mathbf{BLOCKED:} \quad \text{Model training on } D_{\text{target,eval}} \text{ during method development is strictly blocked.}
\end{aligned}$$

---

## 11. Traceability and Artifact Registry

* **Protocol Specification:** [`component3/results/experimental_models/batch05h_benchmark_protocol.json`](file:///d:/SLIIT_Research/J26-DS-361-Intelligent-MLOps-Platform/component3/results/experimental_models/batch05h_benchmark_protocol.json)
* **Historical Baseline Models:** `component3/models/baselines/` (9 frozen models preserved)
* **Experimental Reference Models:** `component3/models/experimental/` ($A0, A1\text{-}B50, A1\text{-}B100, A1\text{-}B200$ preserved)
* **Split Manifests:** `component3/results/adaptation_splits/` (Verified 0 overlap preserved)
* **Verification Tests:** `component3/tests/test_batch05h_protocol.py` (Validates benchmark protocol and isolation rules)
