# Sprint 04 — Task 08: TabDDPM Targeted Data Generation Testing

**Research Project:** Intelligent MLOps Platform — Monitoring, Drift Detection, Generative Repair, and Agentic Retraining\
**Research Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Sprint:** 04 — Synthetic Data Generator Implementation\
**Task:** 08 — Test TabDDPM Targeted Data Generation\
**Planned Sprint Period:** October 1–5, 2026\
**Actual Experiment Date:** October 17, 2026\
**Status:** Implementation, experiments, and artifact validation completed; Git delivery pending

---

## 1. Objective

The objective of this task was to systematically test the TabDDPM (Tabular Denoising Diffusion Probabilistic Model) synthetic data generator under controlled drift conditions and evaluate whether generated records satisfy diagnosed affected-slice requirements.

The experiments focused on:

1. Validating end-to-end integration with Component 3's data preparation and slice evaluation architecture.
2. Generating synthetic tabular data targeting diagnosed drift slices across continuous, categorical, and compound conditions.
3. Preserving the expected 12-feature UCI Adult schema without target label leakage.
4. Measuring target-condition coverage across seven controlled drift scenarios (S01–S07).
5. Assessing stability across multiple random seeds (42, 123, 456).
6. Capturing diagnostic numerical range violations and boundary behavior.
7. Ensuring regression safety across all Component 3 modules.

These experiments provide empirical evidence for subsequent multi-generator comparison, fidelity benchmarking, and repair-planning optimization.

## 2. Experimental Architecture

The TabDDPM experiment runner was implemented in:

`component3/src/experiments/run_tabddpm_targeted.py`

The runner adheres strictly to the experimental architecture established for CTGAN and TVAE, ensuring comparability across all three generators.

### Processing Workflow

1. Load the central Component 3 configuration from `component3/configs/c3_config.json`.
2. Load the selected controlled drift scenario dataset from `component3/data/drifted/`.
3. Retrieve the diagnosed slice condition for the scenario.
4. Extract affected records using `extract_affected_slice()`.
5. Construct an exploratory-only readiness decision (`ReadinessStatus.REVIEW`, `generation_ready=True`, `supervised_retraining_ready=False`).
6. Prepare training data and explicit categorical metadata using `prepare_generator_data()`.
7. Initialize and train `TabDDPMGenerator` (epochs=1, diffusion timesteps=10, batch_size=128).
8. Generate 100 synthetic feature records via reverse diffusion sampling.
9. Evaluate target-condition coverage using `evaluate_target_coverage()`.
10. Run diagnostic numerical-range validation using `validate_numerical_ranges()`.
11. Record training duration, sampling latency, missing values, and governance metadata.
12. Persist the structured experiment summary as a JSON artifact in `component3/results/tabddpm_targeted/`.

### Reused Modules

- `component3/src/core/config_loader.py`
- `component3/src/core/data_loader.py`
- `component3/src/contracts/safe_slice_evaluator.py`
- `component3/src/generators/data_preparer.py`
- `component3/src/generators/target_coverage.py`
- `component3/src/generators/tabddpm_generator.py`
- `component3/src/readiness/readiness_models.py`

All slice rules were identically reused from `SCENARIO_RULES` to maintain parity with CTGAN and TVAE benchmarks.

## 3. Dataset and Experimental Configuration

### Dataset

UCI Adult benchmark dataset, utilizing the controlled drift datasets generated during Sprint 02.

### Feature Structure

The generator operated on 12 input features (target column strictly excluded):

**Continuous features (5):**
- `age`
- `education_num`
- `capital_gain`
- `capital_loss`
- `hours_per_week`

**Categorical features (7):**
- `workclass`
- `marital_status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native_country`

Explicit categorical metadata was supplied to `TabDDPMGenerator` to ensure integer-encoded categorical features were modeled via multinomial diffusion rather than Gaussian continuous diffusion.

### Experimental Parameters

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Generator** | TabDDPM | Diffusion probabilistic architecture |
| **Training Epochs** | 1 | Controlled benchmark across all three generators |
| **Diffusion Timesteps** | 10 | Fast exploratory reverse diffusion sampling |
| **Batch Size** | 128 | Mini-batch optimization size |
| **Synthetic Records per Run** | 100 | Standard sample size matching CTGAN & TVAE |
| **Random Seeds** | 42, 123, 456 | Multi-seed evaluation for reproducibility |
| **Drift Scenarios** | S01–S07 | Full spectrum of drift conditions |
| **Total Experiments** | 21 | 7 scenarios × 3 seeds |
| **Total Generated Records** | 2,100 | 21 experiments × 100 samples |
| **Generation Mode** | `unsupervised_exploratory` | Exploratory diagnostic generation |
| **Readiness Source** | `synthetic_experiment_decision` | Controlled experiment harness |
| **Validation Status** | `diagnostic_only` | Non-production diagnostic validation |

## 4. Controlled Drift Scenarios

Seven diagnosed affected-slice conditions were evaluated:

| Scenario | Drift Type | Slice Condition | Training Records |
| :---: | :--- | :--- | :---: |
| **S01** | Continuous numerical drift | `age > 0.69` | 2,159 |
| **S02** | Continuous numerical drift | `capital_loss > -0.20` | 6,000 |
| **S03** | Categorical drift | `marital_status == 2` | 1,723 |
| **S04** | Categorical drift | `race == 2` | 360 |
| **S05** | Compound continuous drift | `age > 0.69 and education_num < -0.03` | 342 |
| **S06** | Compound categorical drift | `marital_status == 2 and relationship == 0` | 1,674 |
| **S07** | Mixed compound drift | `age > 0.69 and marital_status == 2` | 1,550 |

## 5. Experiment Runner Implementation

The automated experiment runner was created in:

`component3/src/experiments/run_tabddpm_targeted.py`

### Key Implementation Details

- Command-line interface with `--scenario`, `--epochs`, `--samples`, `--seed`, `--timesteps`, and `--batch-size`.
- Explicit discrete column metadata forwarding to `TabDDPMGenerator`.
- Diagnostic tracking using `generator.validate_numerical_ranges(synthetic_df)`.
- Separate measurement of `training_seconds` and `sampling_seconds`.
- Artifact directory: `component3/results/tabddpm_targeted/`.
- Artifact naming: `{scenario}_tabddpm_e{epochs}_t{timesteps}_b{batch_size}_n{samples}_seed{seed}.json`.
- Zero modification to existing CTGAN or TVAE experiment runners.

## 6. Initial Automated Experiment Verification

### Pilot Experiment 1 — Scenario S04 (Categorical Drift)

- Slice rule: `race == 2`
- Training records: 360
- Training epochs: 1 (timesteps: 10, batch_size: 128)
- Random seed: 42
- Generated records: 100
- Matching records: 100 (Target coverage: 100.0%)
- Observed range violations: 0
- Missing values: 0
- Training time: 5.5570 s | Sampling time: 0.0875 s
- Artifact: `S04_tabddpm_e1_t10_b128_n100_seed42.json`

### Pilot Experiment 2 — Scenario S05 (Compound Continuous Drift)

- Slice rule: `age > 0.69 and education_num < -0.03`
- Training records: 342
- Training epochs: 1 (timesteps: 10, batch_size: 128)
- Random seed: 42
- Generated records: 100
- Matching records: 100 (Target coverage: 100.0%)
- Observed range violations: 0
- Missing values: 0
- Training time: 4.6170 s | Sampling time: 0.0303 s
- Artifact: `S05_tabddpm_e1_t10_b128_n100_seed42.json`

Both pilot experiments completed cleanly, validating runner execution across categorical and multi-feature continuous slices.

## 7. Multi-Scenario and Multi-Seed Experiment Design

Following pilot verification, the experiment suite was expanded to the full matrix of 7 scenarios and 3 random seeds:

- **Scenarios:** S01, S02, S03, S04, S05, S06, S07
- **Random Seeds:** 42, 123, 456
- **Runs:** 7 × 3 = 21 independent experiments
- **Sample Count:** 100 synthetic records per experiment (2,100 total)

All 21 experiments were executed sequentially with automated fail-stop verification. All 21 runs completed with exit code 0.

## 8. TabDDPM Experimental Results

### Target Coverage by Scenario and Random Seed

| Scenario | Seed 42 | Seed 123 | Seed 456 | Mean Coverage | Training Time (Mean) | Sampling Time (Mean) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01** | 100% | 100% | 100% | **100.00%** | 5.15 s | 0.1117 s |
| **S02** | 100% | 100% | 100% | **100.00%** | 5.55 s | 0.0955 s |
| **S03** | 100% | 100% | 100% | **100.00%** | 4.64 s | 0.1098 s |
| **S04** | 100% | 100% | 100% | **100.00%** | 4.30 s | 0.0905 s |
| **S05** | 100% | 100% | 100% | **100.00%** | 4.27 s | 0.0857 s |
| **S06** | 100% | 100% | 100% | **100.00%** | 3.34 s | 0.0605 s |
| **S07** | 100% | 100% | 100% | **100.00%** | 1.23 s | 0.0428 s |

### Aggregate Results

| Metric | Result |
| :--- | :---: |
| **Total Experiments** | 21 |
| **Total Generated Records** | 2,100 |
| **Records Matching Target Conditions** | 2,100 |
| **Records Outside Target Conditions** | 0 |
| **Aggregate Target Coverage** | **100.00%** |
| **Total Out-of-Range Violations** | 0 |
| **Total Missing Values** | 0 |
| **Saved JSON Artifacts** | 21 |

$$\text{Aggregate Target Coverage} = \frac{2,100}{2,100} \times 100 = 100.00\%$$

## 9. Preliminary Comparison with CTGAN and TVAE

All three generators were evaluated under identical experimental conditions (1 epoch, 100 samples per run, 21 runs across S01–S07 and seeds 42, 123, 456):

### Per-Scenario Target Coverage Comparison

| Scenario | CTGAN Mean Coverage | TVAE Mean Coverage | TabDDPM Mean Coverage |
| :---: | :---: | :---: | :---: |
| **S01** | 98.33% | 100.00% | **100.00%** |
| **S02** | 100.00% | 100.00% | **100.00%** |
| **S03** | 100.00% | 100.00% | **100.00%** |
| **S04** | 100.00% | 100.00% | **100.00%** |
| **S05** | 93.67% | 99.67% | **100.00%** |
| **S06** | 100.00% | 100.00% | **100.00%** |
| **S07** | 96.67% | 100.00% | **100.00%** |

### Aggregate Cross-Generator Comparison

| Metric | CTGAN | TVAE | TabDDPM |
| :--- | :---: | :---: | :---: |
| **Total Experiments** | 21 | 21 | 21 |
| **Total Generated Records** | 2,100 | 2,100 | 2,100 |
| **Matching Records** | 2,066 | 2,099 | **2,100** |
| **Non-Matching Records** | 34 | 1 | **0** |
| **Aggregate Target Coverage** | 98.38% | 99.95% | **100.00%** |
| **Total Missing Values** | 0 | 0 | 0 |
| **Total Range Violations** | N/A | N/A | **0** |
| **Average Training Time** | ~15–20 s | ~2–19 s | **~1.2–5.6 s** |
| **Average Sampling Time** | ~0.5–1.0 s | ~0.1–0.3 s | **~0.03–0.14 s** |

**Important Scientific Caution:** TabDDPM achieved 100.00% target coverage, outperforming CTGAN (+1.62 pp) and TVAE (+0.05 pp). However, this descriptive metric does not establish statistical superiority or superior data realism. As documented in Section 11, TabDDPM exhibits numerical boundary concentration in single-epoch configurations.

## 10. Automated Experiment Artifact Validation

All 21 saved TabDDPM JSON artifacts were validated using an automated verification script enforcing 11 integrity rules:

1. Exact file count: 21 JSON artifacts.
2. Complete scenario-seed coverage (S01–S07 × {42, 123, 456}) with zero duplicates.
3. Filename conformity: `{scenario}_tabddpm_e1_t10_b128_n100_seed{seed}.json`.
4. Slice rule alignment with `SCENARIO_RULES`.
5. Hyperparameter consistency (`epochs=1`, `num_timesteps=10`, `batch_size=128`).
6. Feature schema preservation: 12 expected columns, target excluded.
7. Governance tags: `generation_mode="unsupervised_exploratory"`, `readiness_source="synthetic_experiment_decision"`, `validation_status="diagnostic_only"`.
8. Coverage arithmetic: `matching_rows + non_matching_rows == 100` and `coverage == matching_rows / 100`.
9. Range diagnostics consistency: 5 numerical columns evaluated, `passed == (total_invalid_values == 0)`.
10. Missing-value count: 0 across all files.
11. Overall integrity check: 0 validation errors.

**Artifact Validation Result: PASS (0 errors)**

## 11. Numerical Boundary Concentration

A key scientific discovery during TabDDPM testing was the presence of **numerical boundary concentration**:

- Generated continuous values frequently concentrate near the minimum or maximum observed values of the training slice.
- For example, in S05 (`age > 0.69 and education_num < -0.03`), synthetic `age` values clustered at `0.694082` and `4.350851`, while `education_num` values clustered at `-0.119532` and `-2.933839`.
- **Root Cause Analysis:** This behavior arises from the interaction of `QuantileTransformer` (which maps input features into standard normal quantiles), reverse diffusion sampling with 10 timesteps, and single-epoch training. Because the training slice only contains points satisfying the slice condition, the quantiles map to bounded regions, and diffusion modes settle at extreme quantiles.
- **Implication:** Zero range violations and 100% target coverage do **not** imply high diversity or statistical fidelity. Multi-epoch tuning and fidelity metrics (Wasserstein distance, Jensen-Shannon divergence) will be critical in Sprint 05.

## 12. Full Component 3 Regression Testing

After completing the 21 batch experiments, full Component 3 regression testing was executed:

**Command:**
```powershell
.\.venv\Scripts\python.exe -m pytest component3/tests/ -q
```

**Results:**
- Tests passed: **321 / 321**
- Tests failed: **0**
- Non-failing warnings: 4 (known PyTorch TorchScript deprecation and serialized model compatibility notices)
- Execution time: 26.31 s

The TabDDPM experiment runner and results introduced zero regression defects to existing Component 3 functionality.

## 13. Research Findings

### Finding 1 — TabDDPM Effectively Isolates Diagnosed Drift Slices
TabDDPM successfully trained on affected data slices ranging from 342 records (S05) to 6,000 records (S02), producing 100 records per run with 100% slice rule compliance.

### Finding 2 — Explicit Discrete Metadata Enables Mixed Diffusion
Passing `discrete_columns` metadata resolved the integer-encoded categorical challenge in UCI Adult, allowing Gaussian diffusion for continuous features and multinomial diffusion for categorical features.

### Finding 3 — Fast Computational Throughput
TabDDPM demonstrated rapid training (~1.2–5.6 s) and very fast reverse diffusion sampling (~0.03–0.14 s for 100 samples) under 10 timesteps, outperforming CTGAN and TVAE in wall-clock execution time.

### Finding 4 — High Target Coverage Does Not Equal High Fidelity
100.00% target coverage indicates that generated records lie within the diagnosed slice boundary, but boundary concentration demonstrates that distributional shape within the slice requires multi-epoch training and separate fidelity validation.

### Finding 5 — Multi-Seed Stability
Zero variance in target coverage was observed across seeds 42, 123, and 456 (all 100%), confirming algorithmic reproducibility across seeds.

## 14. Research Limitations

1. **Short Training Horizon:** 1 epoch was utilized to maintain benchmark consistency with CTGAN and TVAE; higher epochs are required for distributional convergence.
2. **Reduced Timesteps:** 10 diffusion timesteps provided fast sampling but may contribute to mode concentration compared to 100 or 1,000 timesteps.
3. **No Downstream Retraining:** Exploratory records were not evaluated for classifier utility or model repair.
4. **No Differential Privacy:** Diffusion models can memorize training samples in small slices; privacy risk was not measured in this task.
5. **No Saved Raw Data:** Only aggregate metrics were saved in the JSON reports; record-level distributional analysis requires persisting sample arrays.
6. **Diagnostic Scope:** Target coverage measures condition satisfaction only, not overall multidimensional correlation structure.

## 15. Implementation and Evidence Artifacts

### Main Implementation
- `component3/src/experiments/run_tabddpm_targeted.py`

### Generator Core
- `component3/src/generators/tabddpm_generator.py`

### Vendored Diffusion Modules
- `component3/vendor/tab_ddpm/gaussian_multinomial_diffsuion.py`
- `component3/vendor/tab_ddpm/modules.py`
- `component3/vendor/tab_ddpm/utils.py`

### Test Suite
- `component3/tests/test_tabddpm_metadata.py` (24 unit tests)
- `component3/tests/manual_tabddpm_smoke.py` (real training smoke test)
- `component3/tests/manual_tabddpm_adult_smoke.py` (UCI Adult integration test)

### Experiment Results
- `component3/results/tabddpm_targeted/` (21 verified JSON artifacts)

### Technical Report
- `component3/docs/sprint04_tabddpm_targeted_generation_testing.md`

## 16. Conclusion

Sprint 04 Task 08 successfully implemented and validated the automated TabDDPM targeted data generation experiment workflow.

Across 21 controlled experiments spanning seven drift scenarios and three random seeds:
- 2,100 synthetic records were generated.
- 2,100 records satisfied diagnosed slice conditions (**100.00% target coverage**).
- Zero missing values and zero out-of-range numerical violations occurred.
- All 21 JSON experiment artifacts passed automated consistency verification.
- Full regression suite passed with **321/321 tests**.

With the completion of CTGAN (Task 04), TVAE (Task 06), and TabDDPM (Task 08), Component 3 now possesses three fully operational synthetic data generator experiment runners with empirical baseline evidence across all controlled drift scenarios.

**Sprint 04 Task 08 is ready for Git commit and push to branch `c3-mathusan`.**
