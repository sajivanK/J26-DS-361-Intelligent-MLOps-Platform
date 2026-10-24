# Sprint 04 — Task 06: TVAE Targeted Data Generation Testing

**Research Project:** Intelligent MLOps Platform — Monitoring, Drift Detection, Generative Repair, and Agentic Retraining\
**Research Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Sprint:** 04 — Synthetic Data Generator Implementation\
**Task:** 06 — Test TVAE Targeted Data Generation\
**Planned Sprint Period:** October 1–5, 2026\
**Actual Experiment Date:** October 16, 2026\
**Status:** Implementation and experiments verified; technical documentation and Git finalization pending

---

## 1. Objective

The objective of this task was to systematically test the TVAE synthetic data generator under controlled drift conditions and determine whether generated records satisfy the diagnosed affected-slice requirements.

The experiments focused on:

1. Correct integration with Component 3's existing data preparation architecture.
2. Synthetic generation from affected data slices.
3. Preservation of the expected feature schema.
4. Target-condition coverage across different drift scenarios.
5. Stability of observed coverage across multiple random seeds.
6. Collection of structured experiment evidence.
7. Regression safety of the existing Component 3 implementation.

These experiments provide preliminary evidence for future generator comparison and repair-planning research.

## 2. Experimental Architecture

The TVAE experiment runner was implemented in:

`component3/src/experiments/run_tvae_targeted.py`

The runner follows the established CTGAN experimental architecture.

### Processing Workflow

1. Load the central Component 3 configuration.
2. Load the selected controlled drift scenario dataset.
3. Retrieve the corresponding affected-slice condition.
4. Extract the affected records using the safe slice evaluator.
5. Construct an exploratory-only readiness decision.
6. Prepare the generator training dataset.
7. Extract explicit categorical metadata.
8. Initialize and train TVAE.
9. Generate synthetic feature records.
10. Evaluate target-condition coverage.
11. Record training duration and data integrity metrics.
12. Save the experiment results as JSON.

### Reused Components

- `component3/src/core/config_loader.py`
- `component3/src/core/data_loader.py`
- `component3/src/contracts/safe_slice_evaluator.py`
- `component3/src/generators/data_preparer.py`
- `component3/src/generators/target_coverage.py`
- `component3/src/generators/tvae_generator.py`
- `component3/src/readiness/readiness_models.py`

Scenario rules were reused from the existing CTGAN experiment runner.

This approach maintained consistency between the two generator experiments without modifying the CTGAN implementation.

## 3. Dataset and Experimental Configuration

### Dataset

UCI Adult, using the existing controlled drift datasets developed during Sprint 02.

### Feature Structure

The TVAE generator used 12 input features:

**Continuous features:**

- `age`
- `education_num`
- `capital_gain`
- `capital_loss`
- `hours_per_week`

**Categorical features:**

- `workclass`
- `marital_status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native_country`

Categorical feature information was explicitly passed to TVAE instead of relying only on pandas data types.

The target label was excluded from generator training and synthetic output because the experiments were exploratory and did not establish trusted supervised-label provenance.

### Experimental Parameters

| Parameter                        | Value                           |
| -------------------------------- | ------------------------------- |
| Generator                        | TVAE                            |
| Training epochs                  | 1                               |
| Synthetic records per experiment | 100                             |
| Random seeds                     | 42, 123, 456                    |
| Drift scenarios                  | S01–S07                         |
| Total experiments                | 21                              |
| Total generated records          | 2,100                           |
| Generation mode                  | `unsupervised_exploratory`      |
| Readiness provenance             | `synthetic_experiment_decision` |
| Validation status                | `diagnostic_only`               |

One training epoch was selected for preliminary diagnostic testing, not for final model-quality optimization.

## 4. Controlled Drift Scenarios

Seven diagnosed affected-slice conditions were evaluated.

| Scenario | Slice Condition                             | Training Records |
| -------- | ------------------------------------------- | ---------------- |
| S01      | `age > 0.69`                                | 2,159            |
| S02      | `capital_loss > -0.20`                      | 6,000            |
| S03      | `marital_status == 2`                       | 1,723            |
| S04      | `race == 2`                                 | 360              |
| S05      | `age > 0.69 and education_num < -0.03`      | 342              |
| S06      | `marital_status == 2 and relationship == 0` | 1,674            |
| S07      | `age > 0.69 and marital_status == 2`        | 1,550            |

These slice conditions were defined by the controlled experiment framework. They were not independently discovered by a production Component 2 diagnosis engine.

## 5. Experiment Runner Implementation

A dedicated TVAE experiment runner was created.

**Implementation file:**

`component3/src/experiments/run_tvae_targeted.py`

### Important Implementation Properties

- Supports scenario, epoch, sample-count, and random-seed arguments.
- Uses the existing central configuration.
- Uses the safe affected-slice extraction mechanism.
- Reuses the existing generator data preparation module.
- Passes categorical metadata explicitly to TVAE.
- Excludes the target label from exploratory generation.
- Records target coverage and execution metadata.
- Saves results separately from CTGAN experiments.
- Preserves the existing CTGAN runner without modifications.

### Output Directory

`component3/results/tvae_targeted/`

### Artifact Naming Convention

`{scenario}_tvae_e{epochs}_n{samples}_seed{seed}.json`

Example:

`S05_tvae_e1_n100_seed42.json`

Each JSON artifact records experiment settings, affected-slice condition, training row count, generated row count, feature names, missing-value count, target coverage, training duration, and provenance metadata.

Only summary metadata was persisted. The generated synthetic records themselves were not saved by this runner.

## 6. Initial Automated Experiment Verification

The first automated TVAE experiment was conducted on Scenario S04.

### Scenario S04

- Affected-slice rule: `race == 2`
- Training records: 360
- Training epochs: 1
- Random seed: 42
- Generated records: 100
- Matching records: 100
- Non-matching records: 0
- Target coverage: 100%
- Missing values: 0
- Training duration: 19.4865 seconds

The generated output preserved all 12 expected feature columns in the correct order.

The JSON experiment artifact was successfully saved.

### Scenario S05

The second automated experiment tested a compound numerical condition:

`age > 0.69 and education_num < -0.03`

Results:

- Training records: 342
- Training epochs: 1
- Random seed: 42
- Generated records: 100
- Matching records: 100
- Non-matching records: 0
- Target coverage: 100%
- Missing values: 0
- Training duration: 1.7834 seconds

Both initial experiments completed successfully.

These results demonstrated that the runner could process categorical and compound numerical slice conditions.

## 7. Multi-Scenario and Multi-Seed Experiment Design

After the initial tests, the experiment was expanded to seven drift scenarios and three random seeds.

The seed values were:

- 42
- 123
- 456

Each scenario was executed independently for each seed.

Therefore:

7 scenarios × 3 seeds = 21 experiments.

Each experiment generated 100 synthetic records.

Total synthetic records generated:

21 × 100 = 2,100 records.

The experiments were executed sequentially using Windows PowerShell, with the process configured to stop if an experiment failed.

All 21 experiment executions completed successfully.

## 8. TVAE Experimental Results

### Target Coverage by Scenario and Random Seed

| Scenario | Seed 42 | Seed 123 | Seed 456 | Mean Coverage |
| -------- | ------- | -------- | -------- | ------------- |
| S01      | 100%    | 100%     | 100%     | 100.00%       |
| S02      | 100%    | 100%     | 100%     | 100.00%       |
| S03      | 100%    | 100%     | 100%     | 100.00%       |
| S04      | 100%    | 100%     | 100%     | 100.00%       |
| S05      | 100%    | 99%      | 100%     | 99.67%        |
| S06      | 100%    | 100%     | 100%     | 100.00%       |
| S07      | 100%    | 100%     | 100%     | 100.00%       |

### Aggregate Results

| Metric                             | Result |
| ---------------------------------- | ------ |
| Total experiments                  | 21     |
| Total generated records            | 2,100  |
| Records matching target conditions | 2,099  |
| Records outside target conditions  | 1      |
| Aggregate target coverage          | 99.95% |
| Total missing values               | 0      |
| Saved JSON artifacts               | 21     |

The aggregate coverage calculation was:

**Aggregate Target Coverage = 2,099 / 2,100 × 100 = 99.95%**

The only observed non-matching record occurred in Scenario S05 under random seed 123.

This represents a failure to satisfy the complete affected-slice condition, rather than proof that the record was invalid in every other respect.

## 9. Preliminary Comparison with CTGAN

The previous CTGAN experiment used the same seven scenarios, three seeds, one training epoch, and 100 generated records per experiment.

### Per-Scenario Comparison

| Scenario | CTGAN Mean Coverage | TVAE Mean Coverage |
| -------- | ------------------- | ------------------ |
| S01      | 98.33%              | 100.00%            |
| S02      | 100.00%             | 100.00%            |
| S03      | 100.00%             | 100.00%            |
| S04      | 100.00%             | 100.00%            |
| S05      | 93.67%              | 99.67%             |
| S06      | 100.00%             | 100.00%            |
| S07      | 96.67%              | 100.00%            |

### Aggregate Comparison

| Metric                    | CTGAN  | TVAE   |
| ------------------------- | ------ | ------ |
| Experiments               | 21     | 21     |
| Generated records         | 2,100  | 2,100  |
| Matching records          | 2,066  | 2,099  |
| Non-matching records      | 34     | 1      |
| Aggregate target coverage | 98.38% | 99.95% |
| Missing values            | 0      | 0      |

TVAE achieved approximately **1.57 percentage points higher aggregate target coverage** than CTGAN in this experimental configuration.

The difference is descriptive, not a statistically established superiority claim.

Both generators were tested with only one training epoch, which may not provide equivalent convergence or model quality.

Further evaluation is required before recommending either generator for production repair.

## 10. Automated Experiment Artifact Validation

All 21 saved TVAE JSON artifacts were checked using an independent validation script.

The script verified:

1. Total number of JSON artifacts.
2. Unique scenario and seed combinations.
3. Correct filename conventions.
4. Correct affected-slice rules.
5. Correct training epoch values.
6. Correct exploratory generation mode.
7. Correct readiness provenance.
8. Correct diagnostic-only validation status.
9. Preservation of the 12-feature schema.
10. Exclusion of the target label.
11. Positive training row counts.
12. Non-negative training durations.
13. Generated row count consistency.
14. Matching and non-matching count consistency.
15. Coverage calculation consistency.
16. Missing-value count validity.
17. Presence of all expected scenario-seed combinations.

### Validation Results

- JSON artifacts found: 21
- Expected experiments: 21
- Unique experiment pairs: 21
- Total generated records: 2,100
- Total matching records: 2,099
- Total non-matching records: 1
- Total missing values: 0
- Aggregate coverage: 99.95%
- Validation errors: 0

**Artifact Validation Result: PASS**

This validation confirmed the consistency of saved experiment summaries. It did not independently verify the underlying generated records because the runner did not persist raw synthetic datasets.

## 11. Full Component 3 Regression Testing

After implementing and executing the TVAE experiment runner, the complete Component 3 regression suite was executed.

### Command

`.\.venv\Scripts\python.exe -m pytest component3/tests/ -q`

### Results

- Tests passed: 297
- Tests failed: 0
- Non-failing warnings: 4
- Execution time: 75.20 seconds

The warnings were associated with existing PyTorch deprecation notices and serialized model dependency-version compatibility.

No regression failures were observed in the tested Component 3 functionality.

**Full Regression Result: PASS**

## 12. Research Findings

### Finding 1 — TVAE Supports Targeted Slice Generation

TVAE successfully generated synthetic records from the diagnosed affected subsets in all seven controlled scenarios.

### Finding 2 — High Target Coverage Was Observed

TVAE achieved 99.95% aggregate target-condition coverage across the 21 experiments.

### Finding 3 — Compound Numerical Conditions Require Validation

One generated record in S05 failed the combined numerical slice condition.

This demonstrates the importance of explicitly measuring whether synthetic records remain within the diagnosed affected region.

### Finding 4 — Explicit Categorical Metadata Was Compatible with the Experiments

The TVAE generator accepted categorical feature metadata and preserved the expected output schema throughout the tested runs.

### Finding 5 — Generator Selection Cannot Depend on Coverage Alone

TVAE achieved higher observed target coverage than CTGAN, but this metric does not measure the complete quality of synthetic data.

A useful repair generator must also satisfy fidelity, utility, privacy, and computational requirements.

## 13. Research Limitations

The following limitations remain:

1. **Single training epoch:** Generator quality may change substantially with longer training.
2. **Limited random seeds:** Only three seeds were tested.
3. **Limited generated sample size:** Each experiment generated only 100 records.
4. **One dataset:** Experiments were conducted on UCI Adult controlled drift scenarios.
5. **Simulated diagnosis:** Slice rules came from the controlled experiment framework, not the real Component 2 engine.
6. **Synthetic readiness decision:** The experiment used exploratory-only readiness metadata rather than an independently verified readiness authorization.
7. **No fidelity evaluation:** Distributional similarity and feature relationships were not assessed.
8. **No privacy evaluation:** Memorization and disclosure risks were not assessed.
9. **No downstream utility evaluation:** Synthetic data was not evaluated through supervised model retraining.
10. **No comprehensive domain validation:** Zero missing values and correct feature columns do not establish that all generated feature values are valid.
11. **No saved raw synthetic records:** Independent record-level verification and exact reproduction from the saved artifacts are limited.
12. **No statistical significance analysis:** The observed CTGAN–TVAE difference has not been established as statistically significant.
13. **No production integration:** The experiment runner does not constitute the complete Component 3 repair pipeline.

The results should therefore be interpreted as preliminary diagnostic evidence.

## 14. Contribution to Component 3

This task provides an important experimental foundation for the proposed intelligent synthetic data repair engine.

The collected evidence can support later development of:

- Generator performance comparison.
- Target coverage validation.
- Synthetic repair candidate screening.
- Generator selection strategies.
- Minimum effective repair quantity experiments.
- Fidelity, utility, and privacy evaluation.
- Intelligent repair prescription learning.

The findings help establish that generator outputs must be evaluated against the affected-slice requirements rather than accepted solely because synthetic generation completed successfully.

## 15. Implementation and Evidence Artifacts

### Main Implementation

`component3/src/experiments/run_tvae_targeted.py`

### Generator

`component3/src/generators/tvae_generator.py`

### Shared Data Preparation

`component3/src/generators/data_preparer.py`

### Shared Coverage Evaluator

`component3/src/generators/target_coverage.py`

### Scenario Definitions

`component3/src/experiments/run_ctgan_targeted.py`

### TVAE Results

`component3/results/tvae_targeted/`

Contains 21 experiment JSON artifacts.

### CTGAN Comparison Results

`component3/results/ctgan_targeted/`

Contains 21 previously generated CTGAN experiment JSON artifacts.

### Technical Report

`component3/docs/sprint04_tvae_targeted_generation_testing.md`

## 16. Conclusion

Sprint 04 Task 06 successfully implemented and exercised a controlled TVAE targeted-generation experiment workflow.

The generator was evaluated across seven affected-slice scenarios and three random seeds, producing 2,100 synthetic records.

Of these records, 2,099 satisfied their respective target conditions, yielding an aggregate coverage of 99.95%.

All 21 experiment summary artifacts passed automated consistency validation, and all 297 Component 3 regression tests passed.

The results demonstrate successful preliminary TVAE targeted-generation testing and provide structured evidence for later generator comparison and repair validation.

However, target coverage alone is insufficient to establish synthetic-data quality, privacy safety, or downstream repair effectiveness.

**Task 06 completion is pending final documentation verification and Git review.**
