# Sprint 02 — Final Technical Report

## Intelligent MLOps Platform — Component 3

**Project ID:** J26-DS-361\
**Component:** Adaptive Intelligent Synthetic Data Repair Engine\
**Sprint:** 02 — Controlled Drift and Diagnosis Simulation\
**Original Planned Period:** 21–25 September 2026\
**Actual Work Completion:** 12 October 2026\
**Status:** Final documentation verification pending

---

## 1. Sprint Objective

The main objective of Sprint 02 was to establish a reproducible experimental environment for evaluating the behavior of a machine learning model under controlled data drift.

This sprint also prepared simulated evidence from Component 1 (Drift Detection) and Component 2 (Drift Diagnosis) so that Component 3 could begin development without depending on the other components being fully implemented.

The sprint focused on controlled experimentation, reproducibility, contract compatibility, and evidence collection rather than synthetic data repair itself.

## 2. Completed Tasks

| Task | Description                                    | Status      |
| ---- | ---------------------------------------------- | ----------- |
| 01   | Review Shared Drift Injection Framework        | Completed   |
| 02   | Implement and Test Controlled Drift Scenarios  | Completed   |
| 03   | Measure Baseline Model Performance Under Drift | Completed   |
| 04   | Create Reproducible Drift Scenario Dataset     | Completed   |
| 05   | Implement Simulated Component 1 Evidence       | Completed   |
| 06   | Implement Simulated Component 2 Diagnosis      | Completed   |
| 07   | Validate C1/C2 Input Contracts                 | Completed   |
| 08   | Document Drift Experiments and Evidence        | In progress |

## 3. Experimental Environment

The experiments used the preprocessed UCI Adult dataset and the shared XGBoost baseline model.

The reference dataset contained 26,048 rows and 13 columns, including the target.

For controlled drift experiments, a reproducible 6,000-row batch was sampled using random seed 42.

Eight scenarios, S00–S07, were generated using the shared drift injection framework.

The experiments evaluated how feature modifications affected predictions while retaining the original target labels.

These were controlled stress tests, not real-world production drift events.

## 4. Controlled Drift Scenarios

| Scenario | Drift Description                                 | Magnitude |
| -------- | ------------------------------------------------- | --------- |
| S00      | No drift                                          | 0.0       |
| S01      | Age marginal drift                                | 0.4       |
| S02      | Capital-loss marginal drift                       | 0.4       |
| S03      | Marital-status marginal drift                     | 0.4       |
| S04      | Race marginal drift                               | 0.4       |
| S05      | Age and education-number multi-feature drift      | 0.6       |
| S06      | Marital-status and relationship interaction drift | 0.6       |
| S07      | Combined age and interaction drift                | 0.6       |

The scenarios provided different drift patterns for testing Component 3's future repair logic.

## 5. Baseline Model Performance Under Drift

The model's accuracy on the original 6,000-row reference batch was approximately 88.83%.

The following results were obtained:

| Scenario | Accuracy | F1 Score | ROC AUC | Accuracy Drop (pp) |
| -------- | -------- | -------- | ------- | ------------------ |
| S00      | 0.8883   | 0.7535   | 0.9475  | 0.00               |
| S01      | 0.8802   | 0.7433   | 0.9392  | 0.82               |
| S02      | 0.8675   | 0.6962   | 0.9239  | 2.08               |
| S03      | 0.8800   | 0.7237   | 0.9279  | 0.83               |
| S04      | 0.8870   | 0.7464   | 0.9452  | 0.13               |
| S05      | 0.8652   | 0.7324   | 0.9309  | 2.32               |
| S06      | 0.8723   | 0.6987   | 0.9338  | 1.60               |
| S07      | 0.8603   | 0.6831   | 0.9199  | 2.80               |

S07 produced the largest accuracy reduction among these scenarios.

S02, S05, and S07 exceeded the experimental threshold of two percentage points of accuracy loss.

The results also demonstrated that accuracy alone may not fully describe drift impact, because precision, recall, F1 score, and AUC can change differently.

These measurements use the controlled reference batch and should not be confused with performance on the separate held-out test dataset.

## 6. Drift Sensitivity Experiments

Additional experiments examined the relationship between drift magnitude and model performance.

Age drift showed progressively larger accuracy reductions as the configured magnitude increased.

Capital-loss drift did not show a strictly monotonic relationship between magnitude and accuracy loss.

This demonstrates that the configured injection magnitude should not automatically be interpreted as a calibrated measure of downstream model harm.

## 7. Reproducibility and Dataset Integrity

A dataset manifest was generated to record experiment metadata and SHA-256 file hashes.

The manifest includes information about the reference data, model, configuration, drift scenarios, and experimental result files.

All eight scenarios were regenerated independently and compared with the original outputs.

**Reproducibility result: 8/8 regenerated scenario files matched the originals byte-for-byte.**

This provides evidence that the controlled experiments can be reproduced using the recorded configuration and environment.

## 8. Simulated Component 1 Evidence

Simulated C1 evidence was generated for all eight scenarios.

The payloads followed the official C1 contract containing 13 required fields.

They included drift type, drift severity, predicted harm, harm interval, affected features, detector scores, detector counts, and confidence statistics.

The simulation used measurable experimental information where available, including model confidence statistics and feature distribution differences.

However, several detector values were heuristic simulations.

The predicted harm was based on observed accuracy loss rather than a real predictive model, and the harm interval was not statistically calibrated.

These outputs are suitable for initial integration testing but must not be presented as results from the final Component 1 engine.

## 9. Simulated Component 2 Diagnosis

Simulated C2 diagnosis was generated for all eight scenarios.

Each payload followed the official nine-field C2-to-C3 contract.

The simulated diagnosis included root-cause information, affected feature rankings, affected slice rules, slice support, diagnosis stability, drift impact, and temporal window.

The slice rules were evaluated against reference data, and their support values were verified.

The simulated diagnoses were derived from known injection settings and predefined heuristics.

They were not produced by actual Temporal SHAP or Sliceline analysis.

## 10. C1/C2 Input Contract Validation

A unified input validation layer was implemented in:

`component3/src/contracts/input_validator.py`

It validates the individual C1 and C2 payloads and checks cross-component consistency.

The validator distinguishes between hard errors and non-fatal warnings.

All eight simulated input pairs passed validation with zero errors and zero warnings.

Additional negative tests confirmed that malformed fields, invalid values, inconsistent slice support, and incorrect diagnostic information could be detected.

An automated pytest suite was created in:

`component3/tests/test_input_contracts.py`

**Automated regression result: 18/18 tests passed using pytest 9.1.1.**

## 11. Research Findings

The main findings of Sprint 02 were:

1. Controlled drift can reduce model performance, but different features and drift types produce different effects.
2. Higher configured drift magnitude does not always produce greater accuracy loss.
3. Accuracy alone is insufficient to describe every type of model degradation.
4. Reproducible drift datasets and documented experiment settings are important for comparing future repair strategies.
5. Simulated C1/C2 inputs allow Component 3 to be developed independently while preserving interface compatibility.
6. Input validation is necessary before starting any synthetic data generation or repair workflow.

## 12. Limitations

The experiments used one primary dataset and one baseline model.

Drift was artificially injected into feature values while the original target labels were retained. Some modified feature-label combinations may therefore be unrealistic.

The experimental accuracy-drop measurements are controlled stress-test observations, not calibrated predictions of production performance.

C1 and C2 payloads were simulated, and their scientific correctness cannot be inferred from contract validation alone.

The current input validator reuses simulator validation functions and evaluates pandas query expressions. Production integration will require a separate review of dependency boundaries and safe expression handling.

No synthetic repair success, post-retraining improvement, or privacy guarantee was established during this sprint.

## 13. Sprint Deliverables

The sprint produced:

- A reviewed shared drift injection framework.
- Eight controlled drift scenario datasets.
- Model performance evaluation results.
- Drift magnitude sensitivity experiments.
- A reproducibility manifest with SHA-256 hashes.
- Simulated C1 drift evidence for eight scenarios.
- Simulated C2 diagnosis for eight scenarios.
- A C1/C2 input validation module.
- An automated regression suite with 18 passing tests.
- Seven task-specific technical documentation files.

## 14. Next Sprint

Sprint 03 focuses on building the core Component 3 architecture and repair-readiness functionality.

The next stage will include configuration handling, data loading, input schema validation, repair-readiness decisions, label trust checks, data sufficiency checks, affected slice extraction, and failure handling.

## 15. Conclusion

Sprint 02 established a reproducible controlled-drift experimentation environment and an initial validated interface between Components 1, 2, and 3.

The experiments demonstrated measurable changes in model performance under different drift conditions.

The simulated evidence and diagnosis payloads, together with automated input contract tests, provide a practical foundation for the next development stage.

These results support the implementation of Component 3 but do not yet demonstrate successful synthetic data repair.
