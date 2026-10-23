# Sprint 05 — Synthetic Data Validation Framework Design

## 1. Project Information

Project: Intelligent MLOps Platform
Group: J26-DS-361
Component: Component 3 — Adaptive Intelligent Synthetic Data Repair Engine
Sprint: 05 — Validation Engine and Repair Experiments
Task: 01 — Design Synthetic Data Validation Framework
Status: Architecture Design

## 2. Objective

Design a modular validation framework that evaluates synthetic
tabular data before it is considered for downstream model repair.

The framework must distinguish structural validity, statistical
fidelity, privacy risk, and offline predictive utility.

## 3. Existing Reusable Components

- generators.target_coverage.evaluate_target_coverage
- generators.data_preparer.PreparedGeneratorData
- readiness.readiness_models.ReadinessDecision
- contracts.safe_slice_evaluator

The validation subsystem must reuse existing functionality
where appropriate and avoid duplicating safe slice parsing.

## 4. Proposed Package Structure

component3/src/validation/
    __init__.py
    validation_models.py
    validation_engine.py
    structural_validator.py
    fidelity_validator.py
    privacy_validator.py
    utility_evaluator.py
    acceptance_policy.py

These modules are proposed and not yet implemented.

## 5. Validation Inputs

Required:
- Synthetic records as a pandas DataFrame
- Real affected-slice records as a pandas DataFrame
- Explicit continuous and categorical feature metadata
- Target-slice rule
- Generator identifier
- Validation configuration

Optional for offline utility:
- Trusted training labels
- Baseline model and preprocessing resources
- Held-out evaluation data with trusted labels
- Controlled retraining configuration

## 6. Structural Validation

Check:
- Dataset is not empty
- Required columns are present
- Unexpected columns are handled explicitly
- Data types and feature metadata are consistent
- Missing and non-finite values
- Categorical domain validity
- Numerical constraints
- Target-slice coverage

Reuse the existing target coverage evaluator.

Output:
- Stage status
- Structural metrics
- Failure reasons

## 7. Fidelity and Diversity Validation

Evaluate:
- Continuous feature distribution similarity
- Categorical feature distribution similarity
- Feature relationships
- Correlation differences
- Duplicate rates
- Numerical boundary concentration
- Diversity indicators

Candidate metrics:
- Wasserstein distance for continuous features
- Jensen-Shannon divergence for categorical features
- Correlation difference for selected feature pairs

Metric normalization and acceptance thresholds require
calibration and must not be treated as universally valid.

## 8. Privacy Risk Validation

Evaluate:
- Exact synthetic-to-real record matches
- Nearest-neighbor similarity
- Potential memorization indicators

Use mixed-type-aware comparisons where applicable.

Privacy metrics are risk indicators, not privacy guarantees.

## 9. Offline Utility Evaluation

Evaluate whether adding validated synthetic records to
a controlled retraining experiment improves the model.

Required safeguards:
- Trusted training labels
- Separate held-out evaluation data
- No training/test leakage
- Baseline versus repaired-model comparison
- Affected-slice performance
- Overall performance
- Non-affected population performance

If trustworthy labels or evaluation resources are unavailable,
mark utility as NOT_EVALUATED rather than assuming success.

## 10. Validation Status Model

Per-stage statuses:
- PASS
- FAIL
- REVIEW
- NOT_EVALUATED

Final decisions:
- ACCEPT
- REVIEW
- REJECT

An incomplete mandatory validation stage cannot result in ACCEPT.

## 11. Acceptance Policy

REJECT:
- Mandatory structural or safety requirement fails
- A configured hard rejection rule is triggered

REVIEW:
- Required evidence is missing
- Results are inconclusive
- Manual approval is required

ACCEPT:
- All mandatory checks pass
- Required evidence is available
- Intended-use acceptance policy is satisfied

Thresholds must be configurable and empirically justified.

## 12. Validation Output Contract

The final report should contain:
- Generator name
- Dataset and scenario identifiers
- Input row counts
- Feature metadata
- Per-stage statuses
- Per-stage metrics
- Failure and review reasons
- Final decision
- Configuration/version provenance
- Experiment identifiers

Use JSON-serializable output structures.

## 13. Integration Boundaries

The validation subsystem receives outputs from generators.

It does not:
- Modify the generator training algorithms
- Automatically authorize supervised retraining
- Replace Component 1 or Component 2
- Deploy models
- Make Component 4 decisions

Future modules may consume validated repair evidence.

## 14. Testing Strategy

Plan unit tests for:
- Valid inputs
- Empty data
- Missing columns
- Incorrect metadata
- Invalid categories
- Out-of-range values
- Low target coverage
- Distribution mismatch
- Exact-record copying
- Missing trusted labels
- Missing evaluation data
- ACCEPT/REVIEW/REJECT policy behavior
- Input immutability
- JSON serialization

Use synthetic fixtures and controlled-drift scenarios.

## 15. Research Limitations

This document defines an architecture, not experimental
validation results.

No fidelity, privacy, utility, or acceptance metrics are
claimed as implemented by this design alone.

Metric thresholds require later calibration.

## 16. Sprint 05 Implementation Sequence

Task 02: Structural and target coverage validation
Task 03: Fidelity validation
Task 04: Privacy validation
Task 05: Offline utility evaluation
Task 06: Acceptance and rejection decisions

Later tasks will integrate these modules into automated
repair experiments.

## 17. Conclusion

The proposed framework separates synthetic-data generation
from synthetic-data quality assurance.

Its purpose is to ensure that high target coverage alone
does not qualify synthetic records for model repair.

Implementation and experimental evaluation will follow
in subsequent Sprint 05 tasks.
