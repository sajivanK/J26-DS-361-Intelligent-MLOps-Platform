# Sprint 04 — Task 01: Common Synthetic Generator Interface

## 1. Task Information

- Project: Intelligent MLOps Platform
- Component: C3 — Adaptive Intelligent Synthetic Data Repair Engine
- Sprint: 04 — Synthetic Data Generator Implementation
- Task: 01 — Design Common Synthetic Generator Interface
- Original planned sprint: October 1–5, 2026
- Actual implementation and verification: October 13, 2026
- Status: Implementation and tests verified; documentation pending final verification

## 2. Objective

Establish a consistent interface for CTGAN, TVAE, and TabDDPM so the repair engine can interact with different synthetic data generators through the same operations.

## 3. Existing Architecture Review

The repository already contained:

- `component3/src/generators/base.py`
- `component3/src/generators/ctgan_generator.py`
- `component3/src/generators/tvae_generator.py`
- `component3/src/generators/tabddpm_generator.py`

The existing implementations were inspected and reused rather than rewritten.

## 4. Common Generator Contract

The common interface is defined by `BaseGenerator`, an abstract Python class.

Required operations:

```python
def fit(self, data: pd.DataFrame) -> None:
    ...

def sample(self, n_rows: int) -> pd.DataFrame:
    ...
```

`fit()` accepts training data as a pandas DataFrame.

`sample()` requests a specified number of synthetic rows and returns a pandas DataFrame.

The abstract class prevents incomplete implementations from being instantiated.

## 5. Generator Implementations

### CTGAN

- Class: `CTGANGenerator`
- Inherits: `BaseGenerator`
- Uses the CTGAN library
- Detects discrete columns
- Supports fitting and sampling

### TVAE

- Class: `TVAEGenerator`
- Inherits: `BaseGenerator`
- Uses the TVAE implementation from the CTGAN package
- Detects discrete columns
- Supports fitting and sampling

### TabDDPM

- Class: `TabDDPMGenerator`
- Inherits: `BaseGenerator`
- Uses a PyTorch-based diffusion implementation
- Applies numerical transformation and categorical encoding
- Restores original column ordering after generation
- Provides an additional numerical range validation method

## 6. Design Decision

Retain the existing minimal `fit()` and `sample()` interface.

Do not introduce unnecessary abstract methods or rewrite functioning generator adapters.

Generator registration, selection, and identification will be handled by the generator plugin manager in a later Sprint 04 task.

## 7. Automated Interface Tests

Created:

`component3/tests/test_generator_interface.py`

The 20 test cases verify:

- Inheritance from `BaseGenerator`
- Required method signatures
- Rejection of sampling before fitting
- Rejection of empty training data
- Rejection of zero and negative sample counts
- Abstract base class behavior

These tests use lightweight checks and mocking where appropriate to avoid unnecessary neural network training.

## 8. Test Evidence

Existing CTGAN and TVAE smoke tests:

- Result: 2 passed
- Execution time: 125.13 seconds
- Failures: 0

Common generator interface tests:

- Result: 20 passed
- Execution time: 6.24 seconds
- Failures: 0

Full Component 3 regression suite:

- Previous baseline: 234 passed
- New tests: 20
- Final result: 254 passed
- Execution time: 14.96 seconds
- Failures: 0
- Warnings: 4

Warnings include existing XGBoost/scikit-learn serialized-model compatibility warnings and a PyTorch JIT deprecation warning.

## 9. Current Limitations

The common interface tests do not establish synthetic-data fidelity, privacy, downstream utility, or model recovery.

TabDDPM requires additional dedicated tests and numerical-output validation.

The interface does not itself enforce output schema preservation. Structural validation must be performed before synthetic records are accepted into a repair package.

The generator plugin manager has not yet been implemented or verified.

## 10. Conclusion

The existing common generator interface is suitable as the foundation for subsequent generator integration work.

All three generators implement the same required public operations, the new interface tests pass, and the complete existing Component 3 regression suite remains successful.

Next task: Sprint 04 — Task 02, Prepare Generator Training Data and Metadata.
