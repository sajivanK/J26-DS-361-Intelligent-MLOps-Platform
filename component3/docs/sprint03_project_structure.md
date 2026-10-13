# Sprint 03 — Component 3 Project Structure

## Project Information

**Project:** Intelligent MLOps Platform\
**Group ID:** J26-DS-361\
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Sprint:** 03 — Core C3 Architecture & Repair Readiness\
**Task:** 01 — Create Component 3 Project Structure\
**Implementation Date:** 13 October 2026

## 1. Objective

Prepare a modular and maintainable project structure for Component 3 while preserving the existing implementation.

The structure must support configuration management, data loading, input validation, repair-readiness processing, synthetic data generation, testing, and technical documentation.

## 2. Existing Project Structure

The project already contained the following directories:

- `configs/` — Experiment configuration.
- `data/` — Controlled drift datasets and simulated C1/C2 evidence.
- `docs/` — Research documentation and experiment reports.
- `src/contracts/` — Input contract validation.
- `src/experiments/` — Controlled drift experiments and simulations.
- `src/generators/` — CTGAN, TVAE, and TabDDPM generator implementations.
- `tests/` — Component 3 test scripts.
- `vendor/` — Vendored TabDDPM implementation and license information.

These existing modules were preserved.

## 3. New Architecture Packages

Three new packages were created:

### core/

Location: `component3/src/core/`

Purpose:

- Configuration loading.
- Dataset loading.
- Shared model and metadata loading.
- Core processing utilities.

### readiness/

Location: `component3/src/readiness/`

Purpose:

- Repair-readiness evaluation.
- Label trust checking.
- Data sufficiency checking.
- Affected slice extraction.
- Readiness decisions and failure handling.

### utils/

Location: `component3/src/utils/`

Purpose:

- Reusable helper functions.
- Common supporting utilities.
- Shared non-business-logic functionality.

Each new directory contains an `__init__.py` file.

## 4. Python Package Verification

The three packages were imported using Python 3.11.

Verification command:

`.\.venv\Scripts\python.exe -c "import component3.src.core; import component3.src.readiness; import component3.src.utils; print('PASS | All three C3 packages imported successfully')"`

Result:

`PASS | All three C3 packages imported successfully`

## 5. Git Working Tree Review

The Git working tree was inspected using:

`git status --short`

The output showed the new Component 3 packages and existing research artifacts as untracked.

No tracked file modifications or deletions were reported.

The working tree also contained an untracked `.vscode/` directory.

Untracked files require appropriate version-control handling before they can be considered committed project artifacts.

## 6. Architecture Principles

The Component 3 structure follows these principles:

1. Keep Component 3 code isolated from other research components.
2. Reuse existing implementation instead of duplicating code.
3. Separate core loading functionality from repair-readiness decisions.
4. Keep experiments separate from production-oriented modules.
5. Maintain dedicated tests and documentation.
6. Preserve generator implementations for the later synthetic data generation sprint.
7. Use shared integration contracts when communicating with other components.

## 7. Verification Summary

| Check                                          | Result   |
| ---------------------------------------------- | -------- |
| Existing folder structure reviewed             | PASS     |
| Missing architecture packages created          | PASS     |
| Python initialization files created            | PASS     |
| Python package imports                         | PASS     |
| No tracked modifications or deletions reported | PASS     |
| Project structure documentation                | Prepared |

## 8. Conclusion

The Component 3 project structure has been extended to support Sprint 03 development.

The new core, readiness, and utility packages are available and importable.

Existing experiments, contracts, generators, datasets, and documentation were preserved.

The project is ready for Task 02 — Configuration and Data Loading.
