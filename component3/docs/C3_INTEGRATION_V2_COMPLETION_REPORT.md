# C3 real-input integration: completion report

Date: 23 October 2026. Implementation version: 2.0.0. C3-to-C4 contract: 1.1.

## Decision

The C3 implementation now supports actual C1/C2 evidence and newly supplied tabular binary-classification models/datasets through a defined contract. It uses the supplied artifacts, preserves the classifier family, and approves only candidates that pass independent validation. New dataset names do not need code changes.

This is not a universal guarantee of performance recovery. Unsupported task/model interfaces produce explicit errors, unavailable labels cause abstention, and ineffective or unsafe candidates are rejected. Acceptance on the teammates' exact artifacts remains pending because those artifacts and their actual C1/C2 outputs were not supplied in this conversation.

## Changes against the earlier audit

| Earlier issue | Implemented correction |
|---|---|
| Integrated UI used simulated C1/C2 | Real contract import, artifact uploads and latest-real-upstream ingestion; simulation has a separate mode |
| UCI path ignored batch/model | One live service consumes actual reference/current paths and supplied model/preprocessing; platform adapter checks batch equality |
| Unknown datasets became Telco | No dataset-name dispatch or substitution in live repair |
| Contract paths resolved but not used | Referenced CSVs are actually loaded; paths/hashes are recorded with the run |
| C1 v1.1 incompatible | Direct support for renamed fields, marginal/interaction evidence, signed intervals and explicit legacy harm units |
| Invalid slices fell back to all data | Safe AST evaluation; invalid predicates reject; actual support is checked; quoted feature names supported |
| Labels assumed available | Unavailable/unspecified labels block supervised approval |
| New model always became XGBoost | Clone/refit the supplied estimator or Pipeline; use the supplied fitted transformer |
| Privacy exceptions passed | Validation errors reject; feature-level exact-copy and fifth-percentile DCR gates enforced |
| Negative utility could pass | Positive practical gain and positive paired-bootstrap lower bound required |
| No real-only comparator | Compare candidate against matched real-only retraining on the same evaluation slice |
| Weak retention checks | Check source and overall-current retention as well as affected-slice gain |
| TabDDPM silently became TVAE | Real vendored TabDDPM is used; no hidden fallback; report executed generator |
| Stale approved candidates could be reused | Live API rejects pre-generated candidate injection and always runs fresh generation |
| UI displayed hardcoded PASS/zero-copy/learned claims | UI displays measured reports, provenance and the actual rule-based prescription mode |
| Invalid outputs flowed downstream | Platform stops on output-contract violations |
| Missing historical scenario CSVs | Regenerated S00–S07 locally; legacy simulation tests bind support to regenerated rows without rewriting archived evidence |
| Unapproved artifacts downloadable | Only approved, path-checked, hash-verified package artifacts can be downloaded |

## Files to use

- [Handover and operating guide](C3_INTEGRATION_V2_HANDOVER.md)
- [Request template](../examples/integration_request.json)
- [Live service](../src/live_repair.py)
- [Acceptance tests](../tests/test_live_repair.py)
- [Web API tests](../tests/test_webapp_c3_router.py)
- [Machine-readable verification](../results/integration_v2_verification/pytest.xml)

The platform's `POST /api/c3/repair`, standalone server and public `execute_contract_repair` API use the live service. Legacy registry-based research APIs remain available for experiments, but are not the real-input integration path.

## Verification

The selected integration and validation suite covers:

- Unregistered dataset IDs and supplied logistic-regression, random-forest and Pipeline artifacts.
- Separate fitted preprocessing, categorical features and string target mappings.
- Actual batch/model binding, input schema and identity errors, label abstention, unsafe slices, support mismatch, overlap and duplicate-group handling, adaptation budgets, and invalid positive labels.
- Real TVAE, CTGAN and TabDDPM one-epoch generation, followed by real validation. Rejection is an acceptable smoke-run outcome.
- Real Telco artifacts with explicitly separated current/reference batches.
- C4 output validation and rejected-package handling.
- HTTP contract submission, artifact uploads, approved CSV downloads and detection of tampered artifacts.
- Legacy readiness/input contracts, privacy/fidelity validators, generator registry, safe expressions and target coverage.

Final result: **195 passed, zero failures, zero errors, zero skips**, in 37.44 seconds. Counts and timings are recorded in `pytest.xml`. There were 12 third-party warnings. The frontend production build passed with `npm.cmd run build`. The actual platform app's C3 route registration was also checked: `/api/c3/info` returned 200 and an empty repair contract returned 422.

The suite was run with the project's `.venv`. To reproduce the older scenario tests when their ignored CSV fixtures are missing, first run:

```powershell
.\.venv\Scripts\python.exe component3/src/experiments/run_drift_scenarios.py
```

The main verification command was:

```powershell
.\.venv\Scripts\python.exe -m pytest component3/tests/test_live_repair.py component3/tests/test_webapp_c3_router.py component3/tests/test_external_contracts.py component3/tests/test_input_contracts.py component3/tests/test_core_integration.py component3/tests/test_server.py component3/tests/test_c4_consumer_simulation.py component3/tests/test_prescription_contract.py component3/tests/test_privacy_validator.py component3/tests/test_fidelity_validator.py component3/tests/test_generator_plugin_manager.py component3/tests/test_schema_security.py component3/tests/test_target_coverage.py -q --tb=short --disable-warnings --junitxml=component3/results/integration_v2_verification/pytest.xml
```

This is focused verification of the changed integration and its dependencies. It is not a claim that every archived experiment, external data acquisition path or unrelated component was executed.

## Team acceptance still required

C2 and C4 remain placeholder implementations in this checkout. C3 accepts their real externally supplied payloads now; it does not fabricate their research algorithms. The full automatic platform pipeline needs the real C2 implementation, model onboarding context and a C4 consumer agreeing to contract v1.1.

Supply the actual model, fitted preprocessing, dataset schema, reference/current CSVs, ground-truth labels and actual C1/C2 outputs. Prefer evaluation datasets known to be independent of the baseline model's historical training data. Run approved, ineffective-repair, no-harm and invalid-input cases and verify C4 consumes the approved artifact and hashes.

Restart the backend/frontend to load the updated implementation. No model was deployed and no production release was performed.
