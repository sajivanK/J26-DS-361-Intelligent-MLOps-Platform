# C4 Operational Preparation and Integration Handoff

## Verified checkpoint
Commit: d1ee2d5
Full C4 suite: 906 passed.
Actual-model integration test: passed using temporary synthetic data.

## Two separate application paths

### Recorded research-case assessment
The existing C4 page/API runs frozen N1/N2/controller inference on saved evidence.
Its measured policy is review-only.
Synthetic acceptance fixtures verify all three controller branches.

### Operational candidate preparation
The new service validates upstream inputs and trusted artifact bindings,
trains a candidate, evaluates it and saves versioned evidence.
It currently produces preparation status, not a release decision.

Operational N1/N2 forecasting and release authorization remain disabled.

## Commands
Run from the repository root with the verified Python environment active.

Read-only configuration check:
```powershell
python components/c4/scripts/check_runtime_preflight.py `
    --config components/c4/configs/runtime_bindings.example.json
```

The example intentionally reports CONFIGURATION_INCOMPLETE with exit code 2.

With actual reviewed configuration and upstream files, planning is:
```powershell
python components/c4/scripts/run_runtime_candidate.py `
    --config "<operator-config.json>" `
    --upstream-json "<platform-envelope.json>"
```

Planning does not train, write artifacts or verify model execution.
Explicit training requires --execute-training, --data-root and a new --version.
Do not execute against real inputs until their bindings and provenance are reviewed.

## Output storage
C4_DATA_ROOT/c4_artifacts/operational_candidates/adult/<version>/

Files:
- candidate.ubj
- evaluation.json
- manifest.json

Retain the returned manifest SHA256 in the calling audit record.
Existing versions are never overwritten.
Preserve partial directories after failures.

## Required team integration inputs
- Actual C1 champion identity, artifact and matching preprocessing.
- Actual C2 diagnosis, timestamps and explicit slice feature space.
- Actual C3 package, repair CSV and provenance.
- Labelled reference and release-validation sources with verified separation.
- Reviewed operator configuration containing trusted paths and hashes.

## Forecast limitation
The frozen pilot predicts true injected-slice accuracy margins.
It must not be relabelled as diagnosed-slice recovery.
Operational forecast applicability requires separate validation.

## Shared application integration
The leader merges the shared webapp.
Component merge order: C1, C2, C3, C4.
Reconcile local shared files before merging.
Apply C4 registrations only where still required.
Keep the shared C4 placeholder unchanged until its operational adapter is ready.

## Remaining verification
- Actual teammate-output compatibility.
- Operational forecasting/release-policy scope.
- Shared pipeline and operational frontend connection.
- Fresh-checkout setup with external artifacts.
- Full-system failure handling and timing.
