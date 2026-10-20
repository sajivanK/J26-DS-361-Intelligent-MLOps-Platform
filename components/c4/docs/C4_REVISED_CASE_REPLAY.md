# Revised C4 recorded-case assessment

This adapter adds a verified saved replay to the main C4 release page. It does
not execute new N1 inference or replace the original PP1 frozen controller.
The original case still calls `/api/c4/release/evaluate`; the revised case calls
`/api/c4/release/revised/evaluate` through the isolated C4 worker.

## Case and expected output

- Experiment: `adult-hybrid-fresh-validation-v1`.
- Case: `adult-ns51001-r004-interaction-low-broadened_slice-label_corrupted_repair`.
- Selection: lexicographically first admitted outer case in the frozen forecast
  batch. No future outcomes were inspected to select the demo case.
- Default model: hybrid Ridge / LightGBM / LightGBM; default family: adaptive.
- Expected decision: HUMAN_REVIEW; five mandatory gates passed.
- Global and true injected slice: UNCERTAIN. True complement: SAFE.
- Future safety summary: 1 / 3 established. Selected policy: Review only.
- Global pre-release champion accuracy: 85.4064%; candidate: 85.7143%; change:
  +0.3079 accuracy percentage points.
- Pre-release diagnosed slice and diagnosed complement have 1,211 and 413 rows;
  total: 1,624. These are not the populations used to define future slice targets.
- N2 adjustment for the hybrid: 0.6939113141 percentage points.

The repair variant deliberately includes corrupted labels as part of the
controlled benchmark. Its simulation admission is not production approval.
The SAFE complement meets its frozen -2 pp tolerance; it does not mean positive
predicted improvement. The other two boundaries are 0 pp.

## Integrity and execution scope

The reader checks the exact validation-protocol and pre-future forecast-batch
hashes, then checks the chosen case's feature, forecast, N2, policy, upstream,
candidate and evaluation JSON links. All 75 decision features reconstruct from
pre-release records. The reader verifies fixed quantile postprocessing, N2
adjustment and N3 classifications against the stored values, and checks the
selected review-only signal. It does not read raw data, model binaries or future
outcomes, fit anything, regenerate predictions, calibrate N2, select policies or
write research artifacts. Hash checks authenticate the pinned records; this
endpoint does not repeat model-binary validation or independently reproduce the
original predictions from fitted models.

The response marks `inference_executed: false`, `N1_executed: false`,
`N2_executed: false`, `N1_replayed: true` and `N2_replayed: true`. The main page
labels this execution mode. Production release remains unauthorized. Existing
C1/C2/C3 pipeline integration is unchanged.

## Validation before delivery

- 73 backend tests passed, including 21 new revised-case checks.
- Eight shared-router -> actual HTTP worker -> reader checks passed using the
  user's exported real case records (four forecasters, two policy families).
- 12 React interaction checks passed with mocked HTTP responses containing the
  real revised-case payload. They covered default selection, exact request
  routing, displayed candidate metrics, all three chart values/states, original
  case switching, stale-setting labels, request failure/recovery and research tabs.
- An isolated Vite production build passed using the available platform shell.
  The user must also build the current complete frontend locally.
- The original recorded-results JSON and chart component are unchanged.

These are software integration checks, not additional research performance
measurements. The existing revised research findings retain their limitations:
shared external Adult row cohort, held-out scenario groups, weak complement R2,
modest extra global gain over same-training LightGBM, and no measured autonomous
deployments. The replay does not establish live C1/C2/C3 integration, new-case
hybrid inference or production deployment readiness.

## User verification

After installing, restart the worker on 8004 with the C4 environment and
C4_DATA_ROOT, and restart the shared backend on 8000 with the C1 environment.
Run the new test file, the local frontend build and POST the revised endpoint.
Open `/c4`, evaluate the default revised case, check 5/5, 1/3 and Review only,
then switch to the original case and evaluate it. The existing application
verification script continues to exercise the original frozen controller.
No commit or push is part of this installer.
