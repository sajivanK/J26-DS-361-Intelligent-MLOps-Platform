# C3 readiness for real C1/C2 inputs and team-supplied models and datasets

> Historical audit of the implementation before the integration fixes. See [the current integration guide and verification report](C3_INTEGRATION_V2_HANDOVER.md) for the updated implementation and acceptance status. The findings below describe the earlier working tree.

Date: 23 October 2026. Scope: the current local working tree, including existing uncommitted implementation changes and the UI shown in the supplied screenshot. This is a code and targeted execution audit, not a certification of a deployed server. Application code was not changed during this audit.

## Answer to your question

**Your intended design is correct: C1 supplies drift/harm evidence, C2 supplies the diagnosed slice, and C3 should use that evidence together with the actual model, dataset, preprocessing and labels to attempt repair. Your current implementation does not yet reliably do all of that. Receiving all the required information from your teammates will not, by itself, make the current integration work correctly.**

C3 contains real generation and validation implementations. Its UCI interface can accept custom C1/C2 values and use them for readiness, generator selection and slice targeting. However, the current web path still binds execution to fixed Adult artifacts and a simulated drift batch. The separate contract API has additional data plumbing and compatibility gaps. Some configured demonstrations can execute, but successful repair of the supplied production batch and supplied model is not established.

| Situation | Current assessment |
|---|---|
| Run a configured Adult demonstration | Implementation and local artifacts exist; full training success was not reverified in this audit. |
| Supply C1/C2 dictionaries through the Adult demo API | Accepted and used, but execution still uses fixed Adult artifacts and simulated evaluation data. |
| Click the screenshot's Integrated Mode and Start Repair | Runs the selected scenario demonstration; does not fetch actual C1/C2 monitoring outputs. |
| Run the platform pipeline | C1/C2 outputs are passed to C3, but the UCI branch drops the real batch and model context. C2 currently returns fake output. |
| Supply a different model and dataset with all metadata | Not ready for general execution without integration changes and dataset/model onboarding. |
| Supply IEEE-CIS through the platform | Incorrect routing risk: every non-`uci` dataset enters the Telco branch. |
| Guarantee that synthetic repair improves accuracy | Cannot be guaranteed. Correct operation includes rejecting ineffective candidates. |

## Findings and evidence

### 1. The screenshot's automatic ingestion is currently scenario ingestion

In `webapp/frontend/src/pages/c3/C3Page.jsx:46`, the evidence comes from `c3ScenarioEvidence`. In `handleStartRepair` at line 177, Integrated Mode sends only `dataset` and `scenario` to the demo endpoint. Standalone Mode adds the editable C1/C2 dictionaries.

`component3/src/engine_uci.py:126` constructs the scenario evidence using the drift injector, `build_fake_c1_output`, and a locally constructed C2 diagnosis. At line 155, preset `forced_drop_pp` values overwrite the scenario's reported actual accuracy drop. These are demonstration values, so the displayed harm and diagnosis do not prove that real upstream components have been connected.

`handleUseLatestPipeline` at frontend line 213 imports an existing C3 result. It does not start a new repair from that run's C1/C2 evidence.

### 2. The UCI platform path ignores the supplied production batch and model

`webapp/backend/app/components/c3.py:91` forwards `custom_c1` and `custom_c2` to `execute_c3_repair_flow`, but does not forward the `batch`, supplied `model_id`, reference URI, current URI, preprocessing or label status.

`component3/src/engine_uci.py:109` loads fixed `shared/uci_adult` model/reference artifacts. At line 493, `execute_c3_repair_flow` always calls `get_scenario_evidence`; at line 512 it uses the scenario's `drifted_df` for evaluation, including when custom C1/C2 inputs were supplied.

**Executed probe:** a sentinel production batch, `team_supplied_model`, and `team_current.csv` were passed into `C3Component.run`. None reached the patched engine call. C1/C2 dictionaries did reach it. This probe checks argument plumbing without training a generator.

The upload endpoint has the same issue: `/api/c3/repair`, UCI branch, at `webapp/backend/app/routers/c3.py:270` invokes the default scenario flow without reading the uploaded file.

### 3. Other datasets are silently treated as Telco

After the `uci` branch, `webapp/backend/app/components/c3.py:107` fixes the dataset to `telco_churn`, model to `telco_rf_experimental_source`, target to `Churn`, and preprocessing to a Telco artifact. There is no explicit rejection of an unsupported dataset before this happens. The platform dataset list contains `ieee`, making this an immediate integration concern.

The Telco demonstration path may also return a previous approved candidate package when a broad slice-string check matches (`c3.py:139`). That is not proof of fresh generation or validation against the current production batch. Its demo router constructs fixed C1/C2 payloads rather than using the request's custom ones.

### 4. The contract API validates paths but does not complete the supplied-data flow

The canonical Python API, `execute_contract_repair`, accepts the useful three-section structure: `c1_evidence`, `c2_diagnosis`, and `data_context`.

However, `component3/src/api.py:102` resolves the reference path into `ref_p` without subsequently using it to construct the request's source data. It puts the current path into `adaptation_data_path` at line 163, but `RepairOrchestrator.execute_repair` does not read that field. Its default path loads data from the dataset registry (`repair_orchestrator.py:193`) and reconstructs experimental partitions.

That reconstruction assumes `customerID` at line 215. Its concept-drift simulation assumes `tenure` and `Yes`/`No` labels at line 225. These assumptions prevent treating this default execution path as general dataset support. The target column is taken from registry metadata, so supplying a different target in `data_context` is insufficient.

The lower-level Python API allows callers to supply partitions explicitly; this can support controlled experiments, but the canonical contract endpoint does not expose those partition arguments.

### 5. Real C1 v1.1 compatibility is incomplete in the contract validator

The shared C1 v1.1 schema uses `harm_severity` and `marginal_extent`/`interaction_suspected`; `drift_type` was removed. The validator at `component3/src/contracts/external_contracts.py:279` still requires legacy `drift_type` and `drift_severity`.

It also rejects negative harm estimates and negative confidence-interval lower bounds, although the shared C1 contract permits them. A negative lower bound can be a valid uncertainty interval.

**Executed probes:** a representative v1.1-shaped C1 payload failed because the two legacy fields were missing. Even after supplying those fields, an interval starting at `-0.01` failed validation.

The existing test named `test_real_c1_output_schema_compatibility` supplies the legacy fields itself. Its passing result therefore does not establish full current-v1.1 compatibility.

### 6. The current UCI gates can misrepresent readiness or repair success

In `component3/src/engine_uci.py`:

- Readiness hardcodes labels as `AVAILABLE` at line 236 and estimates affected rows from reported support and reference size rather than validating the actual slice population.
- An invalid slice rule falls back to the entire reference dataset at line 317; slices below 20 rows also fall back. An executed probe confirmed that an unknown feature passed readiness and then trained the mocked generator on all 1,000 reference rows.
- Coverage query errors are assigned `0.50` at line 409, which meets the coverage threshold.
- Utility always retrains a new `XGBClassifier` at line 436. It does not preserve an arbitrary supplied model family or its training recipe.
- Utility passes at `>= -0.01` at line 455: a loss of up to one percentage point can pass. The measured gain is overall scenario-batch accuracy, rather than independent affected-slice improvement against a matched real-only retraining control.
- Privacy calculation exceptions produce `PASS` and a placeholder DCR at line 471. Mean distance alone also does not establish zero exact copies.
- Selecting TabDDPM falls back to TVAE if `tab_ddpm` cannot be imported, while the output retains the prescribed TabDDPM name. That import was unavailable in the checked environment. A vendored TabDDPM implementation exists elsewhere but this UCI branch does not use it.
- The UCI prescription function uses explicit rules. The UI's learned meta-model description does not describe that execution path.

These are code findings; the audit did not run model training to quantify their effect. UI structural PASS and zero-copy labels are also partly hardcoded and should not be treated as validation evidence.

### 7. Full upstream/downstream integration is still incomplete

`webapp/backend/app/components/c2.py` has `source = "fake"` and calls `fake_c2`. C4 likewise calls `fake_c4`. Replacing C2 with the real diagnosis implementation remains required for the intended platform workflow. Replacing C4 is required to demonstrate actual downstream retraining/release behavior.

The platform orchestrator records output-contract problems but continues to later components. The UCI adapter/router also log some contract failures as warnings. For example, readiness rejection emits `generator: "NONE"`, which the shared generator enum does not allow.

## What your teammates need to provide

| Provider | Required handover |
|---|---|
| C1 | Actual v1.1 output for the monitored model and batch: harm estimate/interval with explicit units, severity, marginal/interaction signals, affected features, model identity and evidence identity. |
| C2 | Actual diagnosis for the same batch: root cause, slice rule, ranked features/ranges, support, stability, impact and temporal window. |
| Platform/model owner | Accessible reference data and exact current batch; model artifact and training recipe; fitted preprocessor; target and label mapping; feature order, types and encoding; label availability/provenance; dataset/model versions; separate adaptation and evaluation partitions. |

For example, if C2 supplies `education_num > 12` in raw units but C3 uses scaled values, the rule will not select the intended subgroup. The team must specify the representation used by every slice rule.

C1/C2 evidence does not itself provide ground-truth target labels. Synthetic data learned from old reference labels cannot be assumed to correct a changed target relationship. Delayed or unavailable labels require an explicit abstention/unevaluated policy or a separately validated approach.

## Required work before calling this integrated

1. Make one repair service accept validated C1/C2 evidence plus complete data context. Keep scenario generation behind an explicit demo path.
2. Bind execution to the actual supplied batch, reference data, model and fitted preprocessor; carry their versions/hashes through the result. Persist uploaded batches with usable paths and preserve authorized labels for C3 evaluation.
3. Normalize C1 v1.1 directly, including valid negative intervals, with explicit units. Retain marginal and interaction signals instead of inventing legacy evidence.
4. Remove Telco assumptions from general execution and require consistent dataset/model/target metadata. Reject unsupported datasets instead of substituting Telco. Verify the supplied model can be retrained through its adapter.
5. Validate slice features, expression safety, representation and observed support. Reject invalid/empty slices and unknown label status. Use the existing safe slice evaluator consistently.
6. Use independent labeled evaluation, an appropriate metric, affected-slice and source-retention checks, and a matched real-only retraining control. Fail or abstain when checks cannot run; report the generator actually executed.
7. Connect the actual C2 implementation and bind the UI's integrated action to real run evidence. Enforce output contracts before handing packages downstream.
8. Demonstrate acceptance on the teammates' actual artifacts: correct identities, current-batch consumption, valid no-action handling, invalid-slice rejection, ineffective-repair rejection, and C4 consumption.

## Verification performed and limits

The project `.venv` was used. Four checked Adult/Telco reference/model artifacts exist locally; `ctgan` and `xgboost` are importable, and `tab_ddpm` is unavailable through the import used by the UCI engine.

Existing test command:

```powershell
.\.venv\Scripts\python.exe -m pytest component3/tests/test_input_contracts.py component3/tests/test_external_contracts.py component3/tests/test_core_integration.py -q --disable-warnings --tb=short
```

Result: **11 passed, 26 failed, 6 setup errors**, with one warning, in 106.68 seconds. The failures and setup errors were missing `component3/data/drifted/S00.csv` through `S07.csv` fixtures, rather than proof that every corresponding feature is defective. The 11 external-contract tests passed, including their controlled execution fixture; that does not overcome the real-v1.1 and actual-batch gaps above.

Focused probe command:

```powershell
.\.venv\Scripts\python.exe component3/tests/audit_c3_integration_20261023.py
```

The probe uses mocks for generation and engine argument inspection; it does not claim to measure repair quality. Its saved results are in [focused_probes.json](../results/integration_audit_20261023/focused_probes.json). No fresh full generator training, deployed UI execution, teammate-supplied artifact acceptance test, or real C4 release was performed. Existing application changes were preserved.

**Readiness decision:** continue integration work. The implementation has useful repair components, but it is not yet ready to promise that providing all C1/C2 inputs and any supplied model/dataset will repair that exact model on that exact batch.
