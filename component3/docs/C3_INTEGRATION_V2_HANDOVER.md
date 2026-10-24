# C3 integration v2: implementation and handover

Updated: 23 October 2026. This supersedes the earlier C3 integration audit.

## What works now

C3 has one artifact-bound live service for actual C1/C2 evidence and caller-supplied models and datasets. New dataset IDs do not need to be hardcoded or registered. The platform adapter, public contract API, standalone server and C3 web page use that service. Research simulations are explicit and separate.

For supported inputs, C3 validates the evidence and artifacts, extracts the actual diagnosed current-data slice, trains a real generator, retrains a clone of the supplied classifier, evaluates it on independent rows, and returns an approved, rejected or blocked package with reasons. It never switches an unknown dataset to Adult/Telco or replaces the supplied classifier with XGBoost.

**This makes C3 ready for the defined integration contract. Final acceptance of your teammates' exact artifacts still requires running their actual payloads. No system can guarantee that every synthetic repair improves every model. A rejected ineffective repair is a correct result.**

## Supported scope

| Item | Support |
|---|---|
| Task | Tabular binary classification |
| Model | A fitted sklearn-compatible classifier with `classes_`, `get_params`, `fit`, `predict`, and support for `sklearn.base.clone`; sklearn Pipelines are supported |
| Examples verified | Logistic regression, random forest, fitted scaler/classifier Pipeline, separate fitted ColumnTransformer, categorical features and string-label mappings; existing Telco RF artifacts |
| Generators | Real TVAE, CTGAN and the repository's vendored TabDDPM; failures are reported without a hidden fallback |
| Input data | CSV, explicit feature order/types, target column, no missing or non-finite selected values |
| Metrics | F1, accuracy or ROC AUC; AUC requires `predict_proba` |
| Targets | Native model labels or an explicit `target_mapping` matching `model.classes_` |
| Unsupported tasks | Regression, multiclass, images/text/sequences and non-cloneable custom training systems require additional adapters; they are not covered by this implementation |

The supplied fitted transformer is used without refitting. A complete sklearn Pipeline is cloned and its training recipe is refitted on candidate training data. If neither is supplied, declare `input_representation: "model_ready"`; the provided columns must already be exactly what the estimator expects.

## Team input contract

Start with [integration_request.json](../examples/integration_request.json), replace its example evidence, schema and paths with the real values, and submit:

```json
{
  "c1_evidence": { "...": "actual C1 output" },
  "c2_diagnosis": { "...": "actual C2 diagnosis" },
  "data_context": { "...": "actual model, data and schema references" }
}
```

The example JSON is a template, not a ready-to-run dataset. Its `slice_support: 0.5` must be replaced with the support measured by C2 on the supplied current CSV.

C1 v1.1 fields are supported directly, including `harm_severity`, marginal/interaction signals and negative uncertainty bounds. `predicted_harm` and `harm_interval` use proportions: `0.052` is 5.2 percentage points. Legacy `predicted_accuracy_drop` defaults to percentage points; set `harm_unit: "proportion"` if that legacy field contains a proportion. C1 model identity, optional dataset URIs, optional model hash and batch size are checked against the run.

C2 supplies the root-cause feature/category, slice rule, ranked affected features, support, stability, impact and optional temporal window/ranges. Slice rules operate on the exact CSV values used for generation. Use backticks for feature names containing spaces, such as `` `Total Charges` > 100 AND Contract == 'Monthly' ``. Rules use the restricted AST evaluator; Python calls are rejected.

`data_context` must include model and dataset IDs, actual reference/current URIs, target, ordered `feature_columns`, `categorical_features`, and `label_status`. Supply a model URI/path for new models, plus the fitted preprocessing URI or complete Pipeline/model-ready declaration. A registered model ID can resolve an existing model artifact; an explicit supplied model never inherits an unrelated registered preprocessor.

Optional fields include `target_mapping`, `positive_label`, `evaluation_dataset_uri`, `source_evaluation_uri`, `artifact_sha256`, `repair_metric`, `generator`, `generator_epochs`, `augmentation_ratio`, `max_synthetic_rows`, `minimum_utility_gain`, adaptation budget and retention tolerance. A string-label mapping example is `{"No": 0, "Yes": 1}` for an estimator trained on 0/1 labels. `positive_label` names the positive class in the model's label space; it defaults to the second entry of `model.classes_`.

Use observed labels. Declaring labels available does not create them. If labels are unavailable or unspecified, C3 blocks supervised repair approval rather than assuming their availability. A reference-label generator is not a guaranteed solution to a changed target relationship.

## How to run it

Start the platform from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir webapp/backend --port 8000
```

Start the frontend in another terminal:

```powershell
cd webapp/frontend
npm.cmd run dev
```

On the C3 page, select **Real C1/C2 integration**, import the actual contract JSON, upload artifacts if needed, check feature names/target/label mapping and start repair. The JSON editor displays exactly what will be submitted. **Load latest real C1/C2 inputs** accepts real upstream runs and rejects placeholder evidence.

The HTTP interface is `POST /api/c3/repair` with the three-section JSON body. For example:

```powershell
$repairPayload = Get-Content -Raw component3/examples/integration_request.json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/c3/repair -Method Post -ContentType application/json -Body $repairPayload
```

Run this only after replacing the template with your real artifact references. Python callers can use:

```python
from component3.src.api import execute_contract_repair

contract, package, log = execute_contract_repair("my_actual_request.json")
print(package.package_status, package.rejection_reasons)
```

Alternatively, the standalone server remains available:

```powershell
.\.venv\Scripts\python.exe -m component3.src.server --port 8003
```

It uses the same live service. Its response includes both detailed validation/provenance and the exact `c4_flat_contract`.

Artifacts can be uploaded to `POST /api/c3/artifacts` on the platform with multipart fields `role` and `file`. Roles are `model`, `preprocessor`, `reference_dataset`, `current_dataset`, `evaluation_dataset`, and `source_evaluation`. The endpoint returns a unique URI and SHA-256 hash. Models use trusted `.joblib`, `.pkl` or `.pickle` artifacts; datasets use CSV. The UI places returned paths and hashes into the contract. Larger files can use authorized local paths instead of the 50 MB upload endpoint.

For artifacts outside the repository, configure `C3_DATA_ROOTS` and `C3_MODEL_ROOTS` with semicolon-separated Windows directories. All paths are resolved and checked. Missing artifacts and unknown models produce explicit errors.

For the full platform pipeline, an optional `C3_CONTEXTS_FILE` points to a JSON object keyed by exact model ID, with each value containing that model's data context. Batch uploads and Drift Lab runs now persist their actual labeled CSV; C1/C2 receive the feature view while C3 reads the authorized labeled artifact. The adapter verifies that this CSV corresponds to the monitored batch.

## Approval policy

1. Meaningful harm, stable diagnosis, available labels and actual current slice support are required. At least 100 labeled reference/current rows, 50 affected current rows and 5% support are required; after partitioning there must be at least 30 generator-training and 20 evaluation rows in the slice, with both evaluation classes present.
2. C3 uses caller-provided independent evaluation files or derives stratified holdouts. Duplicate feature records stay in the same derived partition. Reference rows also present in the current batch are excluded from source partitions. Overlapping supplied training/evaluation records are rejected. The adaptation budget limits current training rows.
3. The generator sees only the diagnosed adaptation slice. Evaluation rows do not train the generator or candidate. There is one candidate per request, so evaluation results do not drive a generator/quantity search.
4. Generated structure, known discrete values, finite numerical values, target coverage (80%), statistical fidelity and feature-level privacy must pass. Privacy requires zero exact copies and DCR fifth percentile at least 0.05. Calculation failures reject the candidate. The DCR implementation uses bounded distance blocks.
5. Retraining preserves the supplied model family and recipe. Candidate affected-slice gain must meet the minimum margin (default +0.01), have a positive paired-bootstrap 95% lower bound, match or exceed real-only retraining, and preserve source and overall-current performance within the configured tolerance.
6. Only approved datasets/models are published as candidate artifacts. Rejected runs have zero approved rows, `repair_dataset_uri: "NONE"` and no download link. Download endpoints verify package status, path and artifact hashes.

The model/data owner must establish that supplied evaluation data was not used to train the original baseline. C3 can verify overlap with its candidate training partitions, but cannot reconstruct unknown historical baseline training data. The package records whether evaluation was caller supplied or derived.

## C4 contract change

The 12 output field names are preserved. The shared C3 output contract is now version **1.1**:

- `generator` reports the generator actually executed. `NONE` is permitted only on a rejected/no-generation result.
- `utility_score` consistently means affected-slice improvement in the selected metric, in proportions. Rejected candidates can have a negative value; it is not an absolute F1 score.
- Detailed metric names, baseline/control scores, confidence intervals, hashes, input schema and run provenance are in the package/UI response.

C4 consumers must use this version's semantics. C3 approval means a validated candidate is ready for C4 review; it does not deploy a production model.

## Verification and remaining team action

The integration acceptance suite exercises new dataset IDs, multiple supplied classifier families, fitted preprocessing, signed C1 intervals, actual current-batch binding, schema/slice errors, label abstention, copied-data rejection, privacy exceptions, evaluation overlap, budget enforcement, uploads, contract enforcement and hash-checked downloads. Real one-epoch smoke runs execute **TVAE, CTGAN and TabDDPM**. Generator smoke success establishes execution and honest rejection handling, not universal improvement.

The verification evidence is [pytest.xml](../results/integration_v2_verification/pytest.xml). The production frontend build is verified with `npm.cmd run build`. The selected integration/validation suite is documented in [the completion report](C3_INTEGRATION_V2_COMPLETION_REPORT.md); this is not a claim that every unrelated research experiment in the repository was rerun.

**C2 and C4 platform implementations are still owned by their respective components and remain placeholders in this checkout.** The real-input C3 API accepts actual supplied C1/C2 contracts now. Full platform acceptance requires connecting the real C2 implementation, configuring the model context, and having C4 consume contract v1.1. C3 explicitly blocks fake C2 evidence from masquerading as a real integration run.

Your team's final acceptance should run the actual target model, current/reference datasets, fitted preprocessing and C1/C2 outputs through this path. Review both approved and rejected cases and confirm C4 consumes the approved artifact and its hashes. Those artifacts were not supplied in this conversation, so acceptance of those exact inputs remains unverified.
