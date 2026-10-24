# C3 integration fixes and verification — 23 October 2026

C3's integration path has been hardened and its focused checks pass on the team's pinned Python 3.13 environment. A real C2-to-C3 run uses the shared preprocessed UCI values and the supplied model correctly. **The whole platform is not yet certified as fully working:** the exact team-laptop acceptance run is pending, shared storage changes need review, and the complete historical C3 suite is not green.

## Changes completed

| Area | Result |
|---|---|
| Missing C3 libraries | Backend router imports are lazy. C3 remains loaded in fallback mode, with `source: fake` and an explanatory dependency error in health/info. |
| Expected integration blockers | Fake C1/C2, unsupported IEEE diagnosis, missing onboarding/artifacts, unavailable labels and small slices produce explicit rejected/blocked results. No repair dataset is advertised. Genuine execution bugs still surface. |
| C2 slice coordinates | Rules run against the supplied CSV values unchanged. `model_ready` inputs reject a second external preprocessor. Regression coverage includes `age > -0.703 and marital_status == 2`. |
| Team model identity | The experimental Adult baseline no longer shares `adult-xgb-v1`. Team repair requires its explicit model context. A Git ignore rule that accidentally hid the model-registry Python source was corrected. |
| C3 storage | C3 uploads, demo batches and live repair outputs default outside the repository. External platform data and configured C3 storage roots are authorized automatically. |
| Label budget | When subsampling is necessary, the adaptation budget targets the diagnosed group after reserving evaluation data. A 200-row budget no longer randomly yields only a handful of affected rows. |
| Production label sample | An unlabelled current batch can supply `adaptation_dataset_uri` and independent `evaluation_dataset_uri`, with `label_status: AVAILABLE`. Adaptation rows must belong to the actual monitored batch; adaptation/evaluation overlap is rejected. Reference training labels are also required. |
| Label visibility | The C3 page explains the label requirements and shows adaptation and evaluation label counts. Synthetic labels are never treated as evidence that utility improved. |
| Optional configuration | Malformed C3 onboarding cannot stop shared C1/C2 ingestion. C3 rejects when its required context is missing. |
| Dependencies | Runtime requirements are complete and separate test requirements are supplied. Installation resolved successfully with all C1 pins. |

C1, C2, the shared injector/contracts, the platform orchestrator, Drift Lab and shared upload implementation were not edited. Only C3 files and C3's settings block/helper were changed in the application. No changes were committed, pushed or merged in this task.

## Verification results

| Check | Result |
|---|---|
| Fresh isolated interpreter | Python **3.13.15**, under ignored `scratch/`; the existing environment was not changed. |
| Team pins | numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.16.3, joblib 1.6.0, xgboost 3.4.1, lightgbm 4.6.0, torch 2.11.0. CTGAN **0.12.1** installed successfully. |
| Focused integration suite | **217 passed**, including fresh supplied models/datasets, real TVAE/CTGAN/TabDDPM smoke tests, API uploads/download integrity, fallback startup, encoded rules, label samples, budget selection and leakage checks. |
| Missing-library startup | A subprocess blocked imports of C3 libraries; the backend application imported and C3 loaded in explained fallback mode. |
| IEEE continuation | Adapter/orchestrator regression reaches a downstream C4 fixture with a contract-valid rejection. This is not a real IEEE model/data acceptance run. |
| Frontend | Production build passes. |
| Full C3 suite | **771 passed, 53 failed, 8 setup errors**, 234.66 seconds. This broader run preceded the final two additional onboarding/identity regression tests; all final changes are covered by the focused run. |
| Shared review patches | Both patches pass `git apply --check`; neither was applied. |
| Old integration upload directory | No existing contents were found to remove. |

The full-suite failures include absent historical model/candidate/prediction-audit files, scenario-support assertions inconsistent with the currently installed data, a fixed old reference row count (26,048 versus 39,073 now), missing learned-prescription artifacts, and older workflow/contract expectations. They are recorded rather than hidden or weakened. Reconstructing historical scientific evidence or regenerating private model artifacts was not attempted. Some archived sklearn artifacts also emit cross-version warnings under the team pins. Their passing tests do not establish artifact portability.

Evidence:

- [Focused Python 3.13 results](../../results/integration_guide_verification/py313-integration.xml)
- [Complete suite results](../../results/integration_guide_verification/py313-full.xml)
- [Every full-suite failure/setup error](../../results/integration_guide_verification/full-suite-failures.json)
- [Exact installed versions](../../results/integration_guide_verification/environment.json)

## Real C2-to-C3 check

The real C2 engine was warmed on two clean batches, then diagnosed the existing preprocessed S05 batch. C3 used the shared reference CSV and `shared/uci_adult/models/base_model_adult.pkl`, with `input_representation: model_ready`, accuracy as its repair metric and a 200-row adaptation budget. C2 audit logging was disabled for this isolated check; C2 code and stored history were not changed.

**C1 input was representative, not produced by a live C1 engine. This was not the requested capital_gain full-platform acceptance test.**

- C2 rule: `age > -0.3435 and age <= 0.2066 and marital_status == 2`.
- Actual affected rows: **497**, consistent with reported support **0.0828**.
- C3 generator training: **200 actual affected current rows**; independent target evaluation: **1,500 rows**.
- TVAE ran with the configured 15 epochs. Total C2 warm-up/diagnosis and C3 execution: **12.989 seconds** under Python 3.13.
- Result: **REJECTED**. Mean KS **0.551** exceeded **0.350**; DCR p05 **0.0021** was below **0.0500**, and near-duplicates failed the privacy gate. No approved repair CSV was offered.
- Model SHA256: `d90f096b439f369cf9871dc3e681fe6a1982712b5895562c33f7a9d5282e273e`.
- The local shared model scored **87.6446%** on the local **9,769-row** test file. The guide cites **87.44%** for its Windows-saved model. This local model/test pairing therefore does not certify the exact pairing on the team laptop; its actual model and split must be checked there. Loading the local pickle under XGBoost 3.4.1 also emits a serialization-version warning.

This establishes that C3 can consume real C2 output in the correct feature space and execute repair. It does not establish that this generator prescription improves the team's model. Approving this candidate would be incorrect.

[Machine-readable run evidence](../../results/integration_guide_verification/real_c2_probe.json)

## Team configuration and remaining acceptance work

The team's expected `C:\J26\data` artifacts are absent on this machine. No substitute model or dataset was installed there. The checked-in [context example](c3_contexts.example.json) specifies the exact 12 ordered features, seven encoded categorical features, model-ready input, accuracy metric and repair settings from the guide.

Once the team has placed the correct Windows-saved model and preprocessed CSVs at the guide's locations, run:

```powershell
python -m component3.src.configure_team_context
```

This checks paths/schema and writes `C:\J26\data\c3\c3_contexts.json`. It preserves other model entries and refuses to overwrite a different existing Adult configuration. It does not validate model accuracy; that requires the actual model acceptance run. The default path is read automatically; `C3_CONTEXTS_FILE` can override it.

For new models/datasets, provide their exact model, feature order, feature types, representation, reference/current artifacts and authentic labels. Supported tasks are currently sklearn-compatible **tabular binary classification**, with clone/fit/predict support. Compatibility with every arbitrary model, task or serialization environment is not claimed.

The production label sample path requires both labelled sample files; an adaptation sample alone cannot prove improvement. Independently labelled evaluation remains necessary beyond the adaptation budget. The platform's existing upload flow continues to mark batches without labels `UNAVAILABLE`; use C3's explicit contract/API to supply annotation artifacts.

## Shared changes prepared for team review

The guide's ownership rule says shared files need Sajivan's review before merging and contract documentation needs Kesigan's agreement. These reviewable changes are supplied without changing their files:

1. [Platform storage patch](platform-storage.patch): imports `C3_UPLOAD_DIR` in `uploads.py` and `routers/lab.py`, then replaces their old repository directory constants. Batch construction, label removal, C1/C2 inputs and pipeline policy remain the same.
2. [Contract documentation patch](contract-v1.1.patch): documents the existing v1.1 `generator: NONE` rejection and signed `utility_score` semantics. No schema change is proposed.

**Until the storage patch is reviewed and applied, Drift Lab/shared uploads still write their batches inside the repository.** Only C3-owned upload/demo/output paths have been migrated. The orchestrator's continuation policy and per-run badges remain Sajivan's work.

Suggested handoff to Kesigan (prepared, not sent): “Please confirm C4 accepts C3 v1.1 `generator: NONE` with REJECTED, zero rows and URI NONE, and signed `utility_score` in [-1,1] as a proportion. C4 should not retrain on rejected output.”

Still required on the team laptop:

- Install/verify the external context with the actual Windows-saved model and data.
- Run real C1 → C2 → C3 → C4 for UCI, capital_gain, strength 0.5, 2,000 rows, seed 7; record the result, screenshot and runtime.
- Verify a clean UCI batch follows the platform's no-action gate, and real IEEE completes with C3's clear no-diagnosis rejection.
- Review/apply the storage patch and check that subsequent platform batches stay outside the repo.
- Obtain C4 consumer agreement for the existing v1.1 semantics.
- Restore/validate historical research artifacts and reconcile the recorded full-suite failures before claiming all C3 tests pass.

## Reproducing the environment

Use a fresh Python 3.13 virtual environment. The four runtime installations below were resolved together under C1's constraints in the isolated check; they can also be installed in the guide's sequence:

```powershell
pip install -r c1/requirements.txt
pip install -r webapp/backend/requirements.txt -c c1/requirements.txt
pip install -r c2/requirements.txt -c c1/requirements.txt
pip install -r component3/requirements.txt -c c1/requirements.txt
pip install -r component3/requirements-test.txt -c c1/requirements.txt
python -m pytest component3/tests -q
```

The main C1 engine also has its separately documented alibi-detect setup; this task did not change that setup or verify a live C1 engine. The full test command's known failures are listed above.
