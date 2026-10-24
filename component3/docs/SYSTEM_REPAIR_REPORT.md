# System repair report — 24 October 2026

Follow-up: see [TRAINING_REMEDIATION_REPORT.md](TRAINING_REMEDIATION_REPORT.md) for the subsequently authorized C2 threshold-formatting fix, longer discrete-aware generator training and bounded privacy sampling. The original measurements below remain historical evidence.

The supported UCI integrated pipeline now binds C1, C2 and C3 to the same monitored model and processes the actual incoming batch and C2 slice. C3's dashboard uses backend evidence rather than invented measurements. C1/C2 core research implementations and shared contract schemas were not edited. Existing uncommitted work was preserved; nothing was committed or pushed.

## What was wrong and what changed

| Area | Finding and correction |
|---|---|
| Model identity | C1/C3 used `data/c2/base_model_adult.pkl`, while C2 defaulted to a different shared model. Platform configuration now defaults C2 to the same local model/reference/test artifacts, respecting explicit environment overrides. C3 checks fitted model state, including XGBoost trees, objective, base score, feature names and classes, before repair. |
| Batch identity | Uploads and Drift Lab persist the actual labelled batch in configured C3 upload storage. C1/C2 receive the feature view. Model, dataset, pipeline run, drift event and batch IDs are retained in C3 provenance; supplied upstream identity mismatches fail. |
| Automatic orchestration | Full pipeline still automatically passes actual C1/C2 contracts to the same real C3 service used by standalone requests. The gate now requires detected drift, non-none severity and predicted harm of at least .01. Clean/no-harm batches stop without forced repair. |
| Pipeline visibility | The orchestrator publishes running stages and current outputs during execution. Pipeline runs are serialized on the single local worker so the displayed latest run remains coherent with C2's temporal state. |
| C3 progress | Fourteen named stages are saved from actual execution events, with timestamps and measured details. Waiting stages are never marked complete by a timer. Generator duration/rows/configuration are reported; epoch loss is not fabricated. |
| Quality policy | Restored local KS maximum .35, DCR minimum .05, no exact/near copies, utility minimum .01, no permitted loss versus the real-only control, diagnosis stability minimum .50 and positive paired bootstrap interval. Requests can tighten these approval gates, but cannot weaken them. Source-retention tolerance remains the documented .05. |
| Honest UI | Removed invented model/dataset fallbacks, example records, fallback utility/KS/DCR values, synthetic counts, confidence, timestamps and automatic PASS badges. Missing results show unavailable/not evaluated. A newer clean or running pipeline no longer displays an older repair. |
| Diagnosis | The exact C2 rule is evaluated on unchanged current feature values; observed support and actual adaptation/evaluation row counts remain auditable. Model feature order is checked before generation. |
| Generation and quantity | Actual TVAE/CTGAN/TabDDPM execution remains supported; no generator or dataset substitution. Quantity uses eligible slice adaptation rows and the requested ratio/cap. Automatic versus manually requested generator choice is visible. |
| Scientific evaluation | Existing independent partitions, duplicate-group leakage prevention, model-family cloning, real-only control and source/overall retention checks remain in place. A positive bootstrap interval is now required for approval. Quality failure stops before utility evaluation. |
| Results and downloads | Results and stage events persist to disk. The dashboard can select saved repairs, including older manifests, and shows each run's own evidence. Only an approved run exposes a download; existing artifact-integrity checks protect the download. |
| Registration and overview | Backend reports actual configured model/reference artifact paths and availability, loaded monitoring algorithm, saved pipeline counts and blocked repair counts. |
| Standalone | The dashboard accepts a three-section research contract and invokes the same artifact-bound service. Separate scenario demonstration endpoints remain explicitly separate from integrated execution. |
| C4 | The existing adapter receives the contract and reports its fake source. C3 UI explicitly calls it a stub, without claiming actual retraining or release. No teammate's C4 implementation was replaced. |
| Local runtime | Completed the existing isolated Python 3.13 environment with C1's documented runtime dependencies. The original `.venv` was not changed. Added a local launcher that limits default CPU thread counts, starts hidden background processes and preserves processes already serving the ports. |

## Measured acceptance evidence

Full request/response evidence is saved in `component3/results/system_repair_acceptance/acceptance.json`.

| Experiment | Actual result |
|---|---|
| Clean batch | Real C1 predicted .0037 harm; pipeline stopped with `no_action`. No C2/C3 generation was forced. |
| Initial harmful multi-feature run | The new model binding check blocked repair because C2 had loaded the other shared model. This exposed the integration defect before any wrong-model repair. |
| Same experiment after configuration repair | Pipeline `d2bd4c6e08a4` completed in **3.265 seconds**, using real C1/C2/C3 and the C4 stub. |
| Model/batch | `adult-xgb-v1`, UCI, 2,000 rows; `lab_fd5744b399`. Model SHA256 `02c1f225867bb001f143305625e671fee56698974a2dc509472bfe2c2f75cfc9`. |
| C1 | Predicted harm .0627, interval [.0448, .0807], moderate severity. |
| C2 | Stability 1.0; slice `age > -0.4428 and age <= 0.4037 and education_num > 1.818`; reported support .075. |
| Real C3 | TVAE trained on 124 actual adaptation-slice rows and generated **38** records. Quantity is `ceil(124 × .30)`. Target coverage **1.0**, structure passed. |
| Quality decision | **REJECTED**. Mean KS .413582 exceeded .35. DCR p05 .0043 was below .05; near-duplicate ratio .3158 exceeded 0. Exact copies were 0. |
| Utility/download | Utility was **not evaluated** after failed quality gates. No approved dataset URI/download was exposed. This is a safe completed execution, not evidence of model improvement. |

The measured rejection was retained. No threshold tuning, repeated candidate search or invented approval was used to obtain a favourable result.

## Focused verification

- Initial focused regression batch: **62 passed**, 3 slow real-generator cases excluded.
- Follow-up checks for the new safety policy, identity/progress and adapter/API changes: **42 passed**. These overlap with the first batch; they are not 104 distinct tests.
- A final isolated regression for legacy manifest status deserialization passed. Older manifests now load without breaking the dashboard API.
- Frontend production build passed after the final frontend changes.
- Real connected C1/C2/C3 execution and contract handoff were exercised as recorded above.
- Final running API checks passed: real C1/C2/C3 loaded, registered XGBoost artifacts available, correct persisted run/stages returned, 45 current/historical repair records loaded, rejected download returned HTTP 409, frontend returned HTTP 200. Evidence: `component3/results/system_repair_acceptance/final_api_check.json`.
- The first regression attempt used a scratch temporary directory outside the artifact allowlist. The harness was corrected to use its authorized temporary root; production path authorization was not weakened.
- Browser automation exposed no usable browser, so visual browser inspection could not be performed. The frontend build and local HTTP serving were checked; do not treat this as screenshot-based UI acceptance.
- No full historical benchmark suite or repeated generator sweep was run. Multi-model safety is covered by supplied logistic, forest and sklearn Pipeline regression cases; this is not real multi-dataset C2 acceptance.

## Start and inspect

From the repository root:

```powershell
.\scratch\c3-py313\Scripts\python.exe webapp/start_platform.py
```

- Dashboard: `http://127.0.0.1:5173/c3`
- Drift Lab: `http://127.0.0.1:5173/lab`
- API: `http://127.0.0.1:8000/docs`
- Logs: `scratch/platform-backend.log`; frontend launcher output goes to `scratch/platform-frontend.log` when launched through this script.

The launcher leaves existing port listeners unchanged. Restart an existing backend after code changes. Do not use multiple backend workers with this in-process pipeline/C2 state model.

The isolated environment is local and ignored by Git. To reconstruct it elsewhere, create Python 3.13, install platform/C2/C3 runtime requirements constrained by `c1/requirements.txt`, then install `alibi-detect==0.13.0 --no-deps` and its documented light dependencies. C1's own requirements explain why Alibi metadata needs this separate step. Exact installed package versions are recorded alongside the acceptance evidence. Real model and dataset files must be supplied separately; the system does not fabricate them.

## Remaining boundaries

- **Real C4 is absent.** The repository contains a stub only; production retraining/release cannot be certified until the owner supplies their implementation.
- **Real C2 currently supports UCI only.** IEEE monitoring can run C1, but C3 blocks unsupported downstream diagnosis. New datasets require real C2 support and explicit compatible model/data onboarding.
- **No successful real repair is claimed for the measured experiment.** Better synthesis requires separately justified generator training/data experiments under the same quality policy; rejection must remain visible until the measurements pass.
- Supported C3 scope remains compatible tabular binary classifiers. Regression/multiclass/image/text models require implementation rather than an automatic fallback.
- Data leakage checks cover C3 candidate-training partitions. The original model's training history remains an owner-supplied provenance limitation, as recorded in each package.

These changes address the supplied architecture's supported integration and truthful UI requirements. They do not turn an unimplemented C4 or a failing synthetic candidate into a completed production release.
