# Component 3 final metric reconciliation

Date: 2026-10-21 (Asia/Colombo)

## Decision

`RUN-6214C72F25` is a real recorded, approved-for-review TVAE package. Its candidate CSV and model hashes match the manifest, and the selected quantity is 100. The apparent metric conflicts came from displaying two evaluation protocols as though they were one and from the C4 adapter's documented fallback behavior.

### Authoritative values

| Context | Metric | Value | Interpretation |
|---|---|---:|---|
| Flat C4 handoff | `utility_score` | `0.6400` | Candidate positive-class F1 on authentic `D_dev_val`. |
| Authentic retraining | baseline target positive F1 | `0.6286` | Derived from `0.6400−0.0114`; same evaluation context as the C4 utility. |
| Authentic retraining | target F1 gain | `+0.0114` | Clears the `+0.0100` minimum by `0.0014`. |
| Unified validation pipeline | affected-slice F1 | `0.9231 → 0.9615` | Separate 52-row validation-stage comparison; valid, but not the C4 utility context. |
| Unified validation pipeline | F1 gain | `+0.0384` | Same 52-row context only. |
| Unified validation pipeline | composite utility | `0.9423` | Accuracy-centered composite, not F1 and not exported. |
| Flat C4 handoff | `fidelity_score` | `0.7236` | `1−mean_KS`; the adapter fallback used by the recorded package. |
| Internal fidelity report | composite fidelity | `0.7893` | `1−mean(KS, JS, normalized correlation penalty)`. |
| Internal fidelity report | mean KS | `0.276410…` | Mean across tenure, MonthlyCharges and TotalCharges; gate maximum `0.350`. |
| Internal fidelity report | mean TVD | `0.281674…` | Mean across categorical columns; descriptive, not the directly gated categorical statistic. |
| Flat C4 handoff | augmentation ratio | `0.0267` | `100/3752`; 3,752 is the post-augmentation training mixture. |
| Authentic source retention | candidate source F1 / delta | `0.5939 / −0.0137` | Same `D_source_eval`; passes the `−0.0500` tolerance. |
| Privacy | DCR p05 / exact matches | `0.2237 / 0` | Empirical nearest-neighbour evidence; PASS, not formal differential privacy. |
| Coverage | target coverage | `1.0000` | All 100 accepted rows match the target predicate. |
| Prescription | confidence | `0.8000` | Heuristic mapping from `EXPLORATORY_PILOT`; not a calibrated probability. |

### Why `0.7236` and `0.7893` both exist

The manifest's top-level `fidelity_indicators` stores `overall_fidelity_score`, while `C4RetrainingAdapter.to_c4_flat_contract` first looks for a key named `fidelity_score`. Since that exact key is absent at the top level, the adapter uses its fallback `1−mean_ks_statistic`, producing `0.7236`. The internal stage report does contain `metrics.fidelity_score=0.7893`, calculated from continuous KS, categorical JS divergence and normalized correlation difference. The recorded C4 value must remain `0.7236`; rewriting historical evidence to the internal composite would be scientifically incorrect.

### Why `0.6400` and `0.9231 → 0.9615` both exist

The final acceptance evaluator (`RetrainingEvaluator`) uses the authentic development-validation partition and exports positive-class F1 `0.6400` with gain `+0.0114`. The earlier unified validation stage separately trains real-only and augmented models against a 52-row affected-slice evaluation set, obtaining `0.9231` and `0.9615`. The latter stage reports non-affected support zero, confirming it is not an overall/source comparison. Both results belong to the same package execution, but not the same evaluation population or evaluator.

### Augmentation denominator

Reconstructing the seed-42 partitions gives 3,452 source-training rows and 200 authentic adaptation rows. The accepted candidate adds 100 synthetic rows. `RetrainingEvaluationResult.training_rows` is therefore 3,752, and the adapter computes `100/3752=0.0267`. Earlier language describing the denominator as source-only, affected-slice-only, or 3,747 rows is incorrect.

## Artifact verification

- Manifest: `component3/results/final_integration_hardening/scenario_1_real_tvae/pkg-859b8f17c113_manifest.json`.
- Candidate CSV: 100 rows, 20 columns; SHA-256 `6cf32eca7855b50027edbc356216099ffdd1dcb7c9a78cf6141692f9682a0be2`.
- Candidate model: SHA-256 `1e41d1865cfe0467e534ae13cb428a647ff02500c259a6e36588cd70918a149a`.
- Recorded flat C4 contract: preserved in `final_hardening_execution_summary.json`; 12-field schema validated by the existing hardening evidence.
- Generator training configuration in the manifest does not expose every library-level random seed. Those seeds remain **UNVERIFIED**.

## Rejected execution

`RUN-D232CC72A4` is a later real TVAE execution. K=25, 50 and 100 failed the required target utility gain; K=200 failed fidelity (mean KS `0.371 > 0.350`). It exported no approved dataset or model and its C4 status is `REJECTED`. Its zero score fields and `privacy_status=PASS` are schema placeholders resulting from absent accepted-candidate reports and must be interpreted as **NOT EVALUATED / NO ACCEPTED CANDIDATE**.

## Corrected misleading statements

- The approved package's source retention delta is `−0.0137`, not `−0.0080`.
- `0.9231 → 0.9615` must not be presented as the baseline/candidate values behind C4 `utility_score=0.6400`.
- `0.7893` is the internal composite fidelity; `0.7236` is the recorded external flat-C4 fidelity transform.
- `privacy_status=PASS` in rejected or abstained flat output is not measured privacy evidence.
- The decision mode is `RULE_BASED`; “AI decision” is not an accurate description.
- `APPROVED` means approved for human review, never production deployment.
