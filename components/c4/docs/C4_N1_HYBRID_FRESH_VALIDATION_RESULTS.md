# C4 N1 hybrid: fresh scenario-group validation results

Date: 2026-10-20
Experiment: `adult-hybrid-fresh-validation-v1`
Status: completed experimental evaluation; no application-default replacement or production release authorized.

## Main finding

The predeclared hybrid achieved global MAE of 0.103315 percentage points (pp), RMSE of 0.130181 pp and R² of 0.290493. Its additional global improvement over the same-training LightGBM comparator is modest. Slice and complement predictions are identical to that comparator because the hybrid reuses those LightGBM heads. Complement R² is 0.008017, so explained variation remains weak.

All four methods covered every evaluated three-population outcome with their fixed N2 intervals in this run. The intervals are conservative, and every selected policy family remained review-only. This experiment does not establish improved autonomous-release capability or a formal conditional deployment-risk guarantee.

## Question and experimental scope

N1 forecasts future candidate-minus-champion accuracy margins for global performance, the true injected slice, and its complement. It does not directly predict a release class or an individual person's income.

The 75 decision-time input features use pre-release global, diagnosed-slice and diagnosed-complement evidence. Diagnosed populations are not interchangeable with the true injected target populations used by this controlled benchmark.

The revised experiment uses 12,899 eligible processed records from official UCI Adult `adult.test`, bound during external development. The external cohort is shared by development and the new scenario groups. Therefore, the final evaluation is a held-out scenario-group comparison conditional on a shared row pool, not independent-dataset validation. Simulated upstream evidence does not establish live C1/C2/C3 integration.

Original PP1 results used a different outcome-measurement cohort. Subtracting their errors from these errors would combine model and cohort changes. The same-external-training baselines below provide the relevant model comparison; the revised results do not prove that the hybrid alone caused the apparent reduction from the original PP1 error table.

## Fixed model recipe and development

Development used 106 verified training scenarios across 29 nonempty groups. Grouped out-of-fold diagnostics and development comparisons preceded the fresh conformal, policy and outer stages. Development comparison results are selection evidence, not final-test results.

The hybrid recipe was fixed before fresh validation:

- Global: Ridge regression with alpha 100, group-weighted training-only standardization, constant/redundant predictor filtering, and weighted training-residual 0.05/0.95 offsets around its center prediction. These raw global bands are not independently calibrated conditional quantiles.
- True injected slice and complement: the existing fixed LightGBM quantile recipe, with separate 0.05, 0.50 and 0.95 models per target.
- Quantile outputs are sorted and clipped to the valid margin range [-1, 1]; N2 subsequently adjusts interval endpoints without changing the median.
- Comparators: fixed LightGBM, linear quantile and constant-quantile models fitted on the same external-development training table and group weights.
- Each group receives equal total weight for the reported point and pinball metrics. Point errors and R² use the processed median; pinball is evaluated at the same three quantile levels, with the mean reported below.

The hybrid is an adaptation/integration of existing estimators. This study does not establish that combining Ridge and LightGBM is itself a novel algorithm. Its relevant role is inside C4's forecast, joint-uncertainty and governed-release pipeline.

## Frozen execution order and inventory

| Role | Assigned groups | Nonempty admitted groups | Planned cases | Admitted | Blocked |
|---|---:|---:|---:|---:|---:|
| Conformal calibration | 24 | 20 | 888 | 76 | 812 |
| Policy calibration | 12 | 12 | 444 | 48 | 396 |
| Final outer evaluation | 16 | 11 | 592 | 56 | 536 |

All blocked cases and all assigned groups were retained. The five fully blocked outer groups have no candidate future outcomes; they must not be assigned invented coverage or safety labels.

Execution order was: fix fitted N1/comparators; complete conformal admissions/candidates; freeze their forecasts before future outcomes; fit/freeze N2; complete policy admissions/candidates; freeze forecasts before policy outcomes; select/freeze policies; complete all outer admissions; train admitted outer candidates; freeze all outer evidence, forecasts, N2 intervals and selected-policy signals; then evaluate outer future outcomes once. Verified checkpoints supported resume without replacing scenarios or overwriting completed outcomes.

Candidate training reused the original champion configuration and admitted repair input. Pre-release evaluation used the designated release-validation partition. It did not train on release-validation or sealed-future observations.

## N2 and policy rules

N2 used the fixed 0.05 alpha and the replication-group maximum nonconformity score across admitted variants and all three targets. With 20 nonempty calibration groups, the finite rank was 20; the insufficient-support full-range fallback was not used. The resulting endpoint adjustments were:

| Method | N2 adjustment (pp) |
|---|---:|
| Hybrid | 0.6939113141 |
| LightGBM | 0.6939113141 |
| Linear | 0.3903192362 |
| Constant quantiles | 0.6985055231 |

In-sample calibration coverage was 100%; this is not independent validation.

Unchanged safety boundaries, in candidate-minus-champion accuracy pp, were global 0, true injected slice 0, and complement -2. A lower endpoint at or above the boundary is SAFE; an upper endpoint strictly below it is UNSAFE; otherwise the population is UNCERTAIN. Gate failures and unsafe forecasts retain mandatory escalation. Diagnosis stability uses the unchanged minimum 0.50 gate.

The investigator-defined policy design was fixed before policy outcomes: four global buffers and twenty monotone adaptive tuples from 0, 0.1, 0.25 and 0.5 pp. Candidate eligibility required at least three deployment-bearing policy groups and an empirical equal-group conditional unsafe rate no greater than 0.05. These are predeclared design/screening choices, not demonstrated optimal thresholds or a finite-sample deployment-risk bound.

The evidence-context formula is a heuristic, not a calibrated failure probability. In policy calibration, MEDIUM appeared in all 12 nonempty groups; LOW and HIGH appeared in none and remained unsupported. No policy candidate met the frozen selection requirements, so all four methods saved review-only fallbacks for both policy families. Global and adaptive equality under these fallbacks is not evidence that adaptation is generally ineffective.

## Final outer forecast results

All error/pinball values below are accuracy percentage points. Smaller MAE, RMSE and pinball are better. R² is relative to the evaluation-mean squared-error reference; 1 is perfect, 0 matches that reference, and a negative value is worse. No universal R² cutoff establishes suitability for deployment governance.

| Forecaster | Target | MAE (pp) | RMSE (pp) | R² | Mean pinball (pp) |
|---|---|---:|---:|---:|---:|
| Hybrid | Global | 0.103315 | 0.130181 | 0.290493 | 0.027266 |
| Hybrid | True injected slice | 0.147764 | 0.198558 | 0.324359 | 0.041547 |
| Hybrid | True slice complement | 0.104355 | 0.143273 | 0.008017 | 0.028874 |
| Constant quantiles | Global | 0.118050 | 0.154944 | -0.005104 | 0.031034 |
| Constant quantiles | True injected slice | 0.184841 | 0.245459 | -0.032524 | 0.050361 |
| Constant quantiles | True slice complement | 0.119877 | 0.147202 | -0.047142 | 0.029855 |
| LightGBM | Global | 0.107706 | 0.131354 | 0.277643 | 0.029183 |
| LightGBM | True injected slice | 0.147764 | 0.198558 | 0.324359 | 0.041547 |
| LightGBM | True slice complement | 0.104355 | 0.143273 | 0.008017 | 0.028874 |
| Linear | Global | 0.120844 | 0.148839 | 0.072535 | 0.038114 |
| Linear | True injected slice | 0.151867 | 0.196958 | 0.335205 | 0.052009 |
| Linear | True slice complement | 0.124156 | 0.151371 | -0.107293 | 0.036467 |

For global prediction, the hybrid reduced MAE by 4.08%, RMSE by 0.89%, and mean pinball by 6.57% relative to same-training LightGBM. These are descriptive comparisons, not established statistically significant effects. No paired group-level uncertainty analysis or significance test is reported here.

Linear had slightly lower slice RMSE (0.196958 pp versus 0.198558 pp) and higher slice R², while hybrid had lower slice MAE and mean pinball. No method should be called best across every reported metric. Complement R² remains close to zero for hybrid/LightGBM; it is a limitation, not high predictive accuracy.

## Coverage and sharpness

Each method achieved observed scenario joint coverage 56/56 and whole-group joint coverage 11/11 among nonempty admitted outer groups. The nominal N2 target was 95%. The small group count and shared row pool prevent treating observed 100% coverage as a universal guarantee.

Equal-group-weighted mean full interval widths and interval scores at alpha 0.05 were:

| Forecaster | Global width / score (pp) | Slice width / score (pp) | Complement width / score (pp) |
|---|---:|---:|---:|
| Hybrid | 1.69095 | 1.89972 | 1.77075 |
| LightGBM | 1.67314 | 1.89972 | 1.77075 |
| Linear | 1.04530 | 1.22818 | 1.08745 |
| Constant quantiles | 1.83891 | 1.99805 | 1.94573 |

The interval score equals width here because no observed target fell outside its interval; the score's noncoverage penalties are zero. Linear is sharpest in this evaluation. Hybrid does not improve global interval sharpness over LightGBM. High coverage must be assessed together with these widths and the continuing lack of autonomous decisions.

## Final frozen-policy results

Results below apply separately to both global and adaptive families on the 56 admitted outer cases.

| Forecaster | AUTO_DEPLOY | HUMAN_REVIEW | ESCALATE | Equal-group automation | Conditional unsafe deployment rate |
|---|---:|---:|---:|---:|---|
| Hybrid | 0 | 56 | 0 | 0% | Undefined: no deployments |
| LightGBM | 0 | 56 | 0 | 0% | Undefined: no deployments |
| Constant quantiles | 0 | 56 | 0 | 0% | Undefined: no deployments |
| Linear | 0 | 47 | 9 | 0% | Undefined: no deployments |

Linear's nine escalations are forecast-based signals, not proof that precisely those nine candidates were the only actually unsafe cases. Admission-blocked cases are counted separately: 536 blocked escalations plus admitted signals yield 592 planned pipeline outcomes per method/family. Blocked cases were not future-evaluated.

Review-only behavior is the required result under the fixed policy selection. Zero deployments do not demonstrate zero conditional deployment risk, and the synthetic AUTO_DEPLOY branch tests are technical verification rather than empirical automation evidence.

## Implementation and validation evidence

Important files for reproduction and review:

| File or family | Responsibility |
|---|---|
| `scripts/diagnose_n1_development.py` | Training-only diagnostics and baseline comparisons |
| `scripts/experiment_n1_development.py` | Development recipes and grouped metrics |
| `scripts/fit_n1_hybrid_development.py` | Fixed hybrid fitting and JSON/native model serialization |
| `scripts/fit_n1_validation_baselines.py` | Same-training fixed comparator models |
| `scripts/freeze_n1_hybrid_validation.py` | Fresh role assignment and protocol/source bindings |
| `scripts/calibrate_hybrid_fresh_n2.py` | Group-max N2 fitting and persisted calibration |
| `scripts/select_hybrid_fresh_policies.py` | Fixed-grid policy selection and review-only fallback |
| `scripts/_hybrid_outer_execution.py` / `run_hybrid_outer_admissions.py` | Policy-bound outer admission authorization, checkpoints and inventory |
| `scripts/_hybrid_outer_candidate.py` / `run_hybrid_outer_candidates.py` | Admitted candidate training and pre-release evaluation |
| `scripts/_hybrid_outer_forecasts.py` / `run_hybrid_outer_forecast_freeze.py` | Decision evidence, 75 features, N1/N2 forecasts and selected-policy signals before outer outcomes |
| `scripts/_hybrid_outer_future.py` / `run_hybrid_outer_future.py` | Paired future evaluation and fixed forecast/coverage/policy statistics |
| Corresponding `tests/unit/test_*` files | Scope, provenance, numerical, native model I/O and checkpoint/failure checks |

User-reported final local regression: **1,338 tests passed in 118.96 seconds**. `git diff --check` returned without an error; newly untracked files were not staged by that command. The separate final-output audit verified `future_batch.json` against its checksum sidecar and every listed completion output against its saved checksum. Passing software tests and checksums establish tested behavior/integrity; they do not establish strong forecasting performance or scientific generalization.

Checks included data-role and feature alignment, source/model/checksum bindings, save/reload, Windows-safe native byte loading and temporary saves, rejection of unauthorized roles/stages, complete admission retention, no-overwrite/resume behavior, forecast and selected-policy reproducibility before outcomes, paired accuracy arithmetic, group-weighted losses, fixed-N2 coverage, and undefined risk when nobody is deployed.

## Artifact locations and evidence sources

Paths below are relative to `C4_DATA_ROOT`:

- Hybrid: `c4_artifacts/forecasting_experiments/adult/adult-hybrid-development-v1/fit-v1/` (`model.json`, six native LightGBM target/quantile text files, metadata and checksums).
- Fresh validation root: `c4_artifacts/forecasting_experiments/adult/adult-hybrid-fresh-validation-v1/`.
- Within that root: `baseline-fit-v1/`, `n2-calibration-v1/completion-v1/`, `policy-selection-v1/completion-v1/`, and the outer admission/candidate/forecast/future executor directories.
- Final report: `outer-future-evaluator-v1/completion-v1/future_batch.json` plus its checksum sidecar.
- Final tables: `forecast_metrics.csv`, `N2_outer_diagnostics.json`, `selected_policy_decisions.csv`, `policy_group_metrics.csv`, `policy_diagnostics.json`, case index/targets and fixed-N2 interval CSVs.
- Per-case native candidates, decision freezes and paired future predictions: `c4_artifacts/benchmark_runs/adult/adult-hybrid-fresh-validation-v1/<scenario_id>/`.

Frozen validation protocol SHA256: `c6241cfe55680a767ecfd381294be0852bf4fced6b8c6cf2da4f4fa880dcf837`.
Frozen fresh manifest SHA256: `96ffaf2f2b12883a9f7667371f1c5cd9577f951d37e836a8560a59be1b9b5507`.

Numerical evidence for this document comes from the supplied 2026-10-20 admission, candidate, forecast-freeze and final future logs; the subsequent checksum/width inspection; and the user-reported final regression output. Forecast numbers retain six-decimal console precision; width/score numbers retain five-decimal inspection precision. Full precision remains in the checksum-verified artifacts. This document does not replace those artifacts or claim a new independent audit.

## Adoption and reporting decision

Keep the hybrid as the predeclared experimental primary in the report and show every fixed comparator. Retain the current working application/controller until a separately versioned integration is reviewed and verified. No production release is authorized, no original PP1 artifact was replaced by this evaluation, and these experimental results must not silently overwrite the original recorded-case evidence because the outcome cohort differs.

Defensible panel wording: "We completed a fresh scenario-group evaluation of a fixed hybrid forecaster and three same-training baselines. The hybrid gave modest additional global forecasting gains over LightGBM. Joint intervals covered all evaluated outcomes, but conservative uncertainty and unsupported automation evidence led to review-only policies. Complement forecasting and autonomous release remain limitations. Results are conditional on a shared Adult cohort and require independent-cohort or additional-dataset validation."

Remaining implementation work is documentation/source review and, if required for the demo, a clearly labeled research-comparison view that preserves original case replay. Remaining research includes stronger complement prediction, independent data/cohorts, more representative supported deployment/context groups, and joint assessment of coverage, sharpness and automation. Any new tuning must use a new development protocol and new untouched evaluation groups; the completed outer groups must not become a reusable final test for model selection.

No commit or push is performed by creating or extracting this document.
