# C4 PP1 final outer-results review — 17 October 2026

## Verified status

The supplied result archive contains 45 files. ZIP integrity passed. All 17 files pinned by report metadata and all 18 figure files pinned by figure metadata matched their SHA256 values. Figure metadata binds the same report metadata hash. This validates the supplied bundle; it is not a fresh rerun of the original model/data artifacts.
Latest supplied terminal evidence: 797 tests passed; reporting commit `2da0e48` pushed on `c4-kesigan`; working tree clean.
Completed: N1 fitting, original and extended N2 calibration, N3, measured global/N4 policy selection, controller acceptance tests, comparisons/ablations, all outer execution, recovered snapshot, final statistics and six figures in PNG/PDF/SVG. API/frontend/real upstream integration remain separate pending work.

## Experimental scope

Adult accuracy-margin pilot with simulated upstream C1/C2/C3 evidence. There are 16 assigned outer groups, 14 nonempty groups, 56 admitted target rows, 536 blocked cases and 592 admission records. Blocked targets were not imputed. All 16 assigned groups are retained in bootstrap resampling.
The primary safety oracle requires future global and true injected slice candidate-minus-champion accuracy margins >= 0 and true slice complement margin >= -0.02. Diagnosed slice evidence is distinct from true injected slice research ground truth.

## Governance results

| Method | AUTO | HUMAN_REVIEW | ESCALATE | Unsafe AUTO | Equal-group automation | Illustrative cost |
|---|---:|---:|---:|---:|---:|---:|
| PROPOSED | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| B1 | 16 | 0 | 40 | 16 | 32.74% | 3.309524 |
| B2 | 10 | 0 | 46 | 10 | 19.94% | 2.029762 |
| B3 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| B4 | 0 | 0 | 56 | 0 | 0.00% | 0.035714 |
| B5 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| B6 | 56 | 0 | 0 | 55 | 100.00% | 9.821429 |
| B7 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| B8 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| A1 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| A2 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| A3 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| A4 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| A5 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| A6 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| A7 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |
| A8 | 0 | 56 | 0 | 0 | 0.00% | 1.000000 |

55/56 admitted cases fail at least one frozen safety boundary; one case is safe by the observed point-outcome oracle. This does not imply that the one safe case had enough pre-release evidence for deployment.
Proposed C4 reviews all 56 cases: zero unsafe deployments is achieved with zero deployment automation. Conditional deployment risk is undefined, not zero. Its outcome matches review-only comparators. No autonomous-deployment benefit or adaptive-policy superiority is demonstrated.
B1 deploys 16 unsafe cases; B2 deploys 10 unsafe cases; B6 deploys 55 unsafe and one safe case. These observations show risks of these comparators on this pilot. They do not by themselves prove the proposed method outperforms all alternatives.
B4 escalates all 56 cases and has illustrative cost 0.035714 versus 1.0 for Proposed. The proposed method is not the cost winner under the frozen cost definition. Costs are investigator-defined, not measured deployment/business costs.

## Forecast errors

Equal-group-weighted MAE, converted from accuracy-margin fractions to accuracy percentage points.

| Forecaster | Global MAE (pp) | True slice MAE (pp) | Complement MAE (pp) |
|---|---:|---:|---:|
| ORIGINAL | 0.2656 | 0.4552 | 0.2877 |
| linear | 0.2543 | 0.4643 | 0.3369 |
| A1 | 0.2747 | 0.4969 | 0.2884 |
| A2 | 0.2720 | 0.4562 | 0.2914 |
| A3 | 0.2707 | 0.4400 | 0.2849 |

Linear has lower global MAE than original LightGBM; original LightGBM has lower slice and complement MAE than linear. Linear also has lower slice RMSE/higher slice R2. Thus there is no uniform LightGBM win across targets and metrics.
Original LightGBM equal-group R2 is 0.0970 global, 0.1985 slice and -0.1382 complement. Complement point prediction is weak under this metric. Negative R2 and near-zero target scales must be reported alongside small MAEs.
A3 (without four C3 quality predictors in N1) has lower slice/complement MAE than full N1 on these outer groups. Do not claim every feature family improves every outcome. These forecast differences are descriptive; the saved paired governance bootstrap is not a test of MAE differences.

## N2 coverage and interval widths

| Interval variant | Scenario joint coverage | Whole-group joint coverage | Covered groups | Full-bound fallback |
|---|---:|---:|---:|---|
| PROPOSED | 94.64% | 85.71% | 12/14 | False |
| A1 | 94.64% | 85.71% | 12/14 | False |
| A2 | 94.64% | 85.71% | 12/14 | False |
| A3 | 94.64% | 85.71% | 12/14 | False |
| A4 | 92.86% | 78.57% | 11/14 | False |
| A5 | 96.43% | 92.86% | 13/14 | False |
| B5 | 100.00% | 100.00% | 14/14 | True |
| A6 | 100.00% | 100.00% | 14/14 | True |

Primary scenario joint containment is 53/56 (94.64%). Primary whole-group joint containment is 12/14 (85.71%). Because the calibration unit is group-max, the latter is an essential diagnostic. This pilot does not empirically establish the nominal 95% whole-group coverage. With only 14 nonempty outer groups, the observed proportion is coarse; this result alone is neither proof nor disproof of a theoretical exchangeability-based coverage claim. Exchangeability/generalization remain assumptions requiring separate justification.
Equal-group target coverage: global 94.64%, true slice 92.86%, complement 100%. Primary equal-group average widths are 2.1656 pp global, 2.4224 pp true slice, 2.4261 pp complement.
A5 has 13/14 (92.86%) whole-group containment with wider intervals. B5/A6 contain all observations because their unsupported Bonferroni calibration uses full [-1,1] bounds (width 2 margin units = 200 pp); that is trivial containment, not useful uncertainty precision. A4 reports two targets and cannot be treated as a three-target joint comparison.
Preserve these results. Do not recalibrate, lower safety boundaries, switch the primary model or choose a variant based on the observed outer results.

## Statistical interpretation

2,000 paired assigned-group draws; seed 20261016; all 16 assigned groups included. The intervals are exploratory conditional-simulation percentile intervals, not multiplicity-controlled significance tests or evidence of independent-dataset generalization.
Primary automation and unsafe AUTO frequency have [0,0] bootstrap intervals because every primary case is reviewed. Conditional risk is undefined in all 2,000 draws. These degenerate intervals do not establish operational safety beyond abstention.
Proposed-minus-B1 equal-group unsafe AUTO frequency: -0.32738, exploratory 95% interval [-0.47222,-0.19226]. Proposed-minus-B2: -0.19940, interval [-0.35278,-0.07551]. The corresponding automation differences are equally negative: unsafe deployment avoidance is coupled to lost automation.
Only one observed safe case limits safe-automation assessment. The primary safe-automation-coverage bootstrap is undefined in 701/2000 draws, so its confidence interval is correctly null.

## Defensible PP1 wording

> The Adult accuracy-margin pilot completed a frozen end-to-end post-repair governance evaluation. On 56 admitted outer scenarios across 14 nonempty groups, the primary measured policy abstained on all cases. It avoided unsafe automated releases but did not demonstrate deployment automation or superiority to review-only alternatives. Joint interval containment was 94.64% at scenario level and 85.71% at whole-group level. Technical decision branches were verified separately using labelled acceptance fixtures.

## Next actions

1. Preserve the frozen experiment and this measured result set. No more batches or outcome-driven tuning are needed for this pilot.
2. Append this review and exact report/figure hashes to the master thesis evidence; retain prior failure/recovery evidence.
3. Prepare the standalone demo: one real measured case showing review plus labelled AUTO_DEPLOY/HUMAN_REVIEW/ESCALATE acceptance fixtures. The fixture AUTO uses its explicitly eligible test policy and must never be presented as the measured review-only policy deploying.
4. Finalize PP1 findings, limitations and viva explanations. Record weak group coverage, zero automation, limited safe-case support and mixed forecasting/ablation results explicitly.
5. Build API/frontend and real C1-C3 integration only under the agreed leader guidance. Keep simulated upstream declarations until real outputs are connected and checked.
6. Any stronger repair experiment, new dataset, new thresholds or diagnosed-slice/recovery target belongs to a separately versioned future study.

## Exact evidence hashes

- Report metadata SHA256: `ef5a6ecd9d4623258207d0735d3935a4941924f210013818f7273155ea5f805e`
- Figure metadata SHA256: `f251900934ab985cdc152e77910112b9c26fe3655d49f98951bf96e9976af1be`
- Recovered outer snapshot metadata SHA256: `f7e7c41e1a7e54e509dfd7d70069359b73bbf3e1a44a08875b0b09313d8ff895`
- Reporting commit: `2da0e48bf27bd1966305cd397839811ff08d691f`
