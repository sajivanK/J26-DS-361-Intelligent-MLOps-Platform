# Final outer evaluation and PP1 figures

## Verified starting checkpoint

Your recovery transcript reports:

- **784 tests passed** in 95.97 seconds.
- Recovery committed/pushed as `7865339` on `c4-kesigan`.
- All 56 outer cases, paired targets and original bindings passed verification.
- Recovered file checksums and original preservation passed.
- Recovered snapshot: `outer-all16-outcome-audit-recovered-v2`.
- Recovered metadata SHA256: `f7e7c41e1a7e54e509dfd7d70069359b73bbf3e1a44a08875b0b09313d8ff895`.

The snapshot save problem is resolved. Preserve both snapshot versions.

## Add these five new files

```text
components/c4/scripts/_outer_evaluation_inputs.py
components/c4/scripts/evaluate_outer_comparisons.py
components/c4/scripts/export_outer_result_plots.py
components/c4/tests/unit/test_outer_evaluation_reporting.py
components/c4/OUTER_FINAL_EVALUATION_RUN.md
```

No existing frozen code is replaced. The evaluator uses the already frozen metric,
comparison and bootstrap implementations. The new loader binds the recovered
outer snapshot to the pre-future decisions and existing comparison runtime.
It does not use the old development-only loader, which correctly rejects outer artifacts.

Local verification: **13 new tests passed** on synthetic known-value fixtures.
The tests check all 17 comparison paths, frozen decision replay, empty-group
accounting, undefined conditional risks, fallback semantics, report checksums and
figure rendering. Synthetic test results are not experimental findings.
The actual full-repository tests and outer report computation run on your machine.

## 1. Test and commit

Extract at the repository root. Keep the existing Python environment and folder layout.

```powershell
python -m pytest components/c4/tests -q
git diff --check

git add components/c4/scripts/_outer_evaluation_inputs.py `
    components/c4/scripts/evaluate_outer_comparisons.py `
    components/c4/scripts/export_outer_result_plots.py `
    components/c4/tests/unit/test_outer_evaluation_reporting.py `
    components/c4/OUTER_FINAL_EVALUATION_RUN.md

git commit -m "feat(c4): report final outer comparisons statistics and figures"
git push origin c4-kesigan
git status
```

Only proceed after tests pass and the reporting code is committed.

## 2. Calculate and save final outer statistics

```powershell
python components/c4/scripts/evaluate_outer_comparisons.py
python components/c4/scripts/evaluate_outer_comparisons.py --save
```

Both invocations calculate metrics from saved results; neither reruns candidate
training, repair generation, forecasting-model prediction, calibration or policy
selection. This is not another long admission/candidate batch.

No new experimental specification freeze is needed: the statistical definitions
were frozen before outer execution. The report records hashes of this additive
reporting code and its exact input snapshot for reproducibility.

Checks include:

- All 56 admitted cases and all 17 methods, aligned to frozen scenario IDs.
- All 592 admission records, including 536 blocked cases.
- All 16 assigned groups, including both all-blocked groups.
- Replay of every saved decision from its pre-release features and fixed forecasts.
- MAE/RMSE/R2 and pinball loss for original LightGBM, linear and A1-A3 forecasts.
- Per-target, simultaneous and whole-group coverage; interval widths/scores.
- Automation, unsafe AUTO frequency, conditional unsafe rate, review burden and costs.
- All 2,000 shared paired assigned-group bootstrap draws, seed 20261016.
- Per-group/namespace/mechanism/strength/diagnosis/repair support and diagnostics.
- Paired PROPOSED-minus-comparator estimates and exploratory percentile intervals.

If any binding/checksum fails, preserve the error. Do not alter pinned hashes or
rerun the original outer experiment to obtain a different result.

Output:
`c4_artifacts/evaluations/adult/adult-simulation-pp1-v2/outer-comparison-evaluation-v1`.

Saved evidence:

```text
governance.json
forecast.json
interval.json
diagnostic_risk_coverage.json
statistics.json
governance_summary.csv
group_counts.csv
admission_group_summary.csv
subgroup_summary.csv
forecast_summary.csv
pinball_summary.csv
interval_summary.csv
diagnostic_risk_coverage.csv
bootstrap_intervals.csv
paired_differences.csv
cost_sensitivity.csv
C4_OUTER_EVALUATION_EVIDENCE.md
metadata.json
```

JSON documents have checksum sidecars; metadata pins every report/CSV/Markdown byte.
Markdown is written as bytes to avoid the Windows newline fault.

## 3. Export figures

Check that the existing environment has Matplotlib:

```powershell
python -c "import matplotlib; print(matplotlib.__version__)"
```

If it is missing, add the plotting dependency without upgrading the model stack:

```powershell
python -m pip install "matplotlib==3.10.8"
python -m pip check
```

Then run:

```powershell
python components/c4/scripts/export_outer_result_plots.py
python components/c4/scripts/export_outer_result_plots.py --save
```

Six figures are saved as PNG, PDF and SVG under `outer-comparison-evaluation-v1/figures-v1`:

1. `outer_decision_distribution`: admitted-case decision counts/shares.
2. `outer_group_governance`: equal-group automation, unsafe AUTO frequency and review burden.
3. `outer_conditional_risk`: undefined values labelled explicitly.
4. `outer_forecast_mae`: five forecasting variants, accuracy percentage points.
5. `outer_joint_coverage`: three-target whole-group coverage; A4 excluded; full-bound fallbacks labelled.
6. `outer_diagnostic_risk_coverage`: forced score-prefix curves, explicitly non-operational.

The plotting module verifies saved report hashes. It neither evaluates a model nor
changes decisions. Its metadata binds all 18 figure files to the exact report.

## 4. Interpret the measured results

Review the method summaries, primary/paired bootstrap intervals, coverage and
forecast errors before making thesis claims. Keep these distinctions explicit:

- This is held-out group evaluation within the Adult simulated-upstream pilot,
  not validation on a new independent dataset or real C1/C2/C3 production outputs.
- Governance outcome metrics use admitted cases; blocked cases have no imputed labels.
- Zero AUTO means conditional deployment risk is undefined, not proven zero.
- A review-only primary policy cannot demonstrate an autonomous-deployment benefit
  or adaptive-policy superiority.
- Equivalent/abstaining ablations may still differ in forecast or interval metrics.
- B5/A6 full-bound coverage is trivial containment; A4 reports only two targets.
- Costs are frozen illustrative design choices; risk-coverage curves are diagnostics.
- Bootstrap intervals are exploratory; no multiplicity-controlled significance or
  formal conditional-risk guarantee is claimed.
- All three technical branches remain demonstrated separately by labelled controller fixtures.

## 5. Following tasks

After successful reports/figures:

1. Inspect actual outer tables/plots and write the final results and limitations.
2. Append the new checkpoints, tests, commits, hashes and measured results to thesis evidence.
3. Assemble the standalone C4 PP1 demo: a real benchmark case plus labelled branch fixtures.
4. Continue API/frontend and C1-C3 integration only under the leader guidance already agreed.

Do not regenerate scenarios, refit N1/N2, adjust thresholds or change policies in
response to outer-test results. Preserve the primary review-only result exactly.
