# Rule precision and generator training remediation — 24 October 2026

## Changes

- **C2 rule serialization:** `c2/src/localization.py::_fmt` now uses round-trip float precision instead of four significant digits. Candidate masks, feature rankings, scoring and diagnosis selection are unchanged. This is the only C2 source change in this follow-up, explicitly authorized by the request to fix rounding.
- **Configurable generator batch sizes:** TVAE and CTGAN expose batch size, with CTGAN packing validation. The integrated defaults are 128 and 100 respectively; actual batch size, epoch count and training row count are recorded by the selector.
- **Training duration:** the local Adult configuration and onboarding example use 200 epochs rather than 20. This is a training change, not a quality-threshold change. The authentic adaptation-label budget remains 200.
- **Discrete numeric metadata:** Adult's age, education count, capital amounts and weekly hours have discrete support even after numerical preprocessing. They can now be explicitly declared in `generator_discrete_numeric_features`. Generators preserve those numeric values rather than synthesizing unrestricted decimal jitter. Output representation, model schema, C2 slice coordinates and the original numerical fidelity/privacy checks remain unchanged. Other datasets keep their own explicit schema.
- **Bounded privacy sampling:** Adult requests one pool of up to three times the repair quantity from one fitted generator. C3 discards records whose training-data distance is below the existing .05 minimum, removes duplicated feature rows and takes the first required eligible records. If too few remain, it rejects the run. It does not perturb copied records, inspect evaluation labels, search seeds, retrain repeatedly or relax a threshold.
- **Full revalidation:** the retained candidate still undergoes structure, target coverage, fidelity, privacy and independent utility validation. Aggregate privacy failures from the raw pool remain in provenance; they are not hidden by the final PASS.
- **UI:** training batch size, numeric generation metadata, raw pool count, retained candidate count and discarded too-close count are visible. A `?run=<C3 run ID>` link opens the exact saved repair directly.

## Controlled experiment

Input: saved real Drift Lab batch `lab_70628f7169`, 2,000 Adult rows, features `relationship`, `age`, `sex`, `native_country`, strength .8, seed 42. The experiment replays the batch's saved **real C1 output**, obtains a **fresh real C2 diagnosis** using the same model and runs the authoritative **real C3 service**. It is not a new C1 detection run or a real C4 release.

The corrected rule is:

```text
relationship == 2 and education_num > -0.4179650267872039
```

Both C2 and C3 now select **877 / 2,000 = .4385**. The previous shortened threshold selected 1,412 rows instead.

Protocol: evaluate the old 20-epoch/default-batch generator on quality/privacy only; evaluate the declared 200-epoch, discrete-aware candidate; after its privacy failure, add training-only bounded privacy sampling and evaluate the resulting candidate once on the held-out utility observations. The first candidate failed privacy before any utility evaluation, so that failure did not expose utility evidence for tuning. The second experiment reuses the recorded baseline instead of training it again. Artifact hashes and the exact C2 diagnosis are checked before baseline reuse.

| Measurement | Old training | Improved training | Improved training + privacy sampling |
|---|---:|---:|---:|
| CTGAN epochs | 20 | 200 | 200 |
| Training batch size | 500 | 100 | 100 |
| Actual adaptation slice rows | 200 | 200 | 200 |
| Candidate rows | 60 | 60 | 60 retained from 180 |
| Mean KS, maximum .35 | .4650 — FAIL | .0827 — PASS | .0793 — PASS |
| DCR p05, minimum .05 | .1052 — PASS | .0326 — FAIL | .0628 — PASS |
| Final near-duplicate ratio | 0 | 0 | 0 |
| Final target coverage | Not the baseline acceptance test | 1.0 | 1.0 |
| Utility evaluation | Not performed | Skipped after privacy failure | Performed on independent rows |

The final pool contained 180 generated records. Screening discarded **28** records that were too close to real training data, leaving 152 eligible records; the first 60 were retained. Remaining eligible records were unused. No exact copies were found in the raw pool. Its DCR/near-duplicate failures remain recorded in `privacy_sampling.raw_pool_privacy_report`.

## Final utility result

Run: `c3-3e6485104f314297865c5f4d580a61e5`.

| Model | Affected-slice accuracy |
|---|---:|
| Original monitored model | .833333 |
| Real-only retraining control | .865741 |
| Synthetic-assisted candidate | .865741 |

Gain over original: **+.032407**, or **+3.24 percentage points**. Gain over real-only control: **0**. Paired bootstrap 95% interval for gain over original: **[-.016319, .076505]**, which includes zero.

Source accuracy changed from .888466 to .871857, within the unchanged .05 retention tolerance. Overall current accuracy changed from .864 to .882.

**Final decision: REJECTED.** Structure, coverage, fidelity and privacy now pass. The statistical-confidence gate correctly rejects the remaining uncertainty. These observations also do not demonstrate incremental benefit from synthetic data over the real-only control. No approved artifact download is exposed. No successful approved drift preset is claimed.

## Verification and artifacts

- Focused precision/generator/service checks: **74 passed**, 3 slow generator cases excluded.
- Follow-up privacy screening and privacy validator checks: **23 passed**; these overlap with the precision checks and should not be summed as distinct tests.
- Frontend production build passed. Runtime API returns the saved final result with PASS fidelity/privacy and an explicit statistical rejection.
- No full historical benchmark suite, generator sweep or seed search was run.
- Models, existing datasets, C1 algorithms and shared contract schemas were preserved. C2 diagnosis algorithms were preserved; only executable threshold formatting changed.

Evidence:

- `component3/results/training_remediation/controlled_study.json`
- `component3/results/training_remediation/privacy_screened_study.json`
- Persisted C3 package, progress and result in the configured C3 repair storage.

Open the exact result:

`http://127.0.0.1:5173/c3?run=c3-3e6485104f314297865c5f4d580a61e5`

The running backend was restarted with the existing isolated pinned environment. No existing model artifact was overwritten. Real C4 remains unimplemented.
