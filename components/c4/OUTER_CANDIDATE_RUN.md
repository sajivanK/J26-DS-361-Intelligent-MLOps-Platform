# C4 outer candidate, decision-freeze and outcome pipeline

Keep `components/c4/` and the existing Python 3.12 virtual environment.
This package adds eight scripts and one test file. It replaces no frozen files.

## Checkpoint and scope

The package expects the completed outer admission audit:

- 16 assigned groups; 592 admission records.
- 56 admitted cases in 14 groups; 536 blocked cases; two all-blocked groups.
- Admission audit metadata SHA256: `c4003cc8331495a8411ab0f9440bd794c7d1da9da52351b7be5623f11de8560b`.
- Audit directory: `c4_artifacts/outer_admission_audits/adult/adult-simulation-pp1-v2/outer-all16-admission-audit-portable-v2`.

These counts and the hash are checkpoint requirements, not newly measured outputs
from this package. If a binding check fails, preserve the evidence and inspect the
mismatch; do not change the expected hash to bypass it.

The runner reuses all existing repair packages. It trains 56 XGBoost candidate
models using the frozen reference-plus-repair design, then evaluates their
pre-release evidence. It does not refit TVAE, N1, N2 or selected policies.

The primary measured controller remains review-only. Comparator AUTO_DEPLOY
signals, if any, are research comparisons and do not authorize deployment.

## Files to add

```text
components/c4/scripts/_outer_pipeline.py
components/c4/scripts/freeze_outer_execution_protocol.py
components/c4/scripts/validate_outer_candidate.py
components/c4/scripts/freeze_outer_case.py
components/c4/scripts/freeze_outer_pre_release_registry.py
components/c4/scripts/evaluate_outer_case_future.py
components/c4/scripts/run_outer_candidate_batch.py
components/c4/scripts/audit_outer_snapshot.py
components/c4/tests/unit/test_outer_execution_boundaries.py
components/c4/OUTER_CANDIDATE_RUN.md
```

## 1. Extract and verify

Extract the ZIP in the repository root. Its `components/c4/` directory merges
with the existing directory. These filenames must be new in your checkout.

```powershell
python -m pytest components/c4/tests -q
git diff --check
python components/c4/scripts/freeze_outer_execution_protocol.py
```

Local package validation: **41 new tests passed**, including admission rejection,
whole-fold future isolation, execution order, partial-claim preservation and
paired-target recomputation. The full repository and Adult artifacts are not
available in the package-author environment; your full suite and checkpoint plan
are required before execution. Do not report a predicted full-suite test count.

## 2. Commit and push the additive implementation

```powershell
git add components/c4/scripts/_outer_pipeline.py `
    components/c4/scripts/freeze_outer_execution_protocol.py `
    components/c4/scripts/validate_outer_candidate.py `
    components/c4/scripts/freeze_outer_case.py `
    components/c4/scripts/freeze_outer_pre_release_registry.py `
    components/c4/scripts/evaluate_outer_case_future.py `
    components/c4/scripts/run_outer_candidate_batch.py `
    components/c4/scripts/audit_outer_snapshot.py `
    components/c4/tests/unit/test_outer_execution_boundaries.py `
    components/c4/OUTER_CANDIDATE_RUN.md

git commit -m "feat(c4): add frozen outer candidate decision and future pipeline"
git push origin c4-kesigan
git status
```

Stay on `c4-kesigan`. Freeze requires a clean working tree.

## 3. Freeze downstream execution separately

```powershell
python components/c4/scripts/freeze_outer_execution_protocol.py --freeze
```

This creates `outer_protocols/adult/adult-simulation-pp1-v2/outer-execution-spec-v1`.
It preserves the original admission-only freeze and its historical authorization.
Repeated invocation verifies/reuses an already complete execution freeze.

## 4. Plan, then execute once

```powershell
python components/c4/scripts/run_outer_candidate_batch.py --all-groups

python components/c4/scripts/run_outer_candidate_batch.py `
    --all-groups `
    --execute
```

The runner prints every scenario and stage and writes stage logs. It runs:

1. Candidate validation for every admitted case.
2. All 17 frozen method decisions for every admitted case.
3. An immutable registry covering all 56 case freezes.
4. Sealed-future evaluation for all 56 cases.

**No outer future evaluator can execute before the whole-fold registry exists.**
The existing policies and their context-support counts remain fixed from the
policy fold; they are not estimated again using outer evidence.

Full shared checkpoint verification occurs before/after each phase. Each case
also verifies its own dependencies. This avoids the previous admission runner's
repeated full-global verification for every record.

Expected completion counts, not predicted research performance:

```text
COMPLETE: 56 OUTER FUTURES; 536 BLOCKED RETAINED; 16 GROUPS
```

The batch report is saved under:
`benchmark_runs/adult/adult-simulation-pp1-v2/outer_candidate_batches/outer-candidate-all16-v1/batch_results.json`.

Keep the laptop on mains power and prevent sleep for this run. Do not launch the
same runner concurrently. Runtime depends on this machine and storage; no fixed
duration is promised.

## 5. Audit and save the outer snapshot

After a successful batch:

```powershell
python components/c4/scripts/audit_outer_snapshot.py
python components/c4/scripts/audit_outer_snapshot.py --save
```

This verifies reconstructed decision-time features and all frozen method decisions,
then recomputes the three target margins from saved paired future predictions.
It does not rerun future model prediction or change a decision.

Output: `meta_datasets/adult/adult-simulation-pp1-v2/outer-all16-outcome-audit-v1`.

Saved files:

- `features.csv`, `targets.csv`, `scenario_index.csv`.
- `decisions.csv`: 56 scenarios x 17 methods = 952 saved method decisions.
- `forecasts_ORIGINAL.csv`, `forecasts_A1.csv`, `forecasts_A2.csv`, `forecasts_A3.csv`, `forecasts_linear.csv`.
- `admissions.csv`: all 592 records, including blocked cases.
- `group_summary.csv`: all 16 groups, including both all-blocked groups.
- `C4_OUTER_SNAPSHOT_EVIDENCE.md`, `metadata.json` and checksum sidecar.

Report accuracy margins in fractions. True injected slice is the primary research
target; the diagnosed slice remains operational pre-release evidence. This remains
an Adult pilot with simulated upstream producers and no live release.

## Failure or interruption

Stage logs stream to the console and are saved in the attempt directory. A failure
report records the scenario and phase. Do not delete or overwrite a claim.

A fresh batch version can reuse **fully completed, verified** stages:

```powershell
python components/c4/scripts/run_outer_candidate_batch.py `
    --all-groups `
    --batch-version outer-candidate-all16-resume-v2 `
    --execute
```

If a stage directory exists without its completion marker, the runner stops.
Inspect that stage's log and claim before implementing a specific recovery. A fresh
batch version does not bypass an incomplete candidate/freeze/future claim. Once any
outer future claim exists, new candidate or decision freezes are forbidden.

## Following work

After the saved outer audit: connect the existing evaluation engine to this frozen
outer snapshot; produce final B1-B8/A1-A8 metrics, group bootstrap statistics,
tables/plots and thesis evidence. Keep all definitions, models and thresholds fixed
regardless of the observed result. API, frontend and integration remain deferred
until leader guidance. This package does not contain the final outer comparison
report generator or claim that PP1 research evaluation is already finished.
