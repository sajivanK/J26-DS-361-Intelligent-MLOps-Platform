# C4 outer admission stage — installation and execution

This package contains complete code for the **next stage: outer-test repair generation and admission**. It does not contain the downstream outer candidate, decision-freeze, future-outcome or final evaluation pipeline. Those stages remain separately gated and must be implemented and frozen before use. This package does not retrain N1/N2, select policies, alter comparison definitions or enable live deployment.

## Files

All eight files are new. Merge the ZIP's `components` directory into your repository root. No existing source file should be replaced.

- `components/c4/scripts/_outer_group.py`
- `components/c4/scripts/freeze_outer_admission_protocol.py`
- `components/c4/scripts/run_outer_case_admission.py`
- `components/c4/scripts/run_outer_group_admissions.py`
- `components/c4/scripts/run_outer_admission_batch.py`
- `components/c4/scripts/audit_outer_admissions.py`
- `components/c4/tests/unit/test_outer_admission_scope.py`
- `components/c4/OUTER_ADMISSION_RUN.md`

This is an additive package for your existing C4 repository, not a standalone replacement repository. It imports the existing verified preview, preparation, TVAE, repair diagnostics and frozen-artifact helpers.

## Existing verified checkpoint

User-reported full suite: **678 passed in 38.64 seconds**. Commit: `7549982`, pushed to `c4-kesigan`.

- Development evaluation metadata SHA256: `bdf2a8a913e3df147bdc4d238cb9d655dbd949e8786c7e7b7c2e8b9a5e9cafeb`
- Comparison/evaluation readiness SHA256: `840f5723b7ef94853d3359982ba362d39692c5ab15b80ebd329b61989101ff59`
- Frozen controller SHA256: `de09b0220f25b3c82f8849dd2becef4f3301806dfcb1e701d5108b0d252fb4a7`

These checkpoints are checked explicitly. The code never calls development inventory loaders that would reject outer artifacts once admission has started. All original 16 outer groups and 592 cases remain fixed; there is no replacement/early stopping based on admission rates.

## 1. Activate and test (PowerShell, repository root)

```powershell
& ".\.venv\Scripts\Activate.ps1"
$env:C4_DATA_ROOT = '\\?\G:\.shortcut-targets-by-id\1xCM4Ry8l8ads1Xn8rBneZ_AavyLqYC3u\J26-DS-361-Datasets'

python -m pytest components/c4/tests -q
# Continue only when tests pass.
git diff --check
python components/c4/scripts/freeze_outer_admission_protocol.py
```

The last command verifies existing checkpoints and prints the fixed outer inventory without writing a freeze or executing experiments. Stop and inspect any error; do not change frozen hashes to bypass it.

## 2. Commit the complete package

```powershell
git add components/c4/scripts/_outer_group.py `
    components/c4/scripts/freeze_outer_admission_protocol.py `
    components/c4/scripts/run_outer_case_admission.py `
    components/c4/scripts/run_outer_group_admissions.py `
    components/c4/scripts/run_outer_admission_batch.py `
    components/c4/scripts/audit_outer_admissions.py `
    components/c4/tests/unit/test_outer_admission_scope.py `
    components/c4/OUTER_ADMISSION_RUN.md

git commit -m "feat(c4): add frozen outer admission batches and audit"
git push origin c4-kesigan
git status
```

The branch should remain `c4-kesigan`, with a clean working tree.

## 3. Freeze once, then plan and execute batch 1

```powershell
python components/c4/scripts/freeze_outer_admission_protocol.py --freeze
# Continue only if the freeze succeeds. Do not repeat --freeze.
python components/c4/scripts/run_outer_admission_batch.py --batch-number 1
python components/c4/scripts/run_outer_admission_batch.py --batch-number 1 --execute-admissions
```

Each batch has four groups / 148 cases, with at most 76 new TVAE fits plus 72 paired label-corrupted variants. These counts are planned workload, not measured admitted counts. N1 is unchanged. Torch/TVAE imports are checked before new claims are written.

## 4. After batch 1 completes, run batches 2–4 sequentially

```powershell
foreach ($batchNumber in 2..4) {
    Write-Host "Starting outer admission batch $batchNumber"
    python components/c4/scripts/run_outer_admission_batch.py `
        --batch-number $batchNumber `
        --execute-admissions
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Batch $batchNumber failed. Stopping; preserve logs and artifacts."
        break
    }
    Write-Host "Outer admission batch $batchNumber completed."
}
```

Keep the laptop powered and prevent automatic sleep while executing. The loop runs sequentially. Do not launch duplicate batches concurrently.

## 5. Audit after all four batches complete

```powershell
python components/c4/scripts/audit_outer_admissions.py
python components/c4/scripts/audit_outer_admissions.py --save
```

This recomputes the admission screen from saved diagnostics and verifies the original scenario/fold, new authorization, repair/model bytes and record counts. Every one of the 592 admission records remains in the audit, including blocked cases. The audit saves:

```text
C4_DATA_ROOT/c4_artifacts/outer_admission_audits/adult/
  adult-simulation-pp1-v2/outer-all16-admission-audit-v1/
    admissions.csv
    group_summary.csv
    C4_OUTER_ADMISSION_EVIDENCE.md
    metadata.json
    metadata.sha256
```

Admitted and blocked totals are determined by execution; this package does not predict or fabricate them. This is admission evidence only, not measured outer predictive/deployment performance.

## Output locations

Freeze:

```text
c4_artifacts/outer_protocols/adult/adult-simulation-pp1-v2/outer-admission-spec-v1/
```

Batch reports and logs:

```text
c4_artifacts/benchmark_runs/adult/adult-simulation-pp1-v2/
  outer_admission_batches/outer-admission-b01-v1/
  outer_admission_batches/outer-admission-b02-v1/
  outer_admission_batches/outer-admission-b03-v1/
  outer_admission_batches/outer-admission-b04-v1/
```

Per-group/per-case artifacts use the established `group_admissions` and scenario directories. Existing failed attempts and incomplete claims are preserved and rejected. A fresh batch version can reuse fully verified completed group reports; it does not automatically resolve an incomplete group or scenario. Inspect its failure log first. There is no automatic retry or cleanup.

## Verification scope

Local verification: 36 new scope/CLI tests passed, plus syntax compilation. Tests use synthetic manifests and mocked existing checkpoints; no TVAE fits or Windows data/artifact integration were executed locally. The full combined Windows test count and checkpoint verification must come from your run.

The proposed controller remains the frozen `review_only` policy. This package cannot enable AUTO_DEPLOY. Labelled controller fixtures remain separate technical branch checks.

## Next stage

After saving all admission reports, implement/test and freeze the downstream outer candidate + all-method pre-release decision + sealed-future pipeline. Keep forecasts/decisions separate from observed targets. Freeze the required pre-release decisions before future evaluation, retain all assigned groups and apply the existing frozen evaluation/statistical protocol. Do not change N1, N2, N3 or policy parameters based on outer results.
