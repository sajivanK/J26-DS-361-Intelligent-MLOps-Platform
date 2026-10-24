# Outer admission verification scheduling revision

## Observed issue and preserved evidence

The user reported 714 tests passing in 38.64 seconds, commit `7f944c7`, and original admission freeze `d8815e1cc433daa45963ede0f9075e7d2ec4038dd2b9f6785cf38cd7ffa859b9`.

The first outer group `adult-ns42-r002` has 37 completed admissions: **2 admitted / 35 blocked**. Its completed group report SHA256 is `bbe33eae2aad1c681b1675a30dbf38e3542bc676f8f0d198b585323e880a490e`. No group failure was recorded. At the supplied diagnostic snapshot, the batch final report did not yet exist. These are admission results; no outer candidate/future performance results have been reported.

The original batch verifies each completed record through a helper that also repeats the complete shared development-checkpoint verification. That includes repeated reads of frozen models/comparison artifacts and policy support dependencies. This is unnecessary repeated work for every record and can be costly at the external G: path. The logs and parent-process activity are consistent with this delay; no Windows stack trace was obtained, so the exact live instruction was not measured.

## Revision behavior

This package is additive. It does not replace any previously frozen source file, freeze, admission record, group report or failed attempt.

- Full shared checkpoint verification occurs before and after each group.
- Every record still checks its original manifest row, original and new freeze links, expected N1 binding, saved repair metadata, repair CSV and generator checksums, corruption-mask checksum where applicable, privacy diagnostic links, recomputed frozen screen and admission status.
- The original `run_outer_case_admission.py` still generates every new case. Its own full entry checks remain unchanged. TVAE fitting time is unchanged.
- Completed group reports are reused after verification. No completed repair is generated again.
- An incomplete scenario/group claim is still rejected and preserved. This revision is not a recovery bypass for partially written artifacts.
- Case progress is streamed to the console and copied into the new batch log.
- The scheduling revision is separately committed/hashed/frozen. It does not alter model/policy choices, thresholds, seeds or outer group assignments.
- No persistent verification cache or stat-only trust cache is introduced.

## Six new files

```text
components/c4/scripts/_outer_admission_verification.py
components/c4/scripts/run_outer_group_admissions_portable.py
components/c4/scripts/run_outer_admission_batch_portable.py
components/c4/scripts/audit_outer_admissions_portable.py
components/c4/tests/unit/test_outer_portable_verification.py
components/c4/OUTER_ADMISSION_PORTABLE_RUN.md
```

Merge the ZIP's components folder into your repository. All files above are new; keep the eight earlier outer-admission files intact.

## 1. Stop only a parent that is verifying the completed group

Do not run the old and new batch runners concurrently.

In the second PowerShell terminal inspect the current Python commands:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_outer_(admission_batch|group_admissions|case_admission)\.py' } |
    Select-Object ProcessId, ParentProcessId, CommandLine |
    Format-List
```

If only `run_outer_admission_batch.py` is present, with no group/case runner, and the first group has its complete report, the parent is between groups. Press **Ctrl+C once in the original batch terminal**. Preserve its claim, logs and all completed group/scenario files. A KeyboardInterrupt at this boundary does not require regenerating the completed group.

If a `run_outer_group_admissions.py` or `run_outer_case_admission.py` child is active, the old batch may already have advanced. Let that group finish; do not interrupt an active case merely to install this revision. You can use the portable runner for later batches after the current old batch finishes.

## 2. Test and commit (repository root, .venv and C4_DATA_ROOT already set)

```powershell
python -m pytest components/c4/tests -q
git diff --check
```

Continue only after success:

```powershell
git add components/c4/scripts/_outer_admission_verification.py `
    components/c4/scripts/run_outer_group_admissions_portable.py `
    components/c4/scripts/run_outer_admission_batch_portable.py `
    components/c4/scripts/audit_outer_admissions_portable.py `
    components/c4/tests/unit/test_outer_portable_verification.py `
    components/c4/OUTER_ADMISSION_PORTABLE_RUN.md

git commit -m "fix(c4): avoid repeated shared verification per outer admission record"
git push origin c4-kesigan
git status
```

## 3. Freeze this scheduling revision once

```powershell
python components/c4/scripts/run_outer_admission_batch_portable.py --freeze-spec
```

This performs the existing complete checkpoint verification before saving the new scheduling freeze. The first verification can still take time; the script prints an explicit verification message. Do not repeat --freeze-spec after success.

## 4. Resume batch 1 in a new attempt

Use only after the old batch runner and all its children have stopped:

```powershell
python components/c4/scripts/run_outer_admission_batch_portable.py `
    --batch-number 1 `
    --resume-from outer-admission-b01-v1 `
    --batch-version outer-admission-b01-resume-v2
```

After the plan succeeds:

```powershell
python components/c4/scripts/run_outer_admission_batch_portable.py `
    --batch-number 1 `
    --resume-from outer-admission-b01-v1 `
    --batch-version outer-admission-b01-resume-v2 `
    --execute-admissions
```

The first completed group should print `REUSING COMPLETE GROUP WITHOUT TVAE REFIT`, verify all 37 local records with progress, and proceed to the next fixed group. Shared artifacts are verified at each group boundary. No old attempt is edited.

If the old batch finished all four groups normally, do not execute this resume command unnecessarily; proceed with batches 2–4 using the portable runner.

## 5. Remaining batches, sequentially after batch 1 succeeds

```powershell
foreach ($batchNumber in 2..4) {
    python components/c4/scripts/run_outer_admission_batch_portable.py `
        --batch-number $batchNumber `
        --execute-admissions
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Batch $batchNumber failed; preserve all artifacts and stop."
        break
    }
}
```

The attempt defaults are `outer-admission-b02-portable-v2`, `outer-admission-b03-portable-v2` and `outer-admission-b04-portable-v2`. Existing completed groups are reused even if produced by the old runner. Failed attempts are preserved; use a fresh attempt name only after diagnosing an error.

## 6. Use the portable audit after all 16 groups are complete

```powershell
python components/c4/scripts/audit_outer_admissions_portable.py
python components/c4/scripts/audit_outer_admissions_portable.py --save
```

It saves CSV/JSON/Markdown evidence in:

```text
C4_DATA_ROOT/c4_artifacts/outer_admission_audits/adult/
adult-simulation-pp1-v2/outer-all16-admission-audit-portable-v2/
```

The original audit script remains intact; use this portable audit to avoid its repeated whole-checkpoint checks per case.

## Local verification and limitations

16 new tests passed. Combined with the 36 original outer-admission scope tests: **52 passed in 0.45 seconds** locally. Tests cover record tampering, screen recomputation, model-binding changes, preserved incomplete claims, revision hash changes, completed-group reuse without subprocess training, fixed resume identity and shared verification call counts. Tests use synthetic artifacts/mocked development checkpoints. No benchmark was rerun or timed on the user's Windows/G: environment. No numerical speedup is claimed.

The later outer candidate/decision/future harness remains pending and separately gated. The original review-only controller remains unchanged.
