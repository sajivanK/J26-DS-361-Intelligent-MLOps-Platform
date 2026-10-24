# Recover the outer snapshot Markdown save

The outer experiment completed: 56 candidates, 56 pre-release decision freezes,
whole-fold registry, and 56 future evaluations. The original save failed only
when checking `C4_OUTER_SNAPSHOT_EVIDENCE.md` after metadata was written.

Cause: `write_text()` translated LF to CRLF on Windows, whereas the stored
checksum was calculated from the LF string. This recovery accepts exactly that
byte transformation, never an arbitrary checksum mismatch.

## Add these three new files

```text
components/c4/scripts/recover_outer_snapshot_save.py
components/c4/tests/unit/test_outer_snapshot_save_recovery.py
components/c4/OUTER_SNAPSHOT_RECOVERY_RUN.md
```

Do not replace or edit `audit_outer_snapshot.py`, `_outer_pipeline.py`, or the
original snapshot. They are bound to the frozen outer execution protocol.

## Test, commit and push

```powershell
python -m pytest components/c4/tests -q
git diff --check

git add components/c4/scripts/recover_outer_snapshot_save.py `
    components/c4/tests/unit/test_outer_snapshot_save_recovery.py `
    components/c4/OUTER_SNAPSHOT_RECOVERY_RUN.md

git commit -m "fix(c4): add preservation-safe outer snapshot newline recovery"
git push origin c4-kesigan
git status
```

The 11 new recovery tests passed locally. They cover exact CRLF diagnosis,
rejection of changed content/BOM/mixed newlines, reconstructed table matching,
binary output verification and original-file preservation. Full repository tests
and actual dataset recovery must run in the existing Windows environment.

## Verify, then finish

```powershell
python components/c4/scripts/recover_outer_snapshot_save.py
```

Expected plan confirmations:

```text
ALL 56 OUTER CASES / SAVED TABLES / PAIRED TARGETS / ORIGINAL BINDINGS: PASS
SOLE MISMATCH VERIFIED: Markdown LF -> Windows CRLF; original files preserved.
RETRAINING / FUTURE PREDICTION / POLICY RESELECTION: False
PLAN ONLY. Use --finish to save a separate verified snapshot.
```

Only after those checks pass:

```powershell
python components/c4/scripts/recover_outer_snapshot_save.py --finish
```

The verification runs again on the finish invocation. That is saved-evidence
verification, not another experiment or candidate training batch.

Expected completion:

```text
RECOVERED SNAPSHOT FILES / CHECKSUMS / ORIGINAL PRESERVATION: PASS
OUTPUT: .../outer-all16-outcome-audit-recovered-v2
METADATA SHA256: <actual hash printed by your run>
```

Output is under:
`c4_artifacts/meta_datasets/adult/adult-simulation-pp1-v2/outer-all16-outcome-audit-recovered-v2`.

The original `outer-all16-outcome-audit-v1` remains byte-identical, including its
original metadata and failing Markdown representation. The recovered snapshot
copies all saved CSV files byte-for-byte and writes the canonical Markdown using
binary I/O. Its metadata records the original actual/expected checksums, completed
batch hash and recovery source hash.

No TVAE, candidate or N1 training, N2 recalibration, policy selection, new model
prediction, threshold changes or deployment is performed. The script audits saved
paired predictions and reconstructs existing features. It does not change outcomes
or controller decisions.

If the script rejects a mismatch other than the exact newline translation, preserve
all files and inspect the specific error. Do not replace expected checksums or
remove either snapshot directory.

After successful recovery, use the recovered snapshot for final outer B1-B8/A1-A8
comparison reporting and paired group statistics. Do not rerun the outer candidate
batch or the original audit save to repair this documentation error.
