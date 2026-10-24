# C3 output contract changelog

## Version 1.2 — 24 October 2026

- Added optional `repair_strategy`: `TARGETED_SYNTHETIC_REPAIR`, `AUTHENTIC_WEIGHTED_ADAPTATION`, or `NO_REPAIR`.
- Permitted `fidelity_score` and `target_coverage` to be `null` only for authentic weighted adaptation, where synthetic-only measurements do not apply.
- Defined an approved authentic handoff as `generator=NONE`, zero synthetic rows, zero augmentation ratio, privacy `PASS` (synthetic privacy screening not applicable), and a dataset URI pointing to the saved labelled adaptation rows.
- Preserved all twelve v1.1 required fields. Existing synthetic payloads keep their prior meaning.

This shared-contract change requires C4-owner review before merge. C4 consumers using strict extra-field rejection must accept the optional field, and their two nullable metric types must be updated before they consume authentic-only approvals.

## Version 1.1

- Twelve-field synthetic-repair handoff.
- `generator=NONE` represented rejected/no-repair outcomes only.
