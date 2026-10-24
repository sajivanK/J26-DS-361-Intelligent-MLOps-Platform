# Privacy DCR investigation — 24 October 2026

## Finding

The `0.05` DCR threshold was not loosened. The current calculation is internally consistent with its documented normalized mixed-feature distance, and observed failures are caused by generated records being too close to training records rather than by a row-count or unit-conversion bug.

## What the validator calculates

1. Numeric values are min-max scaled using bounds from the real training reference and clipped to `[0, 1]`.
2. Each categorical feature is one-hot encoded; a category mismatch contributes a unit distance after the `sqrt(2)` normalization.
3. The combined vector is divided by `sqrt(number of original evaluated features)`, making feature contributions comparable across schemas.
4. For every synthetic row, C3 measures Euclidean distance to its closest real training row.
5. It rejects exact matches, any forbidden near-duplicate ratio (`distance < 0.01`), or a fifth-percentile DCR below `0.05`.

The privacy-screened bounded-pool path applies the same per-record distance calculation, removes rows below `0.05`, removes feature duplicates, and takes the first eligible rows without inspecting evaluation performance. This prevents threshold tuning against the holdout.

## Existing measured evidence

- A one-epoch CTGAN study on 360 Adult slice rows produced DCR p05 `0.4101`, with no exact or near duplicates. This shows the normalized calculation can pass without threshold relaxation.
- The initial 20-epoch saved real-batch run produced DCR p05 `0.1052`, also passing `0.05`.
- In the later 200-epoch bounded pool, the raw 180-row pool had DCR p05 `0.0219`, two near duplicates, and 28 records below `0.05`. After deterministic filtering, 60 selected rows achieved DCR p05 `0.0628` with no near duplicates.

These results support keeping `0.05`: generator/training configuration materially changes proximity, and the bounded filter removes unsafe rows without changing the acceptance rule.

## Limitations and next evidence

The threshold is an empirical project policy, not differential privacy. Before making a broader privacy claim, run the same fixed calculation across TVAE, CTGAN, and TabDDPM on multiple datasets and seeds, report full DCR distributions and holdout NNDR, and pre-register any threshold revision. Do not change the threshold merely to turn a failed candidate into a pass.
