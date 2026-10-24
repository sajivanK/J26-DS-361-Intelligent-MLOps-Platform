# Novelty 2 results: Drift Impact Score vs single-evidence baselines

Streams: 158 (5 cycles, onset at cycle 2, batch 1000 from test.csv, seeds [11, 22, 33]). Default weights (E, D, P, G) = (0.35, 0.15, 0.35, 0.15).

### Root cause ranking, all drifted streams (n=138)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 95.7% | 97.1% |
| B2_explanation | 95.7% | 97.1% |
| B3_performance | 73.9% | 84.1% |
| B4_subgroup | 0.7% | 5.8% |
| G_subgroup_chg | 39.1% | 54.3% |
| N2_DIS | 98.6% | 99.3% |

### high_importance (n=60)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 100.0% | 100.0% |
| B2_explanation | 91.7% | 95.0% |
| B3_performance | 63.3% | 78.3% |
| B4_subgroup | 0.0% | 1.7% |
| G_subgroup_chg | 48.3% | 63.3% |
| N2_DIS | 98.3% | 100.0% |

### low_importance (n=60)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 100.0% | 100.0% |
| B2_explanation | 100.0% | 100.0% |
| B3_performance | 81.7% | 86.7% |
| B4_subgroup | 0.0% | 0.0% |
| G_subgroup_chg | 20.0% | 36.7% |
| N2_DIS | 100.0% | 100.0% |

### interaction (n=12)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 50.0% | 66.7% |
| B2_explanation | 91.7% | 91.7% |
| B3_performance | 83.3% | 91.7% |
| B4_subgroup | 0.0% | 41.7% |
| G_subgroup_chg | 75.0% | 91.7% |
| N2_DIS | 91.7% | 91.7% |

### combined (n=6)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 100.0% | 100.0% |
| B2_explanation | 100.0% | 100.0% |
| B3_performance | 83.3% | 100.0% |
| B4_subgroup | 16.7% | 33.3% |
| G_subgroup_chg | 66.7% | 66.7% |
| N2_DIS | 100.0% | 100.0% |

### Does the top score track real damage? (all streams, n=158)

| Method | Spearman r with true accuracy drop | p-value |
|---|---|---|
| B1_drift | 0.422 | 3.3e-08 |
| B2_explanation | 0.306 | 9e-05 |
| B3_performance | 0.693 | 6.4e-24 |
| B4_subgroup | 0.149 | 0.062 |
| G_subgroup_chg | 0.561 | 1.8e-14 |
| N2_DIS | 0.606 | 3.3e-17 |

### Novelty 2 harm estimate

- Spearman r (estimated vs true drop): 0.757 (p=1.3e-30)
- Mean absolute error: 0.58 pp

### Malignant detection (true drop > 2.0pp; 31 malignant of 158 streams)

| Rule | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| N1 alarm (any explanation change = malignant) | 24.2% | 100.0% | 39.0% | 38.6% |
| N2 is_malignant | 77.8% | 67.7% | 72.4% | 89.9% |

### Weight ablation (top-1 on drifted streams)

| w_explanation | w_drift | w_performance | w_subgroup | top1 |
|---|---|---|---|---|
| 0.00 | 0.30 | 0.20 | 0.50 | 99.3% |
| 0.10 | 0.30 | 0.30 | 0.30 | 99.3% |
| 0.00 | 0.40 | 0.50 | 0.10 | 99.3% |
| 0.20 | 0.30 | 0.20 | 0.30 | 99.3% |
| 0.50 | 0.20 | 0.30 | 0.00 | 99.3% |
| 0.35 | 0.15 | 0.35 | 0.15 | 98.6% (default, 4 signals) |
| 0.40 | 0.20 | 0.40 | 0.00 | 99.3% (earlier 3-signal setting) |
| 0.00 | 1.00 | 0.00 | 0.00 | 95.7% (single signal) |
| 1.00 | 0.00 | 0.00 | 0.00 | 95.7% (single signal) |
| 0.00 | 0.00 | 1.00 | 0.00 | 73.9% (single signal) |
| 0.00 | 0.00 | 0.00 | 1.00 | 39.1% (single signal) |

36 of 287 weight combinations reach the best top-1. Best with the subgroup signal: 99.3%; best without it: 99.3%.
Weights are reported as a sensitivity analysis on the same streams; the default weights were fixed before this run.
