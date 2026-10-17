# Novelty 2 results: Drift Impact Score vs single-evidence baselines

Streams: 158 (5 cycles, onset at cycle 2, batch 1000 from test.csv, seeds [11, 22, 33]). Default weights (E, D, P) = (0.4, 0.2, 0.4).

### Root cause ranking, all drifted streams (n=138)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 95.7% | 97.1% |
| B2_explanation | 95.7% | 97.1% |
| B3_performance | 73.9% | 84.1% |
| B4_subgroup | 0.7% | 5.8% |
| N2_DIS | 99.3% | 99.3% |

### high_importance (n=60)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 100.0% | 100.0% |
| B2_explanation | 91.7% | 95.0% |
| B3_performance | 63.3% | 78.3% |
| B4_subgroup | 0.0% | 1.7% |
| N2_DIS | 100.0% | 100.0% |

### low_importance (n=60)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 100.0% | 100.0% |
| B2_explanation | 100.0% | 100.0% |
| B3_performance | 81.7% | 86.7% |
| B4_subgroup | 0.0% | 0.0% |
| N2_DIS | 100.0% | 100.0% |

### interaction (n=12)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 50.0% | 66.7% |
| B2_explanation | 91.7% | 91.7% |
| B3_performance | 83.3% | 91.7% |
| B4_subgroup | 0.0% | 41.7% |
| N2_DIS | 91.7% | 91.7% |

### combined (n=6)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 100.0% | 100.0% |
| B2_explanation | 100.0% | 100.0% |
| B3_performance | 83.3% | 100.0% |
| B4_subgroup | 16.7% | 33.3% |
| N2_DIS | 100.0% | 100.0% |

### Does the top score track real damage? (all streams, n=158)

| Method | Spearman r with true accuracy drop | p-value |
|---|---|---|
| B1_drift | 0.422 | 3.3e-08 |
| B2_explanation | 0.306 | 9e-05 |
| B3_performance | 0.693 | 6.4e-24 |
| B4_subgroup | 0.149 | 0.062 |
| N2_DIS | 0.576 | 2.3e-15 |

### Novelty 2 harm estimate

- Spearman r (estimated vs true drop): 0.752 (p=4.7e-30)
- Mean absolute error: 0.59 pp

### Malignant detection (true drop > 2.0pp; 31 malignant of 158 streams)

| Rule | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| N1 alarm (any explanation change = malignant) | 24.2% | 100.0% | 39.0% | 38.6% |
| N2 is_malignant | 77.8% | 67.7% | 72.4% | 89.9% |

### Weight ablation (top-1 on drifted streams)

| w_explanation | w_drift | w_performance | top1 |
|---|---|---|---|
| 0.3 | 0.3 | 0.4 | 99.3% |
| 0.2 | 0.4 | 0.4 | 99.3% |
| 0.4 | 0.3 | 0.3 | 99.3% |
| 0.4 | 0.2 | 0.4 | 99.3% (default) |
| 0.5 | 0.2 | 0.3 | 99.3% |
| 1.0 | 0.0 | 0.0 | 95.7% (single signal) |
| 0.0 | 1.0 | 0.0 | 95.7% (single signal) |
| 0.0 | 0.0 | 1.0 | 73.9% (single signal) |

Weights are reported as a sensitivity analysis on the same streams; the default weights were fixed before this run.
