# Novelty 2 without labels: finding the responsible column

258 monitoring streams (5 cycles, onset at cycle 2, batch 1000 from test.csv). Scenario suite seeds [11, 22, 33], harmful+harmless pair seeds [11, 33, 44, 55]. Production batches carry no labels; labels are used only to grade. Label-free weights (E, D, DI, PS) = (0.35, 0.15, 0.35, 0.15). Impact signal: decision change (share of decisions that flip when the feature is restored); N2_free_DoC tests the harm-drop slot with DoC standing in for C1's predict_harm.

### A. Harmful + harmless pairs, held-out seeds (n=15)

| Method | top1 | top3 | blamed the harmless drifted feature |
|---|---|---|---|
| B1_drift | 66.7% | 100.0% | 33.3% |
| B2_explanation | 33.3% | 100.0% | 46.7% |
| S_decision | 100.0% | 100.0% | 0.0% |
| S_predshift | 0.0% | 33.3% | 6.7% |
| N2_free | 93.3% | 100.0% | 6.7% |
| N2_free_DoC (DoC stand-in for C1) | 66.7% | 100.0% | 33.3% |
| N2_labelled (upper bound, needs labels) | 100.0% | 100.0% | 0.0% |

### A. Harmful + harmless pairs, all seeds (n=19; 100 pair streams, 19 were harmful+harmless)

| Method | top1 | top3 | blamed the harmless drifted feature |
|---|---|---|---|
| B1_drift | 63.2% | 100.0% | 36.8% |
| B2_explanation | 42.1% | 100.0% | 42.1% |
| S_decision | 100.0% | 100.0% | 0.0% |
| S_predshift | 0.0% | 42.1% | 10.5% |
| N2_free | 94.7% | 100.0% | 5.3% |
| N2_free_DoC (DoC stand-in for C1) | 68.4% | 100.0% | 31.6% |
| N2_labelled (upper bound, needs labels) | 100.0% | 100.0% | 0.0% |

### B. Scenario suite, held-out seeds, drifted streams (n=46)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 95.7% | 100.0% |
| B2_explanation | 95.7% | 97.8% |
| S_decision | 87.0% | 95.7% |
| S_predshift | 47.8% | 58.7% |
| N2_free | 97.8% | 100.0% |
| N2_free_DoC (DoC stand-in for C1) | 100.0% | 100.0% |
| N2_labelled (upper bound, needs labels) | 100.0% | 100.0% |

### B. Scenario suite, all seeds, drifted streams (n=138)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 95.7% | 97.1% |
| B2_explanation | 95.7% | 97.1% |
| S_decision | 88.4% | 94.9% |
| S_predshift | 55.1% | 65.9% |
| N2_free | 98.6% | 99.3% |
| N2_free_DoC (DoC stand-in for C1) | 99.3% | 99.3% |
| N2_labelled (upper bound, needs labels) | 98.6% | 99.3% |

### B. high_importance, all seeds (n=60)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 100.0% | 100.0% |
| B2_explanation | 91.7% | 95.0% |
| S_decision | 95.0% | 96.7% |
| S_predshift | 61.7% | 78.3% |
| N2_free | 100.0% | 100.0% |
| N2_free_DoC (DoC stand-in for C1) | 100.0% | 100.0% |
| N2_labelled (upper bound, needs labels) | 98.3% | 100.0% |

### B. low_importance, all seeds (n=60)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 100.0% | 100.0% |
| B2_explanation | 100.0% | 100.0% |
| S_decision | 93.3% | 98.3% |
| S_predshift | 36.7% | 43.3% |
| N2_free | 100.0% | 100.0% |
| N2_free_DoC (DoC stand-in for C1) | 100.0% | 100.0% |
| N2_labelled (upper bound, needs labels) | 100.0% | 100.0% |

### B. interaction, all seeds (n=12)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 50.0% | 66.7% |
| B2_explanation | 91.7% | 91.7% |
| S_decision | 25.0% | 66.7% |
| S_predshift | 91.7% | 100.0% |
| N2_free | 83.3% | 91.7% |
| N2_free_DoC (DoC stand-in for C1) | 91.7% | 91.7% |
| N2_labelled (upper bound, needs labels) | 91.7% | 91.7% |

### B. combined, all seeds (n=6)

| Method | top1 | top3 |
|---|---|---|
| B1_drift | 100.0% | 100.0% |
| B2_explanation | 100.0% | 100.0% |
| S_decision | 100.0% | 100.0% |
| S_predshift | 100.0% | 100.0% |
| N2_free | 100.0% | 100.0% |
| N2_free_DoC (DoC stand-in for C1) | 100.0% | 100.0% |
| N2_labelled (upper bound, needs labels) | 100.0% | 100.0% |

### Development vs held-out seeds (top-1)

The impact signal was designed on seeds 11 and 22; seeds 33, 44 and 55 were not looked at before the design was fixed.

| Method | suite, development | suite, held-out | pairs, development | pairs, held-out |
|---|---|---|---|---|
| B1_drift | 95.7% | 95.7% | 50.0% | 66.7% |
| B2_explanation | 95.7% | 95.7% | 75.0% | 33.3% |
| S_decision | 89.1% | 87.0% | 100.0% | 100.0% |
| S_predshift | 58.7% | 47.8% | 0.0% | 0.0% |
| N2_free | 98.9% | 97.8% | 100.0% | 93.3% |
| N2_free_DoC | 98.9% | 100.0% | 75.0% | 66.7% |
| N2_labelled | 97.8% | 100.0% | 100.0% | 100.0% |

### Does it track real damage? (suite + pairs, n=258)

| Score | Spearman r with true accuracy drop | p-value |
|---|---|---|
| B1_drift top score | 0.506 | 3.8e-18 |
| B2_explanation top score | 0.404 | 1.6e-11 |
| S_decision top score | 0.645 | 8.5e-32 |
| S_predshift top score | 0.656 | 3.7e-33 |
| N2_free top score | 0.710 | 6.6e-41 |
| N2_free_DoC top score | 0.360 | 2.6e-09 |
| N2_labelled top score | 0.773 | 1.3e-52 |
| root feature's share of flipped decisions (label-free) | 0.640 | 4.5e-31 |

### Telling drifted from clean streams by the top score (138 drifted, 20 clean)

| Method | AUC |
|---|---|
| B1_drift | 0.961 |
| B2_explanation | 0.976 |
| N2_free | 0.996 |
| N2_labelled | 0.982 |
| N1 alarm (persistence rule), false alarms on clean | 0.0% |

### Weight ablation, label-free signals (top-1, suite + harmful/harmless pairs, n=157)

| w_E | w_D | w_DI | w_PS | top1 |
|---|---|---|---|---|
| 0.30 | 0.20 | 0.20 | 0.30 | 98.7% |
| 0.40 | 0.20 | 0.10 | 0.30 | 98.7% |
| 0.50 | 0.30 | 0.00 | 0.20 | 98.7% |
| 0.40 | 0.30 | 0.00 | 0.30 | 98.7% |
| 0.40 | 0.20 | 0.30 | 0.10 | 98.7% |
| 0.35 | 0.15 | 0.35 | 0.15 | 98.1% (default) |
| 0.70 | 0.30 | 0.00 | 0.00 | 93.6% (explanation + drift only) |
| 0.00 | 1.00 | 0.00 | 0.00 | 91.7% (single signal) |
| 0.00 | 0.00 | 1.00 | 0.00 | 89.8% (single signal) |
| 1.00 | 0.00 | 0.00 | 0.00 | 89.2% (single signal) |
| 0.00 | 0.00 | 0.00 | 1.00 | 48.4% (single signal) |

14 of 287 weight combinations reach the best top-1. The default weights were fixed before this run (the same as the labelled score); the ablation is a sensitivity analysis on the same streams.
