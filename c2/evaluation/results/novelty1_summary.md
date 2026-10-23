# Novelty 1 results: Temporal SHAP vs baselines

Streams: 158 (5 cycles, drift onset at cycle 2, batch 1000 from test.csv, seeds [11, 22, 33]). N1 alarm threshold z > 4.15.

### All drifted streams (n=138)

| Method | top1 | top3 | alarm |
|---|---|---|---|
| A | 95.7% | 97.1% | 94.2% |
| B | 72.5% | 83.3% | n/a |
| B_pooled | 67.4% | 88.4% | n/a |
| N1_single | 94.2% | 97.1% | 92.0% |
| N1 | 95.7% | 97.1% | 92.8% |

### high_importance (n=60)

| Method | top1 | top3 | alarm |
|---|---|---|---|
| A | 100.0% | 100.0% | 100.0% |
| B | 83.3% | 90.0% | n/a |
| B_pooled | 76.7% | 91.7% | n/a |
| N1_single | 91.7% | 95.0% | 95.0% |
| N1 | 91.7% | 95.0% | 95.0% |

### low_importance (n=60)

| Method | top1 | top3 | alarm |
|---|---|---|---|
| A | 100.0% | 100.0% | 100.0% |
| B | 73.3% | 83.3% | n/a |
| B_pooled | 75.0% | 90.0% | n/a |
| N1_single | 98.3% | 100.0% | 93.3% |
| N1 | 100.0% | 100.0% | 95.0% |

### interaction (n=12)

| Method | top1 | top3 | alarm |
|---|---|---|---|
| A | 50.0% | 66.7% | 33.3% |
| B | 16.7% | 50.0% | n/a |
| B_pooled | 8.3% | 75.0% | n/a |
| N1_single | 83.3% | 91.7% | 66.7% |
| N1 | 91.7% | 91.7% | 66.7% |

### combined (n=6)

| Method | top1 | top3 | alarm |
|---|---|---|---|
| A | 100.0% | 100.0% | 100.0% |
| B | 66.7% | 83.3% | n/a |
| B_pooled | 16.7% | 66.7% | n/a |
| N1_single | 100.0% | 100.0% | 100.0% |
| N1 | 100.0% | 100.0% | 100.0% |

### False alarms on clean streams (n=20)

| Method | False alarm rate |
|---|---|
| A | 10.0% |
| B | n/a (always names a cause) |
| B_pooled | n/a (always names a cause) |
| N1_single | 0.0% |
| N1 | 0.0% |

### Novelty 1 temporal behaviour

- Mean detection delay after onset: 0.79 cycles (detected 128/138 drifted streams)
- Mean diagnosis stability at last cycle (drifted): 0.964
- Interaction streams labelled behaviour_shift: 91.7%
- Marginal streams labelled distribution_shift: 93.3%
