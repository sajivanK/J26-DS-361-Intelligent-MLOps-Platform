# Novelty 3 results: Hierarchical Localization vs slice finding

Full pipeline N1 -> N2 -> N3 on 138 drifted streams (5 cycles, onset at cycle 2, batch 1000 from test.csv, seeds [11, 22, 33]). Truth = rows correct before injection and wrong after.

Evaluated on 120 of 138 drifted streams with at least 10 truly drift-harmed rows. Root cause from the Drift Impact Score was correct on 99.2% of them. Median share of harmed rows per batch: 2.5%. Lift is the median over streams.


### All evaluated streams (n=120)

| Method | Precision | Recall | F1 | Support | Lift | % of oracle F1 |
|---|---|---|---|---|---|---|
| SliceLine | 11.5% | 24.9% | 12.1% | 10.8% | 2.0x | 60% |
| Guide | 6.5% | 30.3% | 7.9% | 22.7% | 1.3x | 39% |
| N3_level2 | 7.3% | 55.5% | 11.8% | 36.0% | 1.6x | 58% |
| N3 | 13.8% | 35.0% | 18.0% | 11.5% | 3.2x | 89% |
| Oracle | 15.3% | 38.9% | 20.2% | 10.8% | 4.3x | 100% |

### high_importance (n=55)

| Method | Precision | Recall | F1 | Support | Lift | % of oracle F1 |
|---|---|---|---|---|---|---|
| SliceLine | 18.7% | 26.2% | 17.0% | 10.8% | 2.3x | 75% |
| Guide | 10.0% | 20.8% | 10.6% | 14.0% | 1.6x | 47% |
| N3_level2 | 8.6% | 49.8% | 13.8% | 37.2% | 1.4x | 60% |
| N3 | 15.6% | 37.5% | 20.1% | 16.0% | 2.5x | 88% |
| Oracle | 17.7% | 38.2% | 22.8% | 14.0% | 3.1x | 100% |

### low_importance (n=47)

| Method | Precision | Recall | F1 | Support | Lift | % of oracle F1 |
|---|---|---|---|---|---|---|
| SliceLine | 5.3% | 24.0% | 7.9% | 10.4% | 1.6x | 40% |
| Guide | 3.2% | 45.3% | 5.4% | 35.3% | 1.0x | 27% |
| N3_level2 | 6.6% | 71.9% | 10.8% | 40.0% | 1.9x | 55% |
| N3 | 13.7% | 38.4% | 18.3% | 7.2% | 6.9x | 93% |
| Oracle | 14.1% | 45.7% | 19.7% | 8.7% | 6.4x | 100% |

### interaction (n=12)

| Method | Precision | Recall | F1 | Support | Lift | % of oracle F1 |
|---|---|---|---|---|---|---|
| SliceLine | 3.9% | 21.8% | 5.8% | 11.9% | 1.7x | 47% |
| Guide | 3.3% | 22.4% | 5.0% | 19.0% | 1.3x | 40% |
| N3_level2 | 4.3% | 31.7% | 6.9% | 24.1% | 2.0x | 56% |
| N3 | 7.5% | 18.1% | 10.0% | 8.5% | 3.3x | 80% |
| Oracle | 10.0% | 21.5% | 12.4% | 5.0% | 4.4x | 100% |

### combined (n=6)

| Method | Precision | Recall | F1 | Support | Lift | % of oracle F1 |
|---|---|---|---|---|---|---|
| SliceLine | 10.1% | 26.5% | 14.0% | 11.9% | 2.2x | 82% |
| Guide | 5.7% | 15.2% | 8.1% | 12.4% | 1.3x | 48% |
| N3_level2 | 7.3% | 27.2% | 11.3% | 16.6% | 1.5x | 66% |
| N3 | 10.2% | 20.2% | 12.8% | 9.4% | 1.9x | 75% |
| Oracle | 14.0% | 26.2% | 17.0% | 9.5% | 3.8x | 100% |

### Novelty 3 depth

- level 2: 4 streams
- level 3: 116 streams

### Example rules (first stream per group)

- high_importance (marital_status, m=0.2): N3 `marital_status == 0 and relationship == 0` vs SliceLine `relationship == 0`
- low_importance (capital_loss, m=0.2): N3 `capital_loss > 0.07438 and occupation == 9` vs SliceLine `marital_status == 2 and occupation == 11`
- interaction (education_num+occupation, m=0.4): N3 `occupation == 13 and marital_status == 2` vs SliceLine `relationship == 0 and education_num > -1.192 and education_num <= -0.418`
- combined (education_num+age+occupation, m=0.4): N3 `education_num > 1.558 and relationship == 0` vs SliceLine `marital_status == 2 and age > -0.04321 and age <= 0.1767`
