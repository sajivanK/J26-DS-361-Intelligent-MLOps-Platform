# Novelty 3 without labels: finding the responsible value range

Full label-free pipeline N1 -> N2 -> N3 on 138 drifted streams (5 cycles, onset at cycle 2, batch 1000 from test.csv, seeds [11, 22, 33]). Production batches carry no labels; the answer key grades afterwards. Truth 1, harmed rows: right before injection, wrong after. Truth 2, changed rows: the drifted feature's value differs from the clean batch.

Evaluated on 120 of 138 drifted streams with at least 10 truly drift-harmed rows. The label-free root cause was correct on 98.3% of them. Median share of harmed rows per batch: 2.5%; median share of changed rows: 85.7%. Lift is the median over streams.


### All evaluated streams (n=120)

| Method | uses labels | vs harmed rows: precision | recall | F1 | support | lift | % of oracle F1 | vs changed rows: precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ShiftRegion | no | 4.5% | 73.4% | 8.2% | 63.0% | 1.2x | 40% | 91.3% | 70.5% | 77.5% |
| N3_free_level2 | no | 6.5% | 63.9% | 10.9% | 47.4% | 1.3x | 54% | 91.5% | 55.1% | 60.0% |
| N3_free | no | 10.6% | 53.3% | 15.9% | 26.9% | 2.2x | 79% | 92.4% | 30.4% | 38.2% |
| N3_free_flips | no | 13.0% | 43.5% | 17.7% | 18.5% | 2.8x | 88% | 92.3% | 20.5% | 25.9% |
| SliceLine | yes | 11.5% | 24.9% | 12.1% | 10.8% | 2.0x | 60% | 79.4% | 11.0% | 17.9% |
| Guide | yes | 6.5% | 30.3% | 7.9% | 22.7% | 1.3x | 39% | 77.2% | 22.5% | 28.7% |
| N3_labelled | yes | 13.8% | 35.0% | 18.0% | 11.5% | 3.2x | 89% | 90.3% | 12.8% | 20.3% |
| Oracle | yes | 15.3% | 38.9% | 20.2% | 10.8% | 4.3x | 100% | 90.9% | 12.5% | 20.3% |

### Held-out seed 33 (n=42)

| Method | uses labels | vs harmed rows: precision | recall | F1 | support | lift | % of oracle F1 | vs changed rows: precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ShiftRegion | no | 4.6% | 74.1% | 8.2% | 64.1% | 1.2x | 43% | 91.7% | 71.4% | 78.4% |
| N3_free_level2 | no | 7.1% | 65.9% | 11.8% | 46.7% | 1.4x | 62% | 91.9% | 54.4% | 59.2% |
| N3_free | no | 10.9% | 52.6% | 16.3% | 24.9% | 2.3x | 86% | 93.3% | 28.0% | 36.3% |
| N3_free_flips | no | 12.9% | 38.5% | 17.4% | 15.2% | 2.8x | 92% | 94.0% | 16.7% | 22.3% |
| SliceLine | yes | 13.1% | 32.2% | 13.9% | 12.6% | 2.6x | 74% | 80.4% | 12.8% | 20.4% |
| Guide | yes | 6.2% | 26.9% | 7.4% | 20.6% | 1.4x | 39% | 74.9% | 19.5% | 25.4% |
| N3_labelled | yes | 13.0% | 31.2% | 16.6% | 12.1% | 3.4x | 88% | 91.8% | 12.8% | 19.8% |
| Oracle | yes | 14.2% | 38.1% | 18.9% | 11.8% | 3.8x | 100% | 92.4% | 13.3% | 21.2% |

### high_importance (n=55)

| Method | uses labels | vs harmed rows: precision | recall | F1 | support | lift | % of oracle F1 | vs changed rows: precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ShiftRegion | no | 6.7% | 72.9% | 11.9% | 65.9% | 1.1x | 52% | 99.2% | 69.0% | 79.7% |
| N3_free_level2 | no | 8.0% | 51.8% | 13.1% | 43.0% | 1.3x | 57% | 99.2% | 46.0% | 57.0% |
| N3_free | no | 12.7% | 46.8% | 18.7% | 24.4% | 2.1x | 82% | 99.4% | 26.1% | 38.4% |
| N3_free_flips | no | 15.8% | 35.2% | 20.4% | 14.6% | 2.6x | 89% | 99.5% | 15.5% | 25.1% |
| SliceLine | yes | 18.7% | 26.2% | 17.0% | 10.8% | 2.3x | 75% | 92.7% | 10.6% | 17.9% |
| Guide | yes | 10.0% | 20.8% | 10.6% | 14.0% | 1.6x | 47% | 90.5% | 13.6% | 22.7% |
| N3_labelled | yes | 15.6% | 37.5% | 20.1% | 16.0% | 2.5x | 88% | 96.1% | 16.4% | 26.2% |
| Oracle | yes | 17.7% | 38.2% | 22.8% | 14.0% | 3.1x | 100% | 96.7% | 14.6% | 23.7% |

### low_importance (n=47)

| Method | uses labels | vs harmed rows: precision | recall | F1 | support | lift | % of oracle F1 | vs changed rows: precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ShiftRegion | no | 2.5% | 82.0% | 4.7% | 65.4% | 1.2x | 24% | 92.9% | 81.6% | 85.6% |
| N3_free_level2 | no | 5.8% | 78.9% | 10.0% | 48.4% | 1.3x | 51% | 92.9% | 64.6% | 69.4% |
| N3_free | no | 10.6% | 57.7% | 16.1% | 18.1% | 2.7x | 82% | 95.0% | 25.0% | 35.9% |
| N3_free_flips | no | 12.3% | 48.6% | 18.2% | 11.4% | 4.8x | 93% | 94.3% | 15.6% | 24.2% |
| SliceLine | yes | 5.3% | 24.0% | 7.9% | 10.4% | 1.6x | 40% | 70.1% | 11.2% | 18.1% |
| Guide | yes | 3.2% | 45.3% | 5.4% | 35.3% | 1.0x | 27% | 68.0% | 35.6% | 38.5% |
| N3_labelled | yes | 13.7% | 38.4% | 18.3% | 7.2% | 6.9x | 93% | 93.1% | 10.1% | 16.2% |
| Oracle | yes | 14.1% | 45.7% | 19.7% | 8.7% | 6.4x | 100% | 94.5% | 12.3% | 19.5% |

### interaction (n=12)

| Method | uses labels | vs harmed rows: precision | recall | F1 | support | lift | % of oracle F1 | vs changed rows: precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ShiftRegion | no | 2.2% | 34.8% | 4.0% | 35.3% | 1.2x | 32% | 44.0% | 32.9% | 31.7% |
| N3_free_level2 | no | 2.4% | 81.3% | 4.4% | 79.9% | 1.0x | 36% | 46.2% | 80.0% | 54.3% |
| N3_free | no | 2.4% | 81.3% | 4.4% | 79.9% | 1.0x | 36% | 46.2% | 80.0% | 54.3% |
| N3_free_flips | no | 3.8% | 71.7% | 5.6% | 68.2% | 1.0x | 45% | 47.7% | 68.4% | 40.7% |
| SliceLine | yes | 3.9% | 21.8% | 5.8% | 11.9% | 1.7x | 47% | 44.9% | 11.6% | 15.4% |
| Guide | yes | 3.3% | 22.4% | 5.0% | 19.0% | 1.3x | 40% | 40.7% | 17.0% | 21.3% |
| N3_labelled | yes | 7.5% | 18.1% | 10.0% | 8.5% | 3.3x | 80% | 47.5% | 8.1% | 11.5% |
| Oracle | yes | 10.0% | 21.5% | 12.4% | 5.0% | 4.4x | 100% | 45.5% | 5.4% | 9.0% |

### combined (n=6)

| Method | uses labels | vs harmed rows: precision | recall | F1 | support | lift | % of oracle F1 | vs changed rows: precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ShiftRegion | no | 5.1% | 88.4% | 9.6% | 73.3% | 1.2x | 57% | 100.0% | 73.3% | 84.4% |
| N3_free_level2 | no | 6.8% | 22.7% | 10.4% | 15.0% | 1.4x | 61% | 100.0% | 15.0% | 25.8% |
| N3_free | no | 8.4% | 22.7% | 12.0% | 13.1% | 2.0x | 70% | 100.0% | 13.1% | 22.4% |
| N3_free_flips | no | 10.7% | 23.5% | 14.3% | 9.9% | 2.7x | 84% | 100.0% | 9.9% | 17.5% |
| SliceLine | yes | 10.1% | 26.5% | 14.0% | 11.9% | 2.2x | 82% | 100.0% | 11.9% | 20.3% |
| Guide | yes | 5.7% | 15.2% | 8.1% | 12.4% | 1.3x | 48% | 100.0% | 12.4% | 22.0% |
| N3_labelled | yes | 10.2% | 20.2% | 12.8% | 9.4% | 1.9x | 75% | 100.0% | 9.4% | 16.6% |
| Oracle | yes | 14.0% | 26.2% | 17.0% | 9.5% | 3.8x | 100% | 100.0% | 9.5% | 16.6% |

### Label-free Novelty 3 depth

- level 1: 9 streams
- level 2: 22 streams
- level 3: 89 streams
