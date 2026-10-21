# Sprint 02 — Task 03: Baseline Model Performance Under Drift

**Research Project:** Intelligent MLOps Platform\
**Component:** C3 — Adaptive Intelligent Synthetic Data Repair Engine\
**Task:** Measure Baseline Model Performance Under Drift\
**Original Sprint:** 21–25 September 2026\
**Actual Completion:** 12 October 2026\
**Status:** Completed — pending documentation file verification

## 1. Objective

The objective was to evaluate how different controlled drift scenarios affect the existing UCI Adult XGBoost model.

The evaluation extends beyond accuracy to include precision, recall, F1-score, ROC-AUC, and confusion-matrix measurements.

These measurements establish baseline evidence for future synthetic data repair experiments.

## 2. Experimental Setup

- Dataset: Preprocessed UCI Adult
- Reference dataset: 26,048 records
- Evaluation batch: 6,000 records
- Model: Existing baseline XGBoost classifier
- Random seed: 42
- Scenarios evaluated: S00–S07
- Number of classification features: 12
- Target: Binary income classification

All scenarios were evaluated using the same baseline model without retraining.

## 3. Evaluation Metrics

The following metrics were calculated:

**Accuracy:** Proportion of correctly classified records.

**Precision:** Proportion of predicted positive records that are actually positive.

**Recall:** Proportion of actual positive records correctly identified.

**F1-score:** Harmonic mean of precision and recall.

**ROC-AUC:** Ability of the model to rank positive records above negative records.

**Confusion Matrix:** Counts of true negatives, false positives, false negatives, and true positives.

The shared injection framework already supported accuracy and ROC-AUC. Additional evaluation metrics were implemented independently inside Component 3.

## 4. Baseline Model Performance

| Metric          | Baseline Value |
| --------------- | -------------- |
| Accuracy        | 88.83%         |
| Precision       | 82.05%         |
| Recall          | 69.66%         |
| F1-score        | 0.7535         |
| ROC-AUC         | 0.9475         |
| True Negatives  | 4,306          |
| False Positives | 224            |
| False Negatives | 446            |
| True Positives  | 1,024          |

The reference batch contained 6,000 records.

## 5. Drift Scenario Performance

| Scenario | Accuracy | Precision | Recall | F1     | ROC-AUC |
| -------- | -------- | --------- | ------ | ------ | ------- |
| S00      | 88.83%   | 82.05%    | 69.66% | 0.7535 | 0.9475  |
| S01      | 88.02%   | 78.21%    | 70.82% | 0.7433 | 0.9392  |
| S02      | 86.75%   | 79.42%    | 61.97% | 0.6962 | 0.9239  |
| S03      | 88.00%   | 83.01%    | 64.15% | 0.7237 | 0.9279  |
| S04      | 88.70%   | 82.89%    | 67.89% | 0.7464 | 0.9452  |
| S05      | 86.52%   | 71.28%    | 75.31% | 0.7324 | 0.9309  |
| S06      | 87.23%   | 82.84%    | 60.41% | 0.6987 | 0.9338  |
| S07      | 86.03%   | 76.92%    | 61.43% | 0.6831 | 0.9199  |

## 6. Important Experimental Findings

### 6.1 Accuracy Alone Can Hide Model Degradation

Scenario S06 caused an accuracy decrease of 1.60 percentage points.

However, recall decreased by 9.25 percentage points, and F1-score decreased by approximately 5.48 percentage points.

False negatives increased from 446 to 582.

This demonstrates why accuracy alone is insufficient for evaluating model degradation.

### 6.2 Precision Degradation Under Multiple-Feature Drift

Scenario S05 produced the largest precision decrease.

Precision decreased from 82.05% to 71.28%, a reduction of 10.77 percentage points.

False positives increased from 224 to 446.

This indicates that different drift scenarios can affect classification errors in different ways.

### 6.3 Combined Drift Produced the Largest Accuracy and F1 Losses

Scenario S07 produced:

- Accuracy drop: 2.80 percentage points
- Precision drop: 5.13 percentage points
- Recall drop: 8.23 percentage points
- F1-score drop: approximately 7.04 percentage points
- ROC-AUC drop: 2.76 percentage points

S07 had the largest accuracy and F1 degradation among the evaluated scenarios, although other scenarios produced larger precision or recall decreases.

### 6.4 No-Drift Control

Scenario S00 reproduced the reference-batch model performance.

This confirms that the no-drift control did not introduce measurable performance degradation.

## 7. Metric Verification

All eight scenarios passed independent mathematical consistency checks.

Verified:

- Confusion-matrix totals equal 6,000 records.
- Accuracy matches the confusion-matrix calculation.
- Precision matches the confusion-matrix calculation.
- Recall matches the confusion-matrix calculation.
- F1-score matches the harmonic mean of precision and recall.
- ROC-AUC values are within the valid range of 0–1.

ROC-AUC was calculated using scikit-learn. The independent verification checked its bounds rather than recomputing it from saved prediction probabilities.

## 8. Relevance to Component 3

These results will support future synthetic data repair experiments.

Component 3 should not evaluate repair success using accuracy alone.

Repair experiments should also measure:

- Positive-class precision and recall
- F1-score recovery
- ROC-AUC changes
- Changes in false positives and false negatives
- Performance on affected and non-affected data slices

A successful repair should improve relevant model performance without causing unacceptable degradation elsewhere.

## 9. Implementation and Evidence

**Evaluation script**

`component3/src/experiments/evaluate_drift_performance.py`

**Saved evaluation results**

`component3/data/drifted/drift_performance_metrics.csv`

**Input datasets**

`component3/data/drifted/S00.csv` through `S07.csv`

**Drift experiment documentation**

`component3/docs/sprint02_controlled_drift_experiments.md`

## 10. Research Limitations

The experiment used one reference-data batch and one random seed.

Because the reference dataset was also used during baseline model development, these results are not independent held-out estimates of real-world performance.

The injected drift scenarios retain original target labels, which may create unrealistic feature-label combinations.

Further experiments should include independent test data, multiple seeds, and statistical uncertainty analysis.

## 11. Conclusion

Sprint 02 Task 03 established a reproducible multi-metric performance evaluation for eight controlled drift scenarios.

All eight scenarios were evaluated, the classification metrics were verified, and the results were saved for later repair experiments.

The findings show that different drift types affect precision, recall, F1-score, and overall accuracy differently.

**Final Status:** Implementation and metric verification completed; documentation file verification pending.
