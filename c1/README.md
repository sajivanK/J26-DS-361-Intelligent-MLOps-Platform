# C1: Drift Detection Engine (J26-DS-361)

Owner: Sajivan K (IT23172296). Branch: `c1-sajivan`.

## Layout
```
c1/
  src/profile.py            ReferenceProfile: everything detectors need from the reference, computed once
  src/detectors/            six detectors (one file each) + run_all_detectors()
  src/prediction_stats.py   max-probability confidence (v1.2) + positive-class stats (v1.1)
  scripts/rescore_benchmark.py   rescores the frozen v1.1 benchmark into v1.2
  tests/                    unit tests on synthetic data (python -m pytest c1/tests -q)
```
Models and benchmark CSVs live on Drive, not in git.

## v1.2 signals (same 1,600 conditions and targets as v1.1)
| Signal | Detector | Applied to | Aggregate |
|---|---|---|---|
| ks_mean, ks_max | Kolmogorov-Smirnov | each continuous feature | mean, max |
| chi2_mean, chi2_max | Chi-squared as bias-corrected Cramer's V, balanced table | each categorical feature | mean, max |
| psi_mean, psi_max | Population Stability Index | every feature (reference-quantile bins / categories) | mean, max |
| mmd | RBF MMD^2, fixed bandwidth | all features (standardised + one-hot) | - |
| lsdd | alibi-detect LSDDDrift | all features (standardised + one-hot) | - |
| clf_auc | XGBoost two-sample test, 3-fold AUC | all features | - |
| conf_mean, conf_std, conf_pct_lt_06 | max class probability of the deployed model | batch predictions | - |

Per-feature scores are also saved (`ks__age`, `chi2__sex`, `psi__native_country`, ...) for
`affected_features` and drift-type work. They are not N1 inputs.
v1.1 columns are kept: `v11_ks_score` ... `v11_clf_auc`, and `mean_pos_prob`, `pos_prob_std`,
`pct_pos_prob_lt_06`.

Feature types always come from the feature-importance JSON (`feature_types`), never guessed.
