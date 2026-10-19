# Component 3: Multi-Dataset Acquisition & Integrity Guide
**Project:** J26-DS-361 — Intelligent MLOps Platform  
**Component:** Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
**Date:** October 19, 2026  
**Status:** 3 Verified Benchmarks Available Locally, 2 Pending Authenticated Download

---

## 1. Overview and Candidate Dataset Summary

Component 3 evaluates synthetic data repair across 5 candidate tabular classification datasets. To maintain research integrity and prevent data leakage, every dataset specification includes:
- Source and licensing information.
- Target column and target mapping.
- Numerical and categorical feature lists.
- ID columns and known leakage-prone columns that **must be dropped** before model training.
- Splitting strategy (Stratified vs. Temporal).
- Reproducible SHA-256 integrity checksums.

| Dataset ID | Display Name | Source & License | Status | Size | Split Strategy | Identified Leakage / Dropped Columns |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `uci_adult` | UCI Adult Census Income | UCI Repository / CC BY 4.0 | **VERIFIED** | 26,048 rows | Stratified | `fnlwgt` (census weight), `education` (redundant) |
| `telco_churn` | Telco Customer Churn | IBM / Apache 2.0 | **VERIFIED** | 7,043 rows | Stratified | `customerID` (ID identifier) |
| `bank_marketing`| Bank Marketing (UCI) | UCI Repository / CC BY 4.0 | **VERIFIED** | 41,188 rows | Stratified | `duration` (call duration: post-decision leakage) |
| `home_credit` | Home Credit Default Risk | Kaggle / Competition License | **PENDING** | 307,511 rows | Stratified | `SK_ID_CURR` (ID identifier) |
| `ieee_cis` | IEEE-CIS Fraud Detection | Vesta & IEEE-CIS / Kaggle | **PENDING** | 590,540 rows | Time-Based | `TransactionID` (ID), `TransactionDT` (temporal) |

---

## 2. Verified Local Datasets

### 2.1 UCI Adult Census Income (`uci_adult`)
- **Location:** `shared/uci_adult/data/reference.csv` (Reference) and `test.csv` (Holdout).
- **Task:** Binary classification (Predict whether income exceeds \$50,000/yr).
- **Target:** `target` (`0` = $\le$50K, `1` = >50K).
- **Features (12):**
  - Numerical (5): `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`.
  - Categorical (7): `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`.
- **Integrity SHA-256:** `84cbc88cbfac5567c9d92e592751df1c72f10b77b63f58ea961f7dbca19c1125`

### 2.2 Telco Customer Churn (`telco_churn`)
- **Location:** `component3/data/raw/telco_churn.csv`.
- **Source:** IBM Developer Data Catalog (`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`).
- **Task:** Binary classification (Predict customer account cancellation).
- **Target:** `Churn` (`No` = 0, `Yes` = 1).
- **Features (19):**
  - Numerical (3): `tenure`, `MonthlyCharges`, `TotalCharges`.
  - Categorical (16): `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
- **Dropped Identifier:** `customerID` (prevents model overfitting to customer account hashes).
- **Integrity SHA-256:** Computed upon automated ingestion via `DatasetAcquisitionManager`.

### 2.3 Bank Marketing (`bank_marketing`)
- **Location:** `component3/data/raw/bank_marketing.csv`.
- **Source:** UCI Machine Learning Repository (`https://archive.ics.uci.edu/ml/datasets/bank+marketing`).
- **Task:** Binary classification (Predict customer subscription to term deposit).
- **Target:** `y` (`0` / `no` = 0, `1` / `yes` = 1).
- **Features (19):**
  - Numerical (9): `age`, `campaign`, `pdays`, `previous`, `emp_var_rate`, `cons_price_idx`, `cons_conf_idx`, `euribor3m`, `nr_employed`.
  - Categorical (10): `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`.
- **CRITICAL DATA LEAKAGE RULE:** `duration` (call duration in seconds) **must be dropped**. As noted in UCI documentation, duration is not known before a call is made and heavily correlates with the target. Including it produces synthetic data and model evaluation leakage.

---

## 3. Pending Datasets Requiring Authentication

Due to Kaggle Terms of Service and access controls, competition datasets cannot be retrieved via unauthenticated public URLs. Follow the instructions below to acquire them:

### 3.1 Home Credit Default Risk (`home_credit`)
1. Create a Kaggle API token from `https://www.kaggle.com/settings` (download `kaggle.json`).
2. Accept the competition rules at `https://www.kaggle.com/c/home-credit-default-risk`.
3. Run:
   ```bash
   kaggle competitions download -c home-credit-default-risk -f application_train.csv -p component3/data/raw/
   ```
4. Verify checksum via `DatasetVerifier().verify("home_credit")`.

### 3.2 IEEE-CIS Fraud Detection (`ieee_cis`)
1. Accept the competition rules at `https://www.kaggle.com/c/ieee-fraud-detection`.
2. Run:
   ```bash
   kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p component3/data/raw/
   ```
3. Verify checksum via `DatasetVerifier().verify("ieee_cis")`.
4. **Note:** Temporal split on `TransactionDT` must be used to preserve temporal validity.

---

## 4. Google Drive Cloud Paths for Colab Workflows

For Google Colab execution, datasets can be placed in your Google Drive at:
```
/content/drive/MyDrive/J26-DS-361-MLOps/component3/data/
├── reference.csv                 # uci_adult reference
├── telco_churn.csv              # Telco churn
├── bank_marketing.csv           # Bank marketing
├── home_credit.csv              # Home credit application train (optional)
└── ieee_cis.csv                 # IEEE-CIS transaction (optional)
```
`colab_runner.py` and Notebooks 01–04 automatically load from this directory.
