# Sprint 05 — Task 10: Repair Experiment Dataset Creation

## 1. Project Information

Project: Intelligent MLOps Platform  
Group: J26-DS-361  
Component: Component 3 — Adaptive Intelligent Synthetic Data Repair Engine  
Sprint: 05 — Validation Engine and Repair Experiments  
Task: 10 — Create Repair Experiment Dataset  

Date: October 18, 2026  

## 2. Objective

Aggregate, structure, and format multi-generator, multi-quantity experimental evidence into a
standardized machine-readable research dataset (`repair_experiments_raw.csv` and `repair_experiments_clean.csv`)
for training the **Intelligent Repair Prescription Engine** in Sprint 06.

## 3. Implemented Files

- `component3/src/experiments/build_repair_dataset.py`
- `component3/tests/test_repair_dataset.py`
- `component3/data/repair_experiments_raw.csv` (15 records, 29 schema features)
- `component3/data/repair_experiments_clean.csv` (15 verified records)
- `component3/data/repair_experiments_metadata.json` (audit metadata)

## 4. Dataset Schema & Data Dictionary

| Column Name | Data Type | Source | Role in Sprint 06 Prescription Model |
|---|---|---|---|
| `experiment_id` | String | Experiment Provenance | Unique experiment primary key |
| `dataset_id` | String | Data Context | Reference benchmark identifier (`uci_adult`) |
| `drift_scenario` | String | Scenario ID | Drift identifier (e.g. `S04`, `S01`, `S03`) |
| `drift_type` | String | C1 Evidence | Pre-repair condition feature (`marginal_single`, `covariate_shift`) |
| `drift_severity` | String | C1 Evidence | Severity tier (`high`, `moderate`, `low`, `none`) |
| `affected_features` | String | C1/C2 Evidence | Targeted feature name (e.g. `race`, `age`, `marital_status`) |
| `slice_support` | Float | C2 Diagnosis | Pre-repair population ratio $[0.0, 1.0]$ |
| `diagnosis_stability` | Float | C2 Diagnosis | Root-cause confidence score $[0.0, 1.0]$ |
| `slice_rule` | String | C2 Diagnosis | Safe slice parsing expression |
| `generator` | String | C3 Candidate | Prescription candidate (`CTGAN`, `TVAE`, `TABDDPM`) |
| `epochs` | Integer | Hyperparameter | Training iterations |
| `repair_quantity` | Integer | Prescription Candidate | Number of requested synthetic records (200, 500, 1000, 2000) |
| `generated_quantity` | Integer | Generation Audit | Actually produced synthetic count |
| `seed` | Integer | Experiment Parameter | Random seed used for generation (e.g. 42) |
| `label_authorized` | Boolean | Governance Authorization | Flag indicating whether synthetic labeling was permitted |
| `label_source` | String | Provenance Tracking | Method used for synthetic labels (`teacher_pseudo_label` or `unlabeled`) |
| `target_coverage` | Float | Stage 1 Validation | Slice targeting efficiency $[0.0, 1.0]$ |
| `fidelity_score` | Float | Stage 2 Validation | Composite statistical fidelity $[0.0, 1.0]$ |
| `privacy_status` | String | Stage 3 Validation | Distance-to-closest-record gate (`PASS` / `FAIL`) |
| `dcr_p05` | Float | Stage 3 Validation | 5th percentile nearest neighbor distance |
| `exact_match_count` | Integer | Stage 3 Validation | Number of exact reference copies (must be 0) |
| `utility_score` | Float | Stage 4 Validation | Offline downstream utility $[0.0, 1.0]$ |
| `delta_overall_f1` | Float | Retraining Outcome | Overall model F1 score delta |
| `delta_affected_slice_f1` | Float | Retraining Outcome | Target slice F1 score delta |
| `delta_non_affected_f1` | Float | Retraining Outcome | Non-affected slice F1 score delta |
| `training_seconds` | Float | Resource Metric | Generator compute training cost |
| `sampling_seconds` | Float | Resource Metric | Generator sampling inference cost |
| `total_seconds` | Float | Resource Metric | End-to-end pipeline latency |
| `decision` | String | Acceptance Policy | Validation decision (`ACCEPT`, `REVIEW`, `REJECT`) |
| `package_status` | String | C4 Contract | Package release flag (`APPROVED`, `REJECTED`) |
| `prescriptive_viability_label` | Integer | Prescriptive Target | Binary meta-learner training target (1 = Viable Repair Candidate, 0 = Non-Viable) |

## 5. Prescriptive Viability Target Definition

To train the Sprint 06 prescriptive classifier or ranker, `prescriptive_viability_label` is formally defined to separate algorithmic candidate viability from strict production deployment approval (`package_status`):
$$\text{Viable} = \mathbb{I}(\text{Coverage} \ge 0.80 \land \text{Privacy} = \text{PASS} \land \Delta F1_{aff} \ge -0.035 \land \text{Fidelity} \ge 0.70)$$

Where:
- `package_status` (`APPROVED`/`REJECTED`) enforces zero-tolerance production deployment gates (where all stages must pass without review flags).
- `prescriptive_viability_label` (`1`/`0`) designates whether a generative recipe produced a structurally sound, private, non-degrading, and statistically viable candidate repair for meta-learner ranking.

Current Distribution:
- Viable (`1`): 11 experiments (73.3%)
- Non-Viable (`0`): 4 experiments (26.7%)

## 6. Conclusion

Sprint 05 — Task 10 successfully compiled 15 rich multi-generator, multi-quantity experimental records with 29 attributes, fully preparing the training dataset for Sprint 06.
Proceeding to **Task 11 Documentation**.
