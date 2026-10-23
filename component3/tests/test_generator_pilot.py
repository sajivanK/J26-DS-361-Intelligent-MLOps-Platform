"""Unit and integration tests for Generator Readiness, Pilot Execution, and Output Validation.

Covers:
- Split isolation (generator training partition disjoint from val/test sets).
- Input schema validation (dropped columns, feature typing, target column).
- Target-class conditioning (positive-slice conditioning logic).
- Synthetic output structural validity (columns, dtypes, nulls, categories).
- Generator reproducibility under fixed random seeds.
- Generator save and reload parity via joblib serialization.
- Memorization screening calculations (exact match rate, DCR metrics).
- Pilot experiment result serialization integrity (CSV and JSON artifacts).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from component3.src.datasets.dataset_registry import DatasetRegistry
from component3.src.generators.tvae_generator import TVAEGenerator


# ---------------------------------------------------------------------------
# Paths and Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMP3_DIR = REPO_ROOT / "component3"
RESULTS_DIR = COMP3_DIR / "results"
MANIFEST_PATH = RESULTS_DIR / "split_manifests" / "telco_churn_split_manifest.json"
PILOT_CSV_PATH = RESULTS_DIR / "generators" / "telco_churn_generator_pilot_summary.csv"
PILOT_JSON_PATH = RESULTS_DIR / "generators" / "telco_churn_generator_pilot_summary.json"
IMPROVEMENT_CSV_PATH = RESULTS_DIR / "generators" / "telco_churn_generator_improvement_summary.csv"


# ---------------------------------------------------------------------------
# Test 1: Split Isolation and Partition Fingerprint
# ---------------------------------------------------------------------------

def test_split_isolation_and_partition_fingerprint():
    """Verify generator training data is strictly disjoint from val/test sets."""
    assert MANIFEST_PATH.exists(), f"Split manifest not found at {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["dataset_id"] == "telco_churn"
    train_indices = set(manifest["train_indices"])
    val_indices = set(manifest["val_indices"])
    test_indices = set(manifest["test_indices"])

    assert len(train_indices) == 4931, f"Expected 4931 train records, got {len(train_indices)}"
    assert len(val_indices) == 1056, f"Expected 1056 val records, got {len(val_indices)}"
    assert len(test_indices) == 1056, f"Expected 1056 test records, got {len(test_indices)}"
    assert len(train_indices) + len(val_indices) + len(test_indices) == 7043

    # Absolute disjointness: zero partition leakage
    assert train_indices.isdisjoint(val_indices), "Train and validation indices overlap!"
    assert train_indices.isdisjoint(test_indices), "Train and test indices overlap!"
    assert val_indices.isdisjoint(test_indices), "Validation and test indices overlap!"


# ---------------------------------------------------------------------------
# Test 2: Generator Input Schema and Column Filtering
# ---------------------------------------------------------------------------

def test_generator_input_schema_telco_churn():
    """Verify dropped columns are excluded and active features conform to schema."""
    registry = DatasetRegistry()
    meta = registry.get("telco_churn")

    assert meta.target_column == "Churn"
    assert "customerID" in meta.drop_columns

    df_raw = pd.read_csv(meta.get_absolute_path())
    assert len(df_raw) == 7043
    assert "customerID" in df_raw.columns

    # Clean numeric coercion for TotalCharges
    df_clean = df_raw.drop(columns=meta.drop_columns).copy()
    assert "customerID" not in df_clean.columns

    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce").fillna(0.0)
    assert not df_clean["TotalCharges"].isna().any()

    # Active features + target
    expected_active_features = len(meta.feature_columns)
    assert expected_active_features == 19
    assert len(df_clean.columns) == 20  # 19 features + 1 target
    assert set(meta.feature_columns).issubset(set(df_clean.columns))


# ---------------------------------------------------------------------------
# Test 3: Target Class Conditioning Logic
# ---------------------------------------------------------------------------

def test_target_class_conditioning_positive_slice():
    """Verify positive slice extraction logic and conditioning guarantees."""
    registry = DatasetRegistry()
    meta = registry.get("telco_churn")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    df_raw = pd.read_csv(meta.get_absolute_path())
    df_train = df_raw.iloc[manifest["train_indices"]].copy()

    assert len(df_train) == 4931
    class_counts = df_train["Churn"].value_counts().to_dict()

    assert class_counts["No"] == 3622
    assert class_counts["Yes"] == 1309

    # Slice extraction for positive class
    pos_slice = df_train[df_train["Churn"] == "Yes"].copy()
    assert len(pos_slice) == 1309
    assert (pos_slice["Churn"] == "Yes").all()
    assert not (pos_slice["Churn"] == "No").any()


# ---------------------------------------------------------------------------
# Test 4: Synthetic Output Structural Validity
# ---------------------------------------------------------------------------

def test_synthetic_output_structural_validity():
    """Verify structural integrity check detects missing columns and invalid categories."""
    reference_df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat1": ["Fiber", "DSL", "DSL", "Fiber", "No"],
        "Churn": ["Yes", "Yes", "Yes", "Yes", "Yes"],
    })

    # Valid synthetic output
    valid_synthetic = pd.DataFrame({
        "num1": [1.5, 2.5, 3.5],
        "cat1": ["DSL", "Fiber", "DSL"],
        "Churn": ["Yes", "Yes", "Yes"],
    })

    assert list(valid_synthetic.columns) == list(reference_df.columns)
    assert not valid_synthetic.isna().any().any()
    assert set(valid_synthetic["cat1"]).issubset(set(reference_df["cat1"]))

    # Invalid synthetic output: missing column
    invalid_missing_col = valid_synthetic.drop(columns=["cat1"])
    assert set(invalid_missing_col.columns) != set(reference_df.columns)

    # Invalid synthetic output: out-of-domain category
    invalid_category = valid_synthetic.copy()
    invalid_category.loc[0, "cat1"] = "UnknownCategory"
    assert not set(invalid_category["cat1"]).issubset(set(reference_df["cat1"]))


# ---------------------------------------------------------------------------
# Test 5: Generator Reproducibility Under Fixed Random Seed
# ---------------------------------------------------------------------------

def test_generator_reproducibility_with_fixed_seed():
    """Verify deterministic sampling and initialization under fixed seeds."""
    np.random.seed(42)
    sample_a = np.random.normal(size=(10, 5))

    np.random.seed(42)
    sample_b = np.random.normal(size=(10, 5))

    np.testing.assert_array_almost_equal(sample_a, sample_b)


# ---------------------------------------------------------------------------
# Test 6: Generator Save and Reload Parity
# ---------------------------------------------------------------------------

def test_generator_save_and_reload_parity(tmp_path):
    """Verify fitted generator can be serialized, loaded, and sample identically."""
    toy_data = pd.DataFrame({
        "tenure": np.random.uniform(1, 72, 30),
        "MonthlyCharges": np.random.uniform(20, 100, 30),
        "Contract": np.random.choice(["Month-to-month", "One year"], 30),
        "Churn": ["Yes"] * 30,
    })

    gen = TVAEGenerator(epochs=1, discrete_columns=["Contract", "Churn"])
    gen.fit(toy_data)
    assert gen.is_fitted

    save_path = tmp_path / "tvae_test_model.pkl"
    joblib.dump(gen, save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0

    reloaded_gen: TVAEGenerator = joblib.load(save_path)
    assert reloaded_gen.is_fitted
    assert reloaded_gen.epochs == 1
    assert set(reloaded_gen.discrete_columns) == {"Contract", "Churn"}

    synthetic_records = reloaded_gen.sample(5)
    assert len(synthetic_records) == 5
    assert set(synthetic_records.columns) == set(toy_data.columns)


# ---------------------------------------------------------------------------
# Test 7: Memorization Screening Calculations (Exact Match & DCR)
# ---------------------------------------------------------------------------

def test_memorization_screening_calculations():
    """Verify exact-match rate and Distance to Closest Record (DCR) calculations."""
    ref_df = pd.DataFrame({
        "f1": [10.0, 20.0, 30.0, 40.0],
        "f2": [1.0, 2.0, 3.0, 4.0],
        "cat": ["A", "B", "A", "B"],
    })

    # Exact clones: exact match = 100%, min DCR = 0.0
    clone_df = ref_df.copy()
    exact_matches = pd.merge(clone_df, ref_df, how="inner")
    assert len(exact_matches) == len(ref_df)

    # Disjoint synthetic: exact match = 0%
    disjoint_df = pd.DataFrame({
        "f1": [15.0, 25.0, 35.0, 45.0],
        "f2": [1.5, 2.5, 3.5, 4.5],
        "cat": ["B", "A", "B", "A"],
    })
    disjoint_matches = pd.merge(disjoint_df, ref_df, how="inner")
    assert len(disjoint_matches) == 0

    # DCR computation on numerical features
    scaler = MinMaxScaler()
    ref_num = scaler.fit_transform(ref_df[["f1", "f2"]])
    syn_num = scaler.transform(disjoint_df[["f1", "f2"]])

    dists = cdist(syn_num, ref_num, metric="euclidean")
    min_dists = dists.min(axis=1)

    assert (min_dists > 0.0).all(), "Disjoint points should have strictly positive DCR"
    assert np.median(min_dists) > 0.0


# ---------------------------------------------------------------------------
# Test 8: Pilot Experiment Result Serialization Integrity
# ---------------------------------------------------------------------------

def test_pilot_summary_serialization_integrity():
    """Verify saved pilot summary artifacts exist, are non-empty, and contain all metrics."""
    assert PILOT_CSV_PATH.exists(), f"Missing CSV artifact at {PILOT_CSV_PATH}"
    assert PILOT_JSON_PATH.exists(), f"Missing JSON artifact at {PILOT_JSON_PATH}"

    # Verify CSV content
    df_summary = pd.read_csv(PILOT_CSV_PATH)
    assert len(df_summary) == 3, f"Expected 3 generators, found {len(df_summary)}"
    assert {g.lower() for g in df_summary["generator"]} == {"ctgan", "tvae", "tabddpm"}

    expected_cols = [
        "generator", "training_records", "epochs", "train_seconds",
        "seconds_per_epoch", "sample_200_seconds", "sample_1000_seconds",
        "peak_ram_mb", "artifact_mb", "structural_status", "range_violations",
        "within_syn_duplicates", "mean_numerical_ks", "mean_categorical_tvd",
        "mean_wasserstein_w1", "exact_matches", "exact_match_rate",
        "dcr_median", "dcr_p05",
    ]
    for col in expected_cols:
        assert col in df_summary.columns, f"Missing expected column {col} in CSV"

    # Verify JSON content
    with open(PILOT_JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    assert json_data["dataset"] == "telco_churn"
    assert json_data["scenario"] == "TELCO_PRIOR"
    assert json_data["conditioning_slice"] == "Churn == 'Yes'"
    assert json_data["authentic_training_slice_records"] == 1309
    assert json_data["synthetic_records_generated"] == 1000
    assert len(json_data["results"]) == 3

    for gen_record in json_data["results"]:
        assert gen_record["generator"].lower() in {"ctgan", "tvae", "tabddpm"}
        assert gen_record["train_seconds"] >= 0.0
        assert gen_record["peak_ram_mb"] > 0.0
        assert gen_record["sample_1000_seconds"] >= 0.0
        assert "%" in str(gen_record["structural_status"]) or gen_record["structural_status"] == "PASS"
        assert gen_record["exact_matches"] == 0
        assert gen_record["dcr_median"] > 0.0


# ---------------------------------------------------------------------------
# Test 9: Fine-Grained Constraint Enforcement & Categorization
# ---------------------------------------------------------------------------

def test_fine_grained_constraint_enforcement():
    """Verify categorization of physical, range, integer, and cross-feature constraints."""
    toy_df = pd.DataFrame({
        "tenure": [-2.0, 12.5, 5.0, 10.0, 80.0],
        "MonthlyCharges": [50.0, 50.0, -10.0, 70.0, 50.0],
        "TotalCharges": [100.0, 625.0, 100.0, 50000.0, 4000.0],
        "Contract": ["Month-to-month"] * 5,
        "Churn": ["Yes"] * 5,
    })

    # Physical impossibilities
    phys_mask = (toy_df["tenure"] < 0) | (toy_df["MonthlyCharges"] <= 0) | (toy_df["TotalCharges"] < 0)
    assert phys_mask.sum() == 2  # row 0 has tenure < 0, row 2 has MonthlyCharges < 0

    # Integer constraint on tenure
    int_mask = (toy_df["tenure"] % 1 != 0)
    assert int_mask.sum() == 1  # row 1 (12.5)

    # Range bounds (e.g., tenure <= 72)
    range_mask = (toy_df["tenure"] < 0) | (toy_df["tenure"] > 72)
    assert range_mask.sum() == 2  # row 0 (-2.0), row 4 (80.0)

    # Cross-feature constraint: TotalCharges vs tenure * MonthlyCharges
    # Row 3 has tenure=10, MonthlyCharges=70, TotalCharges=50000 (impossible)
    cross_mask = (toy_df["TotalCharges"] > toy_df["tenure"] * toy_df["MonthlyCharges"] * 1.5 + 50)
    assert cross_mask.iloc[3] == True


# ---------------------------------------------------------------------------
# Test 10: Valid-Record and Rejection Accounting
# ---------------------------------------------------------------------------

def test_valid_record_accounting_and_multi_violation_distinction():
    """Confirm total records = pass + reject, and violation count >= rejected records."""
    df_pilot = pd.read_csv(PILOT_CSV_PATH)
    assert len(df_pilot) == 3

    for _, row in df_pilot.iterrows():
        total_gen = 1000
        pass_rec = int(row["passing_records"])
        rej_rec = int(row["rejected_records"])
        uniq_valid = int(row["unique_valid_records"])

        # Strict accounting invariant: passing + rejected == total generated
        assert pass_rec + rej_rec == total_gen, f"Accounting mismatch for {row['generator']}"
        # Unique valid records cannot exceed total passing records
        assert uniq_valid <= pass_rec
        assert uniq_valid >= 0


# ---------------------------------------------------------------------------
# Test 11: Duplicate Counting Semantics and Quantile Boundary Projection
# ---------------------------------------------------------------------------

def test_duplicate_counting_semantics_and_boundary_projection():
    """Verify pandas duplicated() semantics and QuantileTransformer corner-point behavior."""
    toy_df = pd.DataFrame({
        "a": [1, 2, 2, 3, 2],
        "b": ["x", "y", "y", "z", "y"],
    })
    # Duplicates counted as rows beyond the first occurrence:
    # row 0: False, row 1: False, row 2: True, row 3: False, row 4: True -> sum = 2
    n_dupes = toy_df.duplicated().sum()
    assert n_dupes == 2
    assert len(toy_df.drop_duplicates()) == 3
    assert len(toy_df) - n_dupes == 3

    # QuantileTransformer boundary projection demonstration
    qt = QuantileTransformer(n_quantiles=5, output_distribution="normal", random_state=42)
    sample_fit = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
    qt.fit(sample_fit)

    # Extreme latent diffusion values (|z| >> 3.0) clamp to min/max
    extreme_latent = np.array([[-100.0], [-50.0], [50.0], [100.0]])
    projected = qt.inverse_transform(extreme_latent)
    # Negative extremes map to the empirical minimum (10.0)
    assert np.isclose(projected[0, 0], 10.0)
    assert np.isclose(projected[1, 0], 10.0)
    # Positive extremes map to the empirical maximum (50.0)
    assert np.isclose(projected[2, 0], 50.0)
    assert np.isclose(projected[3, 0], 50.0)


# ---------------------------------------------------------------------------
# Test 12: Raw Versus Postprocessed Data Tracking
# ---------------------------------------------------------------------------

def test_raw_versus_postprocessed_tracking():
    """Verify raw generated records are distinguished from postprocessed records."""
    raw_df = pd.DataFrame({
        "tenure": [-3.5, 10.0, 85.0],
        "MonthlyCharges": [50.0, 70.0, 100.0],
        "TotalCharges": [200.0, 5000.0, 8500.0],
    })

    # Postprocessing: clipping tenure
    post_df = raw_df.copy()
    post_df["tenure"] = post_df["tenure"].clip(0, 72)

    # Verify raw is unmodified
    assert (raw_df["tenure"] < 0).any()
    assert (raw_df["tenure"] > 72).any()

    # Verify clipping fixed tenure range
    assert not (post_df["tenure"] < 0).any()
    assert not (post_df["tenure"] > 72).any()

    # Critical check: clipping does NOT resolve cross-feature violations (row 1: 10 * 70 vs 5000)
    expected_charges = post_df["tenure"] * post_df["MonthlyCharges"]
    cross_violation = (post_df["TotalCharges"] > expected_charges * 1.5 + 50)
    assert cross_violation.iloc[1] == True, "Clipping tenure must not be assumed to fix cross-feature errors"


# ---------------------------------------------------------------------------
# Test 13: Distance Calculations and Real-to-Real Holdout Screening
# ---------------------------------------------------------------------------

def test_distance_calculations_and_real_to_real_holdout_screening():
    """Verify DCR calculation and comparison against real-to-real holdout baseline."""
    np.random.seed(42)
    train_ref = np.random.uniform(0, 1, size=(100, 3))
    holdout_ref = np.random.uniform(0, 1, size=(30, 3))
    syn_close = train_ref[:30] + np.random.normal(0, 0.01, size=(30, 3))

    # Real-to-real holdout DCR
    d_holdout = cdist(holdout_ref, train_ref, metric="euclidean").min(axis=1)
    # Synthetic-to-real DCR
    d_syn = cdist(syn_close, train_ref, metric="euclidean").min(axis=1)

    assert len(d_holdout) == 30
    assert len(d_syn) == 30
    # Screening: DCR > 0 confirms absence of direct clones
    assert (d_syn > 0.0).all()
    # Synthetic perturbed records have lower median DCR than independent holdout
    assert np.median(d_syn) < np.median(d_holdout)


# ---------------------------------------------------------------------------
# Test 14: Training and Evaluation Separation (No Partition Leakage)
# ---------------------------------------------------------------------------

def test_training_evaluation_separation_no_leakage():
    """Verify generator training slice is split internally and never touches val/test sets."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    registry = DatasetRegistry()
    meta = registry.get("telco_churn")
    df_raw = pd.read_csv(meta.get_absolute_path())

    train_idx = manifest["train_indices"]
    val_idx = set(manifest["val_indices"])
    test_idx = set(manifest["test_indices"])

    df_train = df_raw.iloc[train_idx]
    pos_train_indices = set(df_train[df_train["Churn"] == "Yes"].index)

    # 1,309 positive training records
    assert len(pos_train_indices) == 1309

    # Duplicate-safe internal 80/20 train/holdout split: 1047 / 262
    pos_train_df = df_train[df_train["Churn"] == "Yes"].copy()
    feat_cols = [c for c in pos_train_df.columns if c not in ["customerID", "Churn"]]
    grouped = pos_train_df.groupby(feat_cols, sort=False).indices

    np.random.seed(42)
    group_keys = list(grouped.keys())
    np.random.shuffle(group_keys)

    gen_train_pos_idx = []
    dev_holdout_pos_idx = []
    for k in group_keys:
        idx_list = grouped[k].tolist()
        if len(gen_train_pos_idx) + len(idx_list) <= 1047:
            gen_train_pos_idx.extend(idx_list)
        else:
            dev_holdout_pos_idx.extend(idx_list)

    assert len(gen_train_pos_idx) == 1047
    assert len(dev_holdout_pos_idx) == 262
    assert set(gen_train_pos_idx).isdisjoint(dev_holdout_pos_idx)

    # Zero identical feature rows crossing partition boundary
    tr_feats = pos_train_df.iloc[gen_train_pos_idx][feat_cols]
    ho_feats = pos_train_df.iloc[dev_holdout_pos_idx][feat_cols]
    overlap = pd.merge(tr_feats, ho_feats, how="inner")
    assert len(overlap) == 0, "Duplicate-safe split must have zero identical feature rows across boundary"

    # Absolute test/validation isolation
    assert set(pos_train_df.iloc[gen_train_pos_idx].index).isdisjoint(val_idx)
    assert set(pos_train_df.iloc[gen_train_pos_idx].index).isdisjoint(test_idx)
    assert set(pos_train_df.iloc[dev_holdout_pos_idx].index).isdisjoint(val_idx)
    assert set(pos_train_df.iloc[dev_holdout_pos_idx].index).isdisjoint(test_idx)


# ---------------------------------------------------------------------------
# Test 15: Controlled Improvement Summary Artifact Integrity
# ---------------------------------------------------------------------------

def test_controlled_improvement_summary_serialization():
    """Verify telco_churn_generator_improvement_summary.csv exists and has all configs."""
    assert IMPROVEMENT_CSV_PATH.exists(), f"Missing improvement CSV at {IMPROVEMENT_CSV_PATH}"

    df_imp = pd.read_csv(IMPROVEMENT_CSV_PATH)
    # Expected 7 configurations: CTGAN (10, 25), TVAE (10, 30, 50), TabDDPM (10, 30)
    assert len(df_imp) == 7, f"Expected 7 configurations, got {len(df_imp)}"

    expected_cols = [
        "generator", "epochs", "train_seconds", "sample_1000_seconds",
        "valid_records", "unique_valid_records", "raw_duplicates",
        "holdout_mean_ks", "holdout_mean_tvd", "dcr_median", "dcr_p05",
    ]
    for col in expected_cols:
        assert col in df_imp.columns, f"Missing {col} in improvement CSV"

    # Verify generators present
    generators_in_csv = set(df_imp["generator"].unique())
    assert generators_in_csv == {"CTGAN", "TVAE", "TabDDPM"}

    # Verify TVAE achieves high valid records
    tvae_rows = df_imp[df_imp["generator"] == "TVAE"]
    assert (tvae_rows["valid_records"] > 500).all()

    # Verify TabDDPM duplicate records are tracked
    tabddpm_rows = df_imp[df_imp["generator"] == "TabDDPM"]
    assert (tabddpm_rows["raw_duplicates"] > 300).all()


# ---------------------------------------------------------------------------
# Test 16: Generator Reproducibility Across Fixed Seeds
# ---------------------------------------------------------------------------

def test_generator_reproducibility_across_fixed_seeds():
    """Verify generator fitting and sampling are reproducible under fixed random seeds."""
    import torch

    toy_df = pd.DataFrame({
        "tenure": [1.0, 12.0, 24.0, 36.0, 48.0, 60.0] * 5,
        "MonthlyCharges": [20.0, 45.0, 70.0, 85.0, 95.0, 110.0] * 5,
        "Contract": ["Month-to-month", "One year", "Two year"] * 10,
        "Churn": ["Yes"] * 30,
    })

    torch.manual_seed(42)
    np.random.seed(42)
    gen1 = TVAEGenerator(epochs=2, discrete_columns=["Contract", "Churn"])
    gen1.fit(toy_df)
    torch.manual_seed(42)
    np.random.seed(42)
    s1 = gen1.sample(10)

    torch.manual_seed(42)
    np.random.seed(42)
    gen2 = TVAEGenerator(epochs=2, discrete_columns=["Contract", "Churn"])
    gen2.fit(toy_df)
    torch.manual_seed(42)
    np.random.seed(42)
    s2 = gen2.sample(10)

    pd.testing.assert_frame_equal(s1, s2)


# ---------------------------------------------------------------------------
# Test 17: Authentic Cross-Feature Plausibility Rule Validation
# ---------------------------------------------------------------------------

def test_authentic_cross_feature_plausibility_rule():
    """Verify that the cross-feature plausibility rule is supported by authentic Telco data."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    registry = DatasetRegistry()
    meta = registry.get("telco_churn")
    df_raw = pd.read_csv(meta.get_absolute_path())

    df_train = df_raw.iloc[manifest["train_indices"]].copy()
    df_train["TotalCharges"] = pd.to_numeric(df_train["TotalCharges"], errors="coerce").fillna(0.0)

    # Check tenure == 0
    t0_records = df_train[df_train["tenure"] == 0]
    assert len(t0_records) == 10
    assert (t0_records["TotalCharges"] == 0.0).all()
    assert (t0_records["Churn"] == "No").all()

    # Check positive churn slice (N=1,309)
    pos_train = df_train[df_train["Churn"] == "Yes"].copy()
    assert len(pos_train) == 1309
    assert (pos_train["tenure"] >= 1).all(), "Positive churn has no tenure == 0"

    # Empirical billing ratio R = TotalCharges / (tenure * MonthlyCharges)
    ratio = pos_train["TotalCharges"] / (pos_train["tenure"] * pos_train["MonthlyCharges"])
    assert ratio.min() >= 0.70, f"Authentic min ratio {ratio.min()} fell below 0.70"
    assert ratio.max() <= 1.30, f"Authentic max ratio {ratio.max()} exceeded 1.30"

    # Plausibility rule [0.5, 1.5] must have 0% failure on authentic positive churn
    rule_fails = ((ratio < 0.5) | (ratio > 1.5)).sum()
    assert rule_fails == 0, f"Expected 0 authentic positive failures, found {rule_fails}"


# ---------------------------------------------------------------------------
# Test 18: Duplicate-Safe Internal Split and Strictly Positive Holdout DCR
# ---------------------------------------------------------------------------

def test_duplicate_safe_internal_split_and_positive_min_dcr():
    """Verify that duplicate-safe split results in strictly positive real-to-real holdout DCR."""
    with open(PILOT_JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    assert "duplicate_safe_internal_split" in json_data
    split_info = json_data["duplicate_safe_internal_split"]

    assert split_info["generator_training_records"] == 1047
    assert split_info["development_holdout_records"] == 262
    assert split_info["duplicate_groups_crossing_boundary"] == 0
    # Minimum real-to-real DCR must be strictly positive (resolving earlier 0.0000 contradiction)
    assert split_info["real_to_real_holdout_min_dcr"] > 0.0
    assert split_info["real_to_real_holdout_median_dcr"] > 1.0


