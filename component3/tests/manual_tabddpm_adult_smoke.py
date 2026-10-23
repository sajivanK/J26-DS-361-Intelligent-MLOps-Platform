"""Integration smoke test for TabDDPM on real UCI Adult S04 data."""

import numpy as np
import pandas as pd
import torch

from component3.src.contracts.safe_slice_evaluator import (
    extract_affected_slice,
)
from component3.src.core.config_loader import load_config
from component3.src.core.data_loader import load_current_dataset
from component3.src.generators.tabddpm_generator import (
    TabDDPMGenerator,
)
from component3.src.generators.target_coverage import (
    evaluate_target_coverage,
)


def main():
    print("\n=== REAL UCI ADULT TABDDPM TEST ===")
    np.random.seed(42)
    torch.manual_seed(42)
    config = load_config()

    scenario = "S05"
    slice_rule = "age > 0.69 and education_num < -0.03"
    current_df = load_current_dataset(config, scenario)
    affected_slice = extract_affected_slice(
        current_df,
        slice_rule,
    )
    print("Current dataset rows:", len(current_df))
    print("Affected records:", len(affected_slice))
    assert len(affected_slice) == 342

    # Feature preparation and target exclusion
    feature_types = config["dataset"]["feature_types"]
    categorical_columns = [
        col for col, f_type in feature_types.items() if f_type == "categorical"
    ]
    continuous_columns = [
        col for col, f_type in feature_types.items() if f_type == "continuous"
    ]
    feature_columns = list(feature_types.keys())
    target_column = config["dataset"].get("target_column", "target")

    training_data = affected_slice[feature_columns].copy()
    assert target_column not in training_data.columns, "Target column must be excluded."

    print("Training features:", len(feature_columns))
    print("Continuous features:", len(continuous_columns))
    print("Categorical features:", len(categorical_columns))
    print("Target excluded: PASS")

    print("\n=== MODEL INITIALIZATION & TRAINING ===")
    generator = TabDDPMGenerator(
        epochs=1,
        num_timesteps=10,
        discrete_columns=categorical_columns,
    )
    generator.fit(training_data)

    print("\n=== FEATURE PARTITION VALIDATION ===")
    assert generator.num_columns == continuous_columns, "Incorrect continuous feature partition."
    assert generator.cat_columns == categorical_columns, "Incorrect categorical feature partition."
    print("Feature partition: PASS")

    print("\n=== SYNTHETIC DATA GENERATION ===")
    synthetic = generator.sample(10)
    print("\nGenerated synthetic records:")
    print(synthetic.to_string(index=False))

    print("\n=== OUTPUT VALIDATION ===")
    assert synthetic.shape == (10, 12), "Incorrect generated dataset shape."
    assert synthetic.columns.tolist() == feature_columns, "Generated column order does not match training data."
    assert synthetic.isna().sum().sum() == 0, "Generated data contains missing values."
    print("Synthetic dataset shape: PASS")
    print("Feature schema and order: PASS")
    print("Missing-value validation: PASS")

    print("\n=== TARGET COVERAGE EVALUATION ===")
    coverage = evaluate_target_coverage(synthetic, slice_rule)
    print(f"Target coverage: {coverage.matching_rows}/{coverage.total_rows} ({coverage.coverage * 100:.2f}%)")

    # Diagnostic numerical range check
    range_report = generator.validate_numerical_ranges(synthetic)
    print(f"Diagnostic range validation passed: {range_report['passed']}")
    print(f"Diagnostic range violations: {range_report['total_invalid_values']}")

    print("\n=== FINAL RESULT ===")
    print("TABDDPM REAL ADULT SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
