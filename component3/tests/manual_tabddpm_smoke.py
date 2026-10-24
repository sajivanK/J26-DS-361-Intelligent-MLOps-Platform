"""Manual smoke test for TabDDPM training and sampling."""

import numpy as np
import pandas as pd

from component3.src.generators.tabddpm_generator import TabDDPMGenerator


def main():
    print("=== DATASET PREPARATION ===")
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "age": [20.0 + i for i in range(60)],
            "hours_per_week": [30.0 + (i % 20) for i in range(60)],
            "race": [i % 3 for i in range(60)],
            "sex": [i % 2 for i in range(60)],
        }
    )
    original_data = data.copy(deep=True)
    print(f"Dataset shape: {data.shape}")

    print("\n=== MODEL INITIALIZATION & TRAINING ===")
    generator = TabDDPMGenerator(
        epochs=1,
        num_timesteps=10,
        discrete_columns=["race", "sex"],
    )
    generator.fit(data)

    print("\n=== METADATA & PARTITION VALIDATION ===")
    assert generator.num_columns == [
        "age",
        "hours_per_week",
    ], "Incorrect numerical feature partition."
    assert generator.cat_columns == [
        "race",
        "sex",
    ], "Incorrect categorical feature partition."
    assert generator.num_classes.tolist() == [
        3,
        2,
    ], "Incorrect categorical class counts."
    print("Feature partition: PASS")

    # Verify original input was not modified.
    pd.testing.assert_frame_equal(data, original_data)
    print("Original dataset unchanged: PASS")

    print("\n=== SYNTHETIC DATA GENERATION ===")
    synthetic = generator.sample(10)
    print("\nGenerated synthetic records:")
    print(synthetic.to_string(index=False))

    print("\n=== OUTPUT VALIDATION ===")
    assert synthetic.shape == (10, 4), (
        "Incorrect generated dataset shape."
    )
    assert synthetic.columns.tolist() == data.columns.tolist(), (
        "Generated column order does not match training data."
    )
    assert synthetic.isna().sum().sum() == 0, (
        "Generated data contains missing values."
    )
    assert set(synthetic["race"]).issubset({0, 1, 2}), (
        "Generated race values are outside known categories."
    )
    assert set(synthetic["sex"]).issubset({0, 1}), (
        "Generated sex values are outside known categories."
    )
    assert np.isfinite(
        synthetic[["age", "hours_per_week"]].to_numpy()
    ).all(), "Generated continuous values are not finite."
    print("Synthetic dataset shape: PASS")
    print("Feature schema and order: PASS")
    print("Missing-value validation: PASS")
    print("Categorical domain validation: PASS")
    print("Continuous numerical values: PASS")

    print("\n=== FINAL RESULT ===")
    print("TABDDPM REAL TRAINING SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
