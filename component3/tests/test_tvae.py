import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from component3.src.generators.tvae_generator import TVAEGenerator


def test_tvae():
    # Small example dataset
    data = pd.DataFrame({
        "age": [22, 25, 30, 35, 40, 45, 50, 55] * 5,
        "income": ["Low", "High", "Low", "High"] * 10,
        "education": ["School", "Degree"] * 20,
    })

    # Use 1 epoch for a quick test
    generator = TVAEGenerator(epochs=1)

    print("Training TVAE...")
    generator.fit(data)

    print("Generating synthetic data...")
    synthetic_data = generator.sample(5)

    assert len(synthetic_data) == 5
    assert list(synthetic_data.columns) == list(data.columns)

    print("\nGenerated synthetic data:")
    print(synthetic_data)
    print("\nTVAE smoke test passed!")


if __name__ == "__main__":
    test_tvae()