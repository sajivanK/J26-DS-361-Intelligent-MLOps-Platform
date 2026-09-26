"""
UCI Adult Census Income - Preprocessing Pipeline
J26-DS-361 | Shared Artifact | Maintained by Sajivan K (IT23172296)

IMPORTANT:
- All transforms (LabelEncoder, StandardScaler) are fitted on REFERENCE data only
- Apply the same fitted transforms to production/drifted data
- Never refit on production data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ── COLUMN DEFINITIONS ──────────────────────────────────────────────────────

COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

CONTINUOUS_FEATURES = [
    'age', 'fnlwgt', 'education_num',
    'capital_gain', 'capital_loss', 'hours_per_week'
]

CATEGORICAL_FEATURES = [
    'workclass', 'education', 'marital_status', 'occupation',
    'relationship', 'race', 'sex', 'native_country'
]

TARGET_COLUMN = 'income'

# Drop fnlwgt - census weight, not a real feature for prediction
DROP_COLUMNS = ['fnlwgt', 'education']  # education_num already encodes education


def load_raw_data(filepath):
    """
    Load raw UCI Adult CSV file.
    Handles both the version with header and without header.
    """
    try:
        df = pd.read_csv(filepath, header=0)
        if df.shape[1] != 15:
            raise ValueError("Wrong number of columns with header=0")
    except Exception:
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)

    # Strip whitespace from all string columns
    df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    """Quick exploration check before preprocessing."""
    print("\n=== DATA EXPLORATION ===")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts(normalize=True).round(3))
    print(f"\nMissing values (marked as '?'):")
    for col in df.columns:
        n_missing = (df[col] == '?').sum()
        if n_missing > 0:
            print(f"  {col}: {n_missing:,} ({n_missing/len(df)*100:.1f}%)")
    print(f"\nContinuous features stats:")
    print(df[CONTINUOUS_FEATURES].describe().round(2))


def handle_missing_values(df):
    """
    Replace '?' with NaN and impute.
    Strategy: mode imputation for categorical features.
    """
    df = df.replace('?', np.nan)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(mode_val)
            if n_filled > 0:
                print(f"  Filled {n_filled:,} missing values in '{col}' with '{mode_val}'")

    return df


def encode_target(df):
    """
    Encode target: <=50K → 0, >50K → 1
    Handles variations like '<=50K.' and '>50K.'
    """
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
        lambda x: 1 if '>50K' in str(x) else 0
    )
    print(f"\nTarget encoded: 0 = <=50K, 1 = >50K")
    print(f"Class balance: {df[TARGET_COLUMN].value_counts(normalize=True).round(3).to_dict()}")
    return df


class UCIAdultPreprocessor:
    """
    Stateful preprocessor - fit on reference data, transform any data.

    Usage:
        preprocessor = UCIAdultPreprocessor()
        X_ref, y_ref = preprocessor.fit_transform(reference_df)
        X_prod, y_prod = preprocessor.transform(production_df)

    Save/Load:
        preprocessor.save('shared/uci_adult/models/preprocessor.pkl')
        preprocessor = UCIAdultPreprocessor.load('shared/uci_adult/models/preprocessor.pkl')
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False

    def fit_transform(self, df):
        """Fit on reference data and transform it."""
        df = df.copy()

        # Drop columns not used as features
        cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Separate target
        y = df[TARGET_COLUMN].values
        df = df.drop(columns=[TARGET_COLUMN])

        # Label encode categoricals
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col not in DROP_COLUMNS:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Scale continuous features
        cont_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
        df[cont_cols] = self.scaler.fit_transform(df[cont_cols])

        self.feature_columns = list(df.columns)
        self.is_fitted = True

        print(f"\nPreprocessor fitted on {len(df):,} samples")
        print(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns}")

        return df, y

    def transform(self, df):
        """Transform new data using fitted encoders and scaler."""
        assert self.is_fitted, "Preprocessor must be fitted before transform"

        df = df.copy()

        # Drop columns not used as features
        cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Separate target if present
        y = None
        if TARGET_COLUMN in df.columns:
            y = df[TARGET_COLUMN].values
            df = df.drop(columns=[TARGET_COLUMN])

        # Label encode categoricals using fitted encoders
        for col, le in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen labels
                known_classes = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        # Scale continuous features using fitted scaler
        cont_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
        df[cont_cols] = self.scaler.transform(df[cont_cols])

        # Ensure column order matches
        df = df[self.feature_columns]

        return df, y

    def save(self, path):
        joblib.dump(self, path)
        print(f"Preprocessor saved to: {path}")

    @staticmethod
    def load(path):
        preprocessor = joblib.load(path)
        print(f"Preprocessor loaded from: {path}")
        return preprocessor


def run_preprocessing_pipeline(raw_data_path, save_dir, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline for UCI Adult.
    Saves reference.csv, test.csv, and preprocessor.pkl

    Args:
        raw_data_path: path to raw adult.csv or adult.data file
        save_dir:      directory to save outputs
        test_size:     fraction of data for test split
        random_state:  random seed

    Returns:
        X_ref, y_ref, X_test, y_test, preprocessor
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)

    print("=" * 60)
    print("UCI ADULT PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/6] Loading raw data...")
    df = load_raw_data(raw_data_path)

    # Step 2: Explore
    print("\n[2/6] Exploring data...")
    explore_data(df)

    # Step 3: Handle missing values
    print("\n[3/6] Handling missing values...")
    df = handle_missing_values(df)

    # Step 4: Encode target
    print("\n[4/6] Encoding target variable...")
    df = encode_target(df)

    # Step 5: Train/test split BEFORE preprocessing
    # Important: split on raw data, then preprocess each split
    print(f"\n[5/6] Splitting data ({int((1-test_size)*100)}/{int(test_size*100)})...")
    ref_raw, test_raw = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state
    )
    print(f"  Reference split: {len(ref_raw):,} rows")
    print(f"  Test split:      {len(test_raw):,} rows")

    # Step 6: Preprocess
    print("\n[6/6] Preprocessing...")
    preprocessor = UCIAdultPreprocessor()
    X_ref, y_ref = preprocessor.fit_transform(ref_raw)
    X_test, y_test = preprocessor.transform(test_raw)

    # Save processed data
    ref_df = X_ref.copy()
    ref_df['target'] = y_ref
    ref_df.to_csv(f"{save_dir}/data/reference.csv", index=False)

    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv(f"{save_dir}/data/test.csv", index=False)

    # Save preprocessor
    preprocessor.save(f"{save_dir}/models/preprocessor.pkl")

    print(f"\n✓ Reference data saved: {save_dir}/data/reference.csv")
    print(f"✓ Test data saved:      {save_dir}/data/test.csv")
    print(f"✓ Preprocessor saved:   {save_dir}/models/preprocessor.pkl")
    print("\nPreprocessing complete.")

    return X_ref, y_ref, X_test, y_test, preprocessor


if __name__ == "__main__":
    # Example usage
    X_ref, y_ref, X_test, y_test, preprocessor = run_preprocessing_pipeline(
        raw_data_path="adult.csv",
        save_dir="shared/uci_adult"
    )
