"""
IEEE-CIS Fraud Detection - Preprocessing Pipeline
J26-DS-361 | Shared Artifact | Maintained by Sajivan K (IT23172296)

Dataset: Kaggle IEEE-CIS Fraud Detection
Download: https://www.kaggle.com/c/ieee-fraud-detection/data
Files needed: train_transaction.csv (train_identity.csv optional)

IMPORTANT:
- All transforms fitted on REFERENCE split only
- Never refit on production/drifted data
- Uses random split (not temporal) for injection-based drift experiments
- ELEC2 reserved for PP2 temporal drift validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURATION ────────────────────────────────────────────────────────────

# Maximum rows to use (full dataset is 590K - use subset for efficiency)
# 200K gives enough data while keeping benchmark runs manageable
MAX_ROWS = 200_000

# Drop columns with more than this fraction of missing values
MISSING_THRESHOLD = 0.5

# Target column
TARGET_COLUMN = 'isFraud'

# Known categorical columns in IEEE-CIS
KNOWN_CATEGORICAL = [
    'ProductCD', 'card4', 'card6',
    'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]


def load_raw_data(transaction_path, identity_path=None, max_rows=MAX_ROWS):
    """
    Load IEEE-CIS transaction data.
    Optionally merge with identity data.

    Args:
        transaction_path: path to train_transaction.csv
        identity_path:    path to train_identity.csv (optional)
        max_rows:         maximum rows to load (None = all)
    """
    print(f"Loading transaction data from: {transaction_path}")

    if max_rows:
        df = pd.read_csv(transaction_path, nrows=max_rows)
        print(f"Loaded {len(df):,} rows (limited to {max_rows:,})")
    else:
        df = pd.read_csv(transaction_path)
        print(f"Loaded {len(df):,} rows (full dataset)")

    if identity_path and os.path.exists(identity_path):
        print(f"Loading identity data from: {identity_path}")
        identity_df = pd.read_csv(identity_path)
        df = df.merge(identity_df, on='TransactionID', how='left')
        print(f"After merge: {df.shape[0]:,} rows, {df.shape[1]} columns")

    print(f"Shape: {df.shape}")
    print(f"Class balance:")
    print(df[TARGET_COLUMN].value_counts(normalize=True).round(4))

    return df


def explore_data(df):
    """Quick exploration before preprocessing."""
    print("\n=== DATA EXPLORATION ===")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    fraud_rate = df[TARGET_COLUMN].mean()
    print(f"  Legitimate: {(1-fraud_rate)*100:.1f}%")
    print(f"  Fraud:      {fraud_rate*100:.1f}%")

    # Missing values summary
    missing = df.isnull().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0.3]
    print(f"\nColumns with >30% missing ({len(high_missing)} columns):")
    if len(high_missing) > 0:
        print(high_missing.head(10).round(3))

    print(f"\nTotal columns: {df.shape[1]}")
    print(f"Columns to drop (>{MISSING_THRESHOLD*100:.0f}% missing): "
          f"{(missing > MISSING_THRESHOLD).sum()}")


def drop_high_missing_columns(df, threshold=MISSING_THRESHOLD):
    """Drop columns with too many missing values."""
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()

    # Always keep target
    cols_to_drop = [c for c in cols_to_drop if c != TARGET_COLUMN]

    print(f"Dropping {len(cols_to_drop)} columns (>{threshold*100:.0f}% missing)")
    df = df.drop(columns=cols_to_drop)
    print(f"Remaining columns: {df.shape[1]}")

    return df, cols_to_drop


def identify_column_types(df):
    """
    Automatically identify continuous and categorical columns.
    Excludes TransactionID, TransactionDT, and target.
    """
    exclude = ['TransactionID', 'TransactionDT', TARGET_COLUMN]
    feature_cols = [c for c in df.columns if c not in exclude]

    categorical_cols = []
    continuous_cols = []

    for col in feature_cols:
        if col in KNOWN_CATEGORICAL:
            categorical_cols.append(col)
        elif df[col].dtype == 'object':
            categorical_cols.append(col)
        elif df[col].nunique() <= 10:
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)

    print(f"\nColumn types identified:")
    print(f"  Continuous:   {len(continuous_cols)} columns")
    print(f"  Categorical:  {len(categorical_cols)} columns")

    return continuous_cols, categorical_cols


class IEEECISPreprocessor:
    """
    Stateful preprocessor for IEEE-CIS dataset.
    Fit on reference data only, transform any data.

    Usage:
        preprocessor = IEEECISPreprocessor()
        X_ref, y_ref = preprocessor.fit_transform(reference_df)
        X_prod, y_prod = preprocessor.transform(production_df)
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.continuous_cols = None
        self.categorical_cols = None
        self.feature_columns = None
        self.dropped_cols = []
        self.is_fitted = False

    def fit_transform(self, df):
        """Fit on reference data and transform."""
        df = df.copy()

        # Drop ID and time columns (not features)
        cols_to_drop = [c for c in ['TransactionID', 'TransactionDT']
                        if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Separate target
        y = df[TARGET_COLUMN].values
        df = df.drop(columns=[TARGET_COLUMN])

        # Identify column types
        self.continuous_cols, self.categorical_cols = identify_column_types(
            df.assign(**{TARGET_COLUMN: 0})  # temp add target for identification
        )

        # Fill missing values
        # Continuous: median imputation
        for col in self.continuous_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Categorical: mode imputation + encode
        for col in self.categorical_cols:
            if col in df.columns:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_val).astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        # Scale continuous
        cont_present = [c for c in self.continuous_cols if c in df.columns]
        df[cont_present] = self.scaler.fit_transform(df[cont_present])

        self.feature_columns = list(df.columns)
        self.is_fitted = True

        print(f"\nPreprocessor fitted on {len(df):,} samples")
        print(f"Total features: {len(self.feature_columns)}")

        return df, y

    def transform(self, df):
        """Transform new data using fitted encoders and scaler."""
        assert self.is_fitted, "Preprocessor must be fitted before transform"

        df = df.copy()

        # Drop ID and time columns
        cols_to_drop = [c for c in ['TransactionID', 'TransactionDT']
                        if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Separate target if present
        y = None
        if TARGET_COLUMN in df.columns:
            y = df[TARGET_COLUMN].values
            df = df.drop(columns=[TARGET_COLUMN])

        # Fill and encode categoricals
        for col, le in self.label_encoders.items():
            if col in df.columns:
                mode_val = str(le.classes_[0])
                df[col] = df[col].fillna(mode_val).astype(str)
                known = set(le.classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        # Fill and scale continuous
        cont_present = [c for c in self.continuous_cols if c in df.columns]
        for col in cont_present:
            df[col] = df[col].fillna(df[col].median())
        df[cont_present] = self.scaler.transform(df[cont_present])

        # Ensure only feature columns present and in correct order
        missing_cols = set(self.feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0

        df = df[self.feature_columns]

        return df, y

    def save(self, path):
        joblib.dump(self, path)
        print(f"Preprocessor saved: {path}")

    @staticmethod
    def load(path):
        preprocessor = joblib.load(path)
        print(f"Preprocessor loaded: {path}")
        return preprocessor


def run_preprocessing_pipeline(transaction_path, save_dir,
                                identity_path=None,
                                max_rows=MAX_ROWS,
                                test_size=0.2,
                                random_state=42):
    """
    Full preprocessing pipeline for IEEE-CIS.

    Args:
        transaction_path: path to train_transaction.csv
        save_dir:         directory to save outputs
        identity_path:    path to train_identity.csv (optional)
        max_rows:         max rows to load
        test_size:        test split fraction
        random_state:     random seed

    Returns:
        X_ref, y_ref, X_test, y_test, preprocessor
    """
    os.makedirs(f"{save_dir}/data", exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)

    print("=" * 60)
    print("IEEE-CIS FRAUD DETECTION PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/6] Loading raw data...")
    df = load_raw_data(transaction_path, identity_path, max_rows)

    # Step 2: Explore
    print("\n[2/6] Exploring data...")
    explore_data(df)

    # Step 3: Drop high missing columns
    print("\n[3/6] Dropping high-missing columns...")
    df, dropped_cols = drop_high_missing_columns(df, MISSING_THRESHOLD)

    # Save list of dropped columns
    with open(f"{save_dir}/dropped_columns.txt", 'w') as f:
        f.write('\n'.join(dropped_cols))

    # Step 4: Train/test split BEFORE preprocessing
    print(f"\n[4/6] Splitting data...")
    ref_raw, test_raw = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state
    )
    print(f"  Reference: {len(ref_raw):,} rows")
    print(f"  Test:      {len(test_raw):,} rows")

    # Step 5: Preprocess
    print("\n[5/6] Preprocessing...")
    preprocessor = IEEECISPreprocessor()
    X_ref, y_ref = preprocessor.fit_transform(ref_raw)
    X_test, y_test = preprocessor.transform(test_raw)

    # Step 6: Save
    print("\n[6/6] Saving...")
    ref_df = X_ref.copy()
    ref_df['target'] = y_ref
    ref_df.to_csv(f"{save_dir}/data/reference.csv", index=False)

    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv(f"{save_dir}/data/test.csv", index=False)

    preprocessor.save(f"{save_dir}/models/preprocessor.pkl")

    print(f"\n✓ Reference saved: {save_dir}/data/reference.csv")
    print(f"✓ Test saved:      {save_dir}/data/test.csv")
    print(f"✓ Preprocessor:    {save_dir}/models/preprocessor.pkl")
    print("\nPreprocessing complete.")

    return X_ref, y_ref, X_test, y_test, preprocessor


if __name__ == "__main__":
    X_ref, y_ref, X_test, y_test, preprocessor = run_preprocessing_pipeline(
        transaction_path="train_transaction.csv",
        save_dir="shared/ieee_cis",
        max_rows=200_000
    )
