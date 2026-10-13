import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

from component3.src.generators.base import BaseGenerator
from component3.vendor.tab_ddpm import GaussianMultinomialDiffusion
from component3.vendor.tab_ddpm.modules import MLPDiffusion


class TabDDPMGenerator(BaseGenerator):
    """TabDDPM-based synthetic tabular data generator."""

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 256,
        num_timesteps: int = 100,
        learning_rate: float = 0.001,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.learning_rate = learning_rate
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.diffusion = None
        self.is_fitted = False
        self.num_scaler = None
        self.cat_encoder = None
        self.num_columns = []
        self.cat_columns = []
        self.original_columns = []

    def fit(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Training data cannot be empty.")
        if data.columns.has_duplicates:
            raise ValueError("Training data contains duplicate column names.")
        if data.isna().any().any():
            raise ValueError("Missing values must be handled before training.")

        self.is_fitted = False
        self.model = None
        self.diffusion = None
        self.original_columns = data.columns.tolist()
        self.original_dtypes = data.dtypes.to_dict()

        # Store observed numerical ranges for later validation.
        self.numerical_ranges = {}
        for column in data.select_dtypes(include=["number"]).columns:
            self.numerical_ranges[column] = {
                "min": data[column].min(),
                "max": data[column].max()
            }

        self.cat_columns = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        self.num_columns = [
            col for col in self.original_columns
            if col not in self.cat_columns
        ]

        if not self.num_columns and not self.cat_columns:
            raise ValueError("No supported columns found.")

        if self.num_columns:
            n_rows = len(data)
            self.num_scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(min(n_rows // 30, 1000), 10),
                subsample=1_000_000_000,
                random_state=42
            )
            x_num = self.num_scaler.fit_transform(
                data[self.num_columns]
            )
        else:
            self.num_scaler = None
            x_num = np.empty((len(data), 0))

        if self.cat_columns:
            self.cat_encoder = OrdinalEncoder(
                dtype=np.int64
            )
            x_cat = self.cat_encoder.fit_transform(
                data[self.cat_columns]
            )
            self.num_classes = np.array(
                [len(categories) for categories in self.cat_encoder.categories_],
                dtype=np.int64
            )
        else:
            self.cat_encoder = None
            x_cat = np.empty((len(data), 0), dtype=np.int64)
            self.num_classes = np.array([0])

        self.training_array = np.concatenate(
            [x_num, x_cat], axis=1
        ).astype(np.float32)

        print("Numerical columns:", self.num_columns)
        print("Categorical columns:", self.cat_columns)
        print("Training data shape:", self.training_array.shape)

        # Validate training configuration
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("Epochs and batch size must be positive.")
        if self.num_timesteps <= 0 or self.learning_rate <= 0:
            raise ValueError("Timesteps and learning rate must be positive.")

        # Calculate TabDDPM neural network input size
        num_numerical = len(self.num_columns)
        num_categorical = (
            int(self.num_classes.sum())
            if self.cat_columns else 0
        )
        input_dim = num_numerical + num_categorical

        # Create the genuine TabDDPM neural network
        self.model = MLPDiffusion(
            d_in=input_dim,
            num_classes=0,
            is_y_cond=False,
            rtdl_params={
                "d_layers": [64, 64],
                "dropout": 0.0
            }
        )

        # Create the genuine diffusion model
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.num_classes,
            num_numerical_features=num_numerical,
            denoise_fn=self.model,
            num_timesteps=self.num_timesteps,
            device=self.device
        ).to(self.device)

        # Convert training data to PyTorch tensors
        training_tensor = torch.tensor(
            self.training_array,
            dtype=torch.float32,
            device=self.device
        )

        optimizer = torch.optim.Adam(
            self.diffusion.parameters(),
            lr=self.learning_rate
        )

        print("Starting TabDDPM training...")
        self.diffusion.train()

        for epoch in range(self.epochs):
            indices = torch.randperm(
                len(training_tensor),
                device=self.device
            )
            epoch_losses = []
            for start in range(
                0, len(training_tensor), self.batch_size
            ):
                batch_indices = indices[
                    start:start + self.batch_size
                ]
                batch = training_tensor[batch_indices]

                optimizer.zero_grad()
                loss_multi, loss_gauss = (
                    self.diffusion.mixed_loss(batch, {})
                )
                loss = loss_multi + loss_gauss

                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        "TabDDPM training loss is not finite."
                    )

                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            average_loss = float(np.mean(epoch_losses))
            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"- Loss: {average_loss:.4f}"
            )

        self.diffusion.eval()
        self.is_fitted = True
        print("TabDDPM training completed!")

    def sample(self, n_rows: int) -> pd.DataFrame:
        if not self.is_fitted or self.diffusion is None:
            raise RuntimeError("Train TabDDPM before sampling.")
        if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows <= 0:
            raise ValueError("n_rows must be a positive integer.")

        self.diffusion.eval()

        # Generate data using the genuine TabDDPM reverse diffusion process.
        with torch.no_grad():
            samples, _ = self.diffusion.sample(
                num_samples=n_rows,
                y_dist=torch.tensor([1.0], device=self.device)
            )

        generated = samples.cpu().numpy()

        if not np.isfinite(generated).all():
            raise RuntimeError("TabDDPM generated non-finite values.")

        result = pd.DataFrame(index=range(n_rows))

        # Restore numerical columns to their original scale.
        num_count = len(self.num_columns)
        if self.num_columns:
            numerical_data = self.num_scaler.inverse_transform(
                generated[:, :num_count]
            )
            for index, column in enumerate(self.num_columns):
                result[column] = numerical_data[:, index]

        # Restore categorical columns to their original values.
        if self.cat_columns:
            categorical_data = generated[:, num_count:].copy()
            for index, categories in enumerate(self.cat_encoder.categories_):
                categorical_data[:, index] = np.clip(
                    np.rint(categorical_data[:, index]),
                    0,
                    len(categories) - 1
                )
            restored_categories = self.cat_encoder.inverse_transform(
                categorical_data
            )
            for index, column in enumerate(self.cat_columns):
                result[column] = restored_categories[:, index]

        # Preserve original numerical data types.
        for column in self.num_columns:
            original_dtype = self.original_dtypes[column]
            if pd.api.types.is_integer_dtype(original_dtype):
                result[column] = np.rint(result[column])
                if not np.isfinite(result[column].to_numpy()).all():
                    raise RuntimeError(
                        f"Non-finite generated values in {column}."
                    )
                result[column] = result[column].astype(original_dtype)
            elif pd.api.types.is_float_dtype(original_dtype):
                result[column] = result[column].astype(original_dtype)

        # Preserve the original column order.
        return result[self.original_columns]

    def validate_numerical_ranges(
        self, synthetic_data: pd.DataFrame
    ) -> dict:
        """Check generated numerical values against observed training ranges."""
        if not self.is_fitted:
            raise RuntimeError("Train TabDDPM before validation.")
        if not isinstance(synthetic_data, pd.DataFrame):
            raise TypeError("synthetic_data must be a pandas DataFrame.")
        if synthetic_data.empty:
            raise ValueError("synthetic_data cannot be empty.")

        missing_columns = [
            column for column in self.numerical_ranges
            if column not in synthetic_data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing numerical columns: {missing_columns}"
            )

        report = {
            "total_rows": len(synthetic_data),
            "columns": {},
            "total_invalid_values": 0,
            "passed": True,
        }

        for column, limits in self.numerical_ranges.items():
            values = pd.to_numeric(
                synthetic_data[column], errors="coerce"
            )
            invalid_mask = (
                values.isna()
                | ~np.isfinite(values)
                | (values < limits["min"])
                | (values > limits["max"])
            )
            invalid_count = int(invalid_mask.sum())
            report["columns"][column] = {
                "min": limits["min"],
                "max": limits["max"],
                "invalid_count": invalid_count,
                "invalid_percentage": round(
                    invalid_count / len(synthetic_data) * 100, 2
                ),
            }
            report["total_invalid_values"] += invalid_count

        report["passed"] = (
            report["total_invalid_values"] == 0
        )
        return report

