from abc import ABC, abstractmethod
import pandas as pd


class BaseGenerator(ABC):
    """Common interface for synthetic data generators."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Train the generator using real tabular data."""
        pass

    @abstractmethod
    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate the requested number of synthetic rows."""
        pass
