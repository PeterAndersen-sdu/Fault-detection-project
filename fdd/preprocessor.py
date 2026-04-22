# fdd/preprocessor.py

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .dataset import TimeSeriesDataset


@dataclass
class StandardPreprocessor:
    """
    Standardizes sensor data and optionally creates lagged features
    for future DPCA-style models.
    """
    with_standardization: bool = True
    n_lags: int = 0
    drop_na: bool = True

    # Internal attributes set during fitting.
    def __post_init__(self) -> None:
        self._fitted = False
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None
        self.feature_names_out_: Optional[list[str]] = None

    # Fits the preprocessor to the dataset, calculating means and stds if needed.
    def fit(self, dataset: TimeSeriesDataset) -> "StandardPreprocessor":
        X = dataset.sensors.copy()

        if self.n_lags > 0:
            X = self._build_lagged_dataframe(X, self.n_lags)

        if self.drop_na:
            X = X.dropna().reset_index(drop=True)

        if self.with_standardization:
            self.mean_ = X.mean()
            self.std_ = X.std(ddof=0).replace(0, 1.0)
        else:
            self.mean_ = pd.Series(0.0, index=X.columns)
            self.std_ = pd.Series(1.0, index=X.columns)

        self.feature_names_out_ = list(X.columns)
        self._fitted = True
        return self

    # Transforms the dataset using the fitted parameters, applying standardization and lagging.
    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        X = dataset.sensors.copy()

        if self.n_lags > 0:
            X = self._build_lagged_dataframe(X, self.n_lags)

        if self.drop_na:
            valid_index = X.dropna().index
            X = X.loc[valid_index].reset_index(drop=True)

        X_scaled = (X - self.mean_) / self.std_     # Standardization

        return TimeSeriesDataset(
            name=f"{dataset.name}_preprocessed",
            sensors=X_scaled,
            metadata={
                **dataset.metadata,
                "preprocessing": {
                    "with_standardization": self.with_standardization,
                    "n_lags": self.n_lags,
                    "drop_na": self.drop_na,
                },
            },
        )

    # Convenience method to fit and transform in one step.
    def fit_transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        return self.fit(dataset).transform(dataset)

    # Builds a lagged version of the input dataframe, creating new columns for each lag.
    @staticmethod
    def _build_lagged_dataframe(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
        lagged_parts = [df.copy()]
        for lag in range(1, n_lags + 1):
            lagged = df.shift(lag).copy()
            lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
            lagged_parts.append(lagged)

        return pd.concat(lagged_parts, axis=1)