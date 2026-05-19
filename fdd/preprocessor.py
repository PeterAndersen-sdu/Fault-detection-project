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
    for future DPCA-style models. It can also remove outlier rows using
    an interquartile range rule before fitting or transforming.
    """
    with_standardization: bool = True
    n_lags: int = 0
    drop_na: bool = True
    remove_outliers: bool = False

    # Internal attributes set during fitting.
    def __post_init__(self) -> None:
        self._fitted = False
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None
        self.feature_names_out_: Optional[list[str]] = None
        self.lower_bound_: Optional[pd.Series] = None
        self.upper_bound_: Optional[pd.Series] = None

    # Fits the preprocessor to the dataset, calculating means and stds if needed.
    def fit(self, dataset: TimeSeriesDataset) -> "StandardPreprocessor":
        X = dataset.sensors.copy()

        if self.remove_outliers:
            self.lower_bound_, self.upper_bound_ = self._compute_outlier_bounds(X)
            X = self._apply_outlier_bounds(X, self.lower_bound_, self.upper_bound_)
        else:
            self.lower_bound_ = None
            self.upper_bound_ = None

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

        if self.remove_outliers:
            if self.lower_bound_ is None or self.upper_bound_ is None:
                raise RuntimeError("Outlier bounds are not available. Call fit() before transform().")
            X = self._apply_outlier_bounds(X, self.lower_bound_, self.upper_bound_)

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
                    "remove_outliers": self.remove_outliers,
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

    # Computes IQR-based bounds from the data used during fit.
    @staticmethod
    def _compute_outlier_bounds(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return lower_bound, upper_bound

    # Applies the fitted IQR bounds to remove outlier rows.
    @staticmethod
    def _apply_outlier_bounds(
        df: pd.DataFrame,
        lower_bound: pd.Series,
        upper_bound: pd.Series,
    ) -> pd.DataFrame:
        mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
        return df.loc[mask].reset_index(drop=True)