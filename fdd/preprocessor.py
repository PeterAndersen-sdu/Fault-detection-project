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
        # Outlier removal is now a separate step; fit does not remove rows.
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
        if self.feature_names_out_ is not None:
            X = X.reindex(columns=self.feature_names_out_, fill_value=0.0)

        # NOTE: outlier removal is intentionally not applied here. Use
        # `compute_outlier_bounds` and `remove_outliers_from_dataset` to
        # perform that step explicitly before calling `transform`.
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

    # Public helper: compute IQR-based outlier bounds from a dataset (on raw sensors)
    def compute_outlier_bounds(self, dataset: TimeSeriesDataset) -> tuple[pd.Series, pd.Series]:
        """Compute and return (lower_bound, upper_bound) using IQR rule from dataset.sensors.

        This does not modify the preprocessor state.
        """
        X = dataset.sensors.copy()
        return self._compute_outlier_bounds(X)

    # Public helper: remove outliers from a dataset using provided bounds
    def remove_outliers_from_dataset(
        self, dataset: TimeSeriesDataset, lower_bound: pd.Series, upper_bound: pd.Series
    ) -> TimeSeriesDataset:
        """Apply outlier bounds to `dataset` and return a cleaned TimeSeriesDataset.

        Removal is applied before lagging (consistent with previous behavior).
        """
        X = dataset.sensors.copy()
        X_clean = self._apply_outlier_bounds(X, lower_bound, upper_bound)

        if self.n_lags > 0:
            X_clean = self._build_lagged_dataframe(X_clean, self.n_lags)

        if self.drop_na:
            X_clean = X_clean.dropna().reset_index(drop=True)

        return TimeSeriesDataset(
            name=f"{dataset.name}_outliers_removed",
            sensors=X_clean,
            metadata={**dataset.metadata, "outliers_removed": True},
        )

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