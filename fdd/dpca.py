# fdd/dpca.py

# This file contains the functions for Dynamic PCA (DPCA).

import numpy as np
import pandas as pd

from fdd.pca import pca_fit
from fdd.pca import pca_transform


import numpy as np
import pandas as pd


def build_lagged_matrix(X: pd.DataFrame, time_lags: int) -> pd.DataFrame:
    """
    Build lagged DPCA matrix:
        z(t) = [x(t), x(t-1), ..., x(t-L)]
    """
    m, n = X.shape

    if time_lags >= m:
        raise ValueError("time_lags must be smaller than the number of rows in X.")

    n_rows = m - time_lags
    n_cols = n * (time_lags + 1)

    Z = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(time_lags + 1):
            Z[i, j*n:(j+1)*n] = X.iloc[i + time_lags - j].to_numpy()

    columns = []
    for lag in range(time_lags + 1):
        for col in X.columns:
            if lag == 0:
                columns.append(f"{col}_t")
            else:
                columns.append(f"{col}_t-{lag}")

    return pd.DataFrame(Z, index=X.index[time_lags:], columns=columns)


def dpca_fit(X_train: pd.DataFrame, n_components: int, time_lags: int) -> dict:
    """
    Fit DPCA model on healthy/training data.
    """
    # Step 1: Build lagged matrix
    Z_train = build_lagged_matrix(X_train, time_lags)

    # Step 2: Fit mean/std on training lagged data
    mean = Z_train.mean()
    std = Z_train.std()
    std = std.replace(0, 1e-12)

    Z_train_standardized = (Z_train - mean) / std

    # Step 3: Fit PCA on standardized lagged training data
    pca_model = pca_fit(Z_train_standardized, n_components)

    model = {
        "time_lags": time_lags,
        "mean": mean,
        "std": std,
        "pca_model": pca_model,
        "explained_variance_ratio": pca_model["explained_variance_ratio"],
    }

    return model


def dpca_transform(X: pd.DataFrame, model: dict) -> pd.DataFrame:
    """
    Transform data using fitted DPCA model.
    """
    time_lags = model["time_lags"]

    # Step 1: Build lagged matrix
    Z = build_lagged_matrix(X, time_lags)

    # Step 2: Use training mean/std
    Z_standardized = (Z - model["mean"]) / model["std"]

    # Step 3: Transform using training PCA model
    Z_dpca = pca_transform(Z_standardized, model["pca_model"])

    return Z_dpca


def dpca_prepare_matrix(X: pd.DataFrame, model: dict) -> pd.DataFrame:
    """
    Build and standardize lagged matrix using fitted DPCA model.
    Useful for Q/SPE and T² calculations.
    """
    Z = build_lagged_matrix(X, model["time_lags"])
    Z_standardized = (Z - model["mean"]) / model["std"]
    return Z_standardized

