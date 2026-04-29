# fdd/dpca.py

# This file contains the functions for Dynamic PCA (DPCA).

import numpy as np
import pandas as pd

from fdd.pca import pca_transform


def dpca_transform(X: pd.DataFrame, n_components: int, time_lags: int) -> pd.DataFrame:
    # Step 1: Build lagged matrix
        # Builds:
        # z(t) = [x(t), x(t-1), ..., x(t-L)]
    
    [m, n] = X.shape

    nRows = m - time_lags
    nCols = n * (time_lags + 1)

    Z = np.zeros((nRows, nCols))

    for i in range(nRows):
        for j in range(time_lags + 1):
            Z[i, j*n:(j+1)*n] = X.iloc[i+time_lags-j].to_numpy()
    Z_df = pd.DataFrame(Z, index=X.index[time_lags:])
    
    # Step 2: Center and scale the lagged data
    Z_standardized = (Z_df - Z_df.mean()) / Z_df.std()

    # Step 3: Apply PCA to the standardized lagged data
    Z_dpca, evr = pca_transform(Z_standardized, n_components)
    return Z_dpca, evr, Z_df
