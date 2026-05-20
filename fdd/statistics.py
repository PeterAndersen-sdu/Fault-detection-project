# fdd/statistics.py

# This file contains function for computing Q-statics and T2-stats for FD methods.

# from typing import Hashable

import numpy as np
import pandas as pd

# Computes the Q-statistic for each sample in X based on its reconstruction from Z.
def q_statistic(X, Z):
    x_reconstructed = Z @ np.linalg.pinv(Z.values) @ X.values
    e = X.values - x_reconstructed
    return np.sum(e ** 2, axis=1)

# Computes Hotelling's T2-statistic for each sample in Z based on the covariance of Z.
def t2_statistic(Z):
    covariance_matrix = np.cov(Z, rowvar=False)
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    mean_vector = np.mean(Z, axis=0)
    t2_values = []
    for i in range(Z.shape[0]):
        diff = Z.iloc[i].values - mean_vector
        t2 = diff.T @ inv_covariance_matrix @ diff
        t2_values.append(t2)
    return np.array(t2_values)

#     fdd/statistics.py

# """
# Monitoring statistics for PCA- and DPCA-based fault detection.

# This implementation follows the classical PCA formulation used in the MATLAB
# teaching scripts:

#     X_hat = T P^T
#     E = X - X_hat
#     Q = sum(E^2)

# and:

#     T^2 = sum(t_i^2 / lambda_i)

# where T is the PCA/DPCA score matrix, P is the loading matrix, and lambda_i
# are the retained eigenvalues.
# """

# import numpy as np
# import pandas as pd


# def q_statistic(X: pd.DataFrame, scores: pd.DataFrame, model: dict) -> np.ndarray:
#     """
#     Compute the classical Q statistic / Squared Prediction Error (SPE).

#     Classical PCA reconstruction:
#         X_hat = T P^T

#     Residual:
#         E = X - X_hat

#     Q statistic:
#         Q = sum(E^2, axis=1)

#     Parameters
#     ----------
#     X:
#         Preprocessed input data.
#         For PCA: standardized sensor data.
#         For DPCA: standardized lagged matrix.

#     scores:
#         PCA or DPCA score matrix T.

#     model:
#         PCA model dictionary containing:
#             - "projection_matrix"

#     Returns
#     -------
#     np.ndarray
#         Q statistic for each observation.
#     """
#     P = model["projection_matrix"]

#     X_values = X.to_numpy()
#     T_values = scores.to_numpy()

#     X_hat = T_values @ P.T
#     E = X_values - X_hat

#     Q = np.sum(E ** 2, axis=1)

#     return Q


# def t2_statistic(scores: pd.DataFrame, model: dict) -> np.ndarray:
#     """
#     Compute classical Hotelling's T² statistic in PCA/DPCA score space.

#     Classical formulation:
#         T² = sum(t_i² / lambda_i)

#     Parameters
#     ----------
#     scores:
#         PCA or DPCA score matrix T.

#     model:
#         PCA model dictionary containing:
#             - "eigenvalues"
#             - "n_components"

#     Returns
#     -------
#     np.ndarray
#         Hotelling's T² statistic for each observation.
#     """
#     n_components = model["n_components"]
#     eigenvalues = model["eigenvalues"][:n_components]

#     T_values = scores.to_numpy()

#     T2 = np.sum((T_values ** 2) / eigenvalues, axis=1)

#     return T2


# def reconstruction(X: pd.DataFrame, scores: pd.DataFrame, model: dict) -> pd.DataFrame:
#     """
#     Reconstruct X from retained PCA/DPCA components.

#     Reconstruction:
#         X_hat = T P^T

#     Parameters
#     ----------
#     X:
#         Original preprocessed input data.

#     scores:
#         PCA/DPCA score matrix.

#     model:
#         PCA model dictionary containing:
#             - "projection_matrix"

#     Returns
#     -------
#     pd.DataFrame
#         Reconstructed data matrix with same index and columns as X.
#     """
#     P = model["projection_matrix"]

#     X_hat = scores.to_numpy() @ P.T

#     return pd.DataFrame(
#         X_hat,
#         index=X.index,
#         columns=X.columns,
#     )


# def residual_matrix(X: pd.DataFrame, scores: pd.DataFrame, model: dict) -> pd.DataFrame:
#     """
#     Compute residual matrix.

#     Residual:
#         E = X - X_hat

#     Returns
#     -------
#     pd.DataFrame
#         Residual matrix.
#     """
#     X_hat = reconstruction(X, scores, model)

#     return X - X_hat