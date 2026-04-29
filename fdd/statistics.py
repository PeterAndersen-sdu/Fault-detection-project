# fdd/statistics.py

# This file contains function for computing Q-statics and T2-stats for FD methods.

from typing import Hashable

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