# fdd/pca.py

# The preprocessor has centered and scaled the data. Now we can apply PCA.

# Step 1: Build covariance matrix of the standardized data.

# Step 2: Compute eigenvalues and eigenvectors of the covariance matrix.

# Step 3: Sort eigenvalues and eigenvectors in descending order of eigenvalues.

# Step 4: Select the top k eigenvectors to form the projection matrix.

# Step 5: Project the standardized data onto the selected eigenvectors to get the PCA-transformed data.

# Example code for PCA transformation (not yet integrated into a class):

import numpy as np
import pandas as pd


def pca_fit(X: pd.DataFrame, n_components: int):
    """
    Fit PCA model on training/healthy data.
    """
    covariance_matrix = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    projection_matrix = sorted_eigenvectors[:, :n_components]

    evr = pd.Series(
        sorted_eigenvalues[:n_components] / sorted_eigenvalues.sum(),
        index=[f"Z{i+1}" for i in range(n_components)]
    )

    model = {
        "projection_matrix": projection_matrix,
        "eigenvalues": sorted_eigenvalues,
        "explained_variance_ratio": evr,
        "n_components": n_components,
    }

    return model


def pca_transform(X: pd.DataFrame, model: dict) -> pd.DataFrame:
    """
    Transform data using an already-fitted PCA model.
    """
    P = model["projection_matrix"]
    n_components = model["n_components"]

    Z = X.to_numpy() @ P

    Z_df = pd.DataFrame(
        Z,
        index=X.index,
        columns=[f"Z{i+1}" for i in range(n_components)],
    )

    return Z_df