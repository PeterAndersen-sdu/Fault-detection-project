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


# Computes the PCA projection Z = X * P using the top principal components.
def pca_transform(X: pd.DataFrame, n_components: int) -> pd.DataFrame:
    # Step 1: Compute covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)

    # Step 2: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 3: Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4: Select top k eigenvectors
    projection_matrix = sorted_eigenvectors[:, :n_components]

    # Step 5: Project standardized data X into PCA space to get Z
    Z = X.to_numpy() @ projection_matrix

    return pd.DataFrame(
        Z,
        index=X.index,
        columns=[f"Z{i+1}" for i in range(n_components)],
    )