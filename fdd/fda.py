# fdd/fda.py

import numpy as np
import pandas as pd

from fdd.dataset import TimeSeriesDataset
from fdd.preprocessor import StandardPreprocessor


class FDAModel:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.preprocessor = StandardPreprocessor(with_standardization=True)
        self.fitted = False

    def fit(self, dataset: TimeSeriesDataset, y: np.ndarray):
        X_scaled = self.preprocessor.fit_transform(dataset).to_numpy()
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n, m = X_scaled.shape
        p = len(self.classes_)

        xbar = X_scaled.mean(axis=0).reshape(-1, 1)

        Sw = np.zeros((m, m))
        Sb = np.zeros((m, m))

        for c in self.classes_:
            Xc = X_scaled[y == c]
            mu_c = Xc.mean(axis=0).reshape(-1, 1)

            Xc_centered = Xc - mu_c.ravel()
            Sw += Xc_centered.T @ Xc_centered

            d = mu_c - xbar
            Sb += Xc.shape[0] * (d @ d.T)

        Sw_reg = Sw + self.eps * np.eye(m)

        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw_reg) @ Sb)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        idx = np.argsort(eigvals)[::-1]
        self.eigenvalues_ = eigvals[idx]
        self.Wp_ = eigvecs[:, idx[:p - 1]]

        Z_train = X_scaled @ self.Wp_

        self.muZ_ = {}
        self.SigmaZ_ = {}

        for c in self.classes_:
            Zc = Z_train[y == c]
            self.muZ_[c] = Zc.mean(axis=0).reshape(-1, 1)

            cov = np.cov(Zc.T)
            cov = np.atleast_2d(cov)
            self.SigmaZ_[c] = cov + self.eps * np.eye(cov.shape[0])

        self.fitted = True
        return self

    def transform(self, dataset: TimeSeriesDataset) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FDAModel must be fitted before transform().")

        X_scaled = self.preprocessor.transform(dataset).to_numpy()
        Z = X_scaled @ self.Wp_

        return pd.DataFrame(
            Z,
            columns=[f"FDA{i+1}" for i in range(Z.shape[1])],
        )

    def predict(self, dataset: TimeSeriesDataset):
        Z = self.transform(dataset).to_numpy()

        classes_sorted = sorted(self.classes_)
        distances = np.zeros((len(Z), len(classes_sorted)))

        for i in range(len(Z)):
            zi = Z[i].reshape(-1, 1)

            for j, c in enumerate(classes_sorted):
                d = zi - self.muZ_[c]
                distances[i, j] = (
                    d.T @ np.linalg.pinv(self.SigmaZ_[c]) @ d
                ).item()

        pred_idx = np.argmin(distances, axis=1)
        pred = np.array([classes_sorted[j] for j in pred_idx])

        distance_to_healthy = distances[:, classes_sorted.index(0)]
        distance_to_fault = distances[:, classes_sorted.index(1)]
        decision_margin = distance_to_healthy - distance_to_fault

        return pred, distances, distance_to_healthy, distance_to_fault, decision_margin