# fdd/fda.py

import numpy as np
import pandas as pd
from scipy.linalg import eig as generalized_eig

from fdd.dataset import TimeSeriesDataset


class FDAModel:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.fitted = False

    def fit(self, dataset: TimeSeriesDataset, y: np.ndarray):
        X = dataset.sensors.copy()
        y = np.asarray(y)

        if X.shape[0] != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("FDA requires at least two classes.")

        self.class_labels_ = list(self.classes_)
        self.reference_class_ = self.class_labels_[0]
        self.alternative_class_ = self.class_labels_[1]

        self.n_classes_ = len(self.classes_)
        self.n_components_ = min(self.n_classes_ - 1, X.shape[1])

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1).replace(0, 1.0)
        X_scaled = ((X - self.mean_) / self.std_).to_numpy()

        n, m = X_scaled.shape

        xbar = X_scaled.mean(axis=0).reshape(-1, 1)

        self.total_scatter_ = np.zeros((m, m))
        Sw = np.zeros((m, m))
        Sb = np.zeros((m, m))

        for c in self.classes_:
            Xc = X_scaled[y == c]
            if Xc.shape[0] == 0:
                raise ValueError(f"Class {c} has no samples.")

            mu_c = Xc.mean(axis=0).reshape(-1, 1)

            Xc_centered = Xc - mu_c.ravel()
            Sw += Xc_centered.T @ Xc_centered

            Xc_total_centered = Xc - xbar.ravel()
            self.total_scatter_ += Xc_total_centered.T @ Xc_total_centered

            d = mu_c - xbar
            Sb += Xc.shape[0] * (d @ d.T)

        Sw_reg = Sw + self.eps * np.eye(m)

        eigvals, eigvecs = generalized_eig(Sb, Sw_reg)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        idx = np.argsort(eigvals)[::-1]
        self.eigenvalues_ = eigvals[idx]
        self.Wp_ = eigvecs[:, idx[:self.n_components_]]
        self.projection_matrix_ = self.Wp_

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

        X_scaled = ((dataset.sensors.copy() - self.mean_) / self.std_).to_numpy()
        Z = X_scaled @ self.Wp_

        return pd.DataFrame(
            Z,
            index=dataset.sensors.index,
            columns=[f"FDA{i+1}" for i in range(Z.shape[1])],
        )

    def predict(self, dataset: TimeSeriesDataset):
        Z = self.transform(dataset).to_numpy()

        classes_sorted = list(self.class_labels_)
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

        distance_to_healthy = distances[:, classes_sorted.index(self.reference_class_)]
        distance_to_fault = distances[:, classes_sorted.index(self.alternative_class_)]
        decision_margin = distance_to_healthy - distance_to_fault

        return pred, distances, distance_to_healthy, distance_to_fault, decision_margin