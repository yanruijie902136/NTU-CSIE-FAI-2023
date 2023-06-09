import matplotlib.pyplot as plt
import numpy as np
import torch


# def plot_vector(vector, filepath):
    # plt.imshow(vector.reshape(61, 80), cmap='gray')
    # plt.savefig(filepath)
    # plt.clf()

"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        # Hint: Use existing method to calculate covariance matrix and its eigenvalues and eigenvectors

        # Covariance matrix.
        self.mean = np.mean(X, axis=0)
        # plot_vector(self.mean, "mean_vector")
        covariance_matrix = np.cov(X - self.mean, rowvar=False)

        # Top eigenvectors.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices][:, 0:self.n_components]

        # for i in range(min(self.n_components, 4)):
            # plot_vector(self.components[:, i], "eigenvector_#{}".format(i + 1))
        # raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        # Hint: Use the calculated principal components to project the data onto a lower dimensional space

        return (X - self.mean) @ self.components

        # raise NotImplementedError

    def reconstruct(self, X):
        #TODO: 2%
        # Hint: Use the calculated principal components to reconstruct the data back to its original space

        return self.transform(X) @ self.components.T + self.mean

        # raise NotImplementedError
