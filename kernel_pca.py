import numpy as np
from typing import Callable
from preprocessing.kernel_centerer import KernelCenterer


class KernelPCA:
    def __init__(self, kernel: Callable, n_components: int = 2, eigtol: float = 1e-4):
        """
        Initialize Kernel PCA.

        :param kernel: a positive definite kernel function taking two numpy array parameters as input
        :param n_components: the # of principal components to represent data with (dimension of embedding)
        :param eigtol: a tolerance threshold on negative eigenvalues, should be "close" to machine precision
        """
        self.kernel = kernel
        self.n_components = n_components
        self.eigtol = eigtol
        self._centerer = KernelCenterer()
        self._components = None
        self._all_components = None
        self._X_fit = None

    def update_dimension(self, new_n_components: int) -> None:
        """
        Change the dimension of the KPCA embedding. No re-fitting is required.

        :param new_n_components: int, the new # of principal components to represent data with
        """
        if self._components is None:
            raise ValueError('Cannot update KPCA dimension before fitting. Please fit first')
        if self.n_components == new_n_components:
            print('[KernelPCA.update_dimension] new_n_components is the same as current n_components : no-op')
            return

        self.n_components = new_n_components
        self._components = self._all_components[:self.n_components]

    def _get_eigs(self, K):
        """
        Compute eigenvalues/vectors of K and return those with positive eigenvalue
        """
        eigvals, eigvecs = np.linalg.eigh(K)  # numpy returns this sorted in eigvals ascending order
        # convert eigvecs to array-of-vectors (eigvecs[0] is now the 0-th eigenvector, etc) + descending order
        eigvecs = eigvecs.T[::-1]
        eigvals = eigvals[::-1]  # descending order
        if eigvals[-1] < -self.eigtol:
            raise np.linalg.LinAlgError(
                f'Centered kernel is not PSD, found smallest eigenvalue {eigvals[-1]}'
            )
        pos_eig_mask = eigvals > 0
        return eigvals[pos_eig_mask], eigvecs[pos_eig_mask]

    def fit(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit the Kernel PCA and return the transformed data.

        :param X: the data matrix to fit, of shape (n_samples, n_features)
        :param y: ignored parameter, to comply with sklearn API
        :return: the transformed training data, of shape (n_samples, n_components)
        """
        self._X_fit = X
        K = self._centerer.fit_transform(self.kernel(X, X))
        eigvals, eigvecs = self._get_eigs(K)
        sqrt_eigvals = np.sqrt(eigvals)
        self._all_components = eigvecs / sqrt_eigvals[:, None]
        self._components = self._all_components[:self.n_components]
        return (sqrt_eigvals[:self.n_components][:, None] * eigvecs[:self.n_components]).T

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Alias for KernelPCA.fit().

        Since the fitting process involves transforming the training data, it is always returned in fit().
        """
        return self.fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Tranform the data to embed it using principal components projections.

        :param X: the data matrix to transform, of shape (n1, n_features)
        :return: the projection of the data onto the principal components, of shape (n1, n_components)
        """
        K = self._centerer.transform(self.kernel(X, self._X_fit))
        # shapes : K : (n1, n), components : (n_components, n) => K @ components.T : (n1, n_components) : the embeddings
        return K @ self._components.T
