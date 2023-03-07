import numpy as np


class KernelCenterer:
    """
    Heavily inspired from sklearn source :
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_data.py

    Allows centering kernels, notably perform the (I-U) @ K @ (I-U) operation to center RKHS embeddings, and computing
    KernelPCA embeddings
    """
    def __init__(self):
        self.K_fit_cols = None  # mean of columns
        self.K_fit_all = None  # mean of matrix (float)

    def fit(self, K: np.ndarray) -> None:
        """
        Fit the KernelCenterer.

        :param K: a symmetric PSD kernel matrix
        """
        if K.shape[0] != K.shape[1]:
            raise ValueError(f'Kernel matrix is not symmetric, received shape is {K.shape}')

        n = len(K)
        self.K_fit_cols = np.sum(K, axis=0) / n
        self.K_fit_all = np.sum(self.K_fit_cols) / n

    def transform(self, K: np.ndarray) -> np.ndarray:
        """
        Center the kernel matrix K.

        Note : when K is the same matrix as was used in fit(), this is exactly equivalent to computing the product
        (I-U) @ K @ (I-U) where U is a matrix filled with 1 / len(K)
        """
        if self.K_fit_cols is None:
            raise ValueError('KernelCenterer object is not fitted')

        K_pred_rows = (np.sum(K, axis=1) / len(self.K_fit_cols))[:, None]  # mean of rows

        _K = K - self.K_fit_cols
        _K -= K_pred_rows
        _K += self.K_fit_all

        return _K

    def fit_transform(self, K: np.ndarray) -> np.ndarray:
        """
        Fit the KernelCenterer and return the centered input kernel.
        """
        self.fit(K)
        return self.transform(K)
