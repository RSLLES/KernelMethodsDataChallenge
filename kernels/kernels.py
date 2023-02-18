import numpy as np


class GaussianKernel:
    def __init__(self, sigma) -> None:
        assert sigma > 0, "sigma parameter must be > 0."
        self.gamma = 1. / (2 * sigma**2)

    def __call__(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2))


class LinearKernel:
    def __call__(self, x1, x2):
        # different behavior if inputs are matrices : compute the whole gram matrix at once
        if hasattr(x1, 'ndim') and x1.ndim == 2:  # matrix case
            return np.sum(x1[:, None] * x2[None, :], axis=2)
        else:  # scalar or 1D array case
            return np.dot(x1, x2)


class PolynomialKernel:
    def __init__(self, scale, offset, degree):
        self.scale = scale
        self.offset = offset
        self.degree = degree

    def __call__(self, x1, x2):
        # different behavior if inputs are matrices : compute the whole gram matrix at once
        if hasattr(x1, 'ndim') and x1.ndim == 2:  # matrix case
            return np.power(np.sum(x1[:, None] * x2[None, :], axis=2) / self.scale + self.offset, self.degree)
        else:  # scalar or 1D array case
            return np.power(np.dot(x1, x2) / self.scale + self.offset, self.degree)
