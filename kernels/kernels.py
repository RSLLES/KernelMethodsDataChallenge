import numpy as np


class GaussianKernel:
    def __init__(self, sigma) -> None:
        assert sigma > 0, "sigma parameter must be > 0."
        self.gamma = 1. / (2 * sigma**2)

    def __call__(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2))


class LinearKernel:
    def __call__(self, x1, x2):
        return np.sum(x1[:, None] * x2[None, :], axis=2)
