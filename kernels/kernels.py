import numpy as np


class GaussianKernel:
    def __init__(self, sigma) -> None:
        assert sigma > 0, "sigma parameter must be > 0."
        self.inv_sigma = 1.0 / sigma

    def __call__(self, x1, x2):
        return np.exp(-0.5 * np.square((x1 - x2) * self.inv_sigma))


class LinearKernel:
    def __call__(self, x1, x2):
        return x1 * x2
