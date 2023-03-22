import numpy as np
from kernels.kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self, sigma) -> None:
        super().__init__()
        assert sigma > 0, "sigma parameter must be > 0."
        self.gamma = 1.0 / (2 * sigma**2)

    def kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2))


class LinearKernel(Kernel):
    def kernel(self, x1, x2):
        return np.dot(x1, x2)


class PolynomialKernel(Kernel):
    def __init__(self, scale, offset, degree):
        super().__init__()
        self.scale = scale
        self.offset = offset
        self.degree = degree

    def kernel(self, x1, x2):
        return np.power(np.dot(x1, x2) / self.scale + self.offset, self.degree)
