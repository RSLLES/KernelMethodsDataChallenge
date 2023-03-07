from kernel_pca import KernelPCA
from data.generate_dumb import gen_circles
from kernels.kernels import GaussianKernel
from matplotlib import pyplot as plt
import numpy as np
import unittest


class KernelPCATest(unittest.TestCase):
    def setUp(self) -> None:
        X, y = gen_circles(300, 2)
        self.X = X
        self.y = y

    def test_kpca(self):
        kernel = GaussianKernel(sigma=0.3)
        kpca = KernelPCA(kernel=kernel)
        indices = np.arange(300)
        np.random.shuffle(indices)
        train_idx, test_idx = indices[:int(0.7 * 300)], indices[int(0.7 * 300):]
        X_train, X_test = self.X[train_idx], self.X[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]

        X_tr_embed = kpca.fit_transform(X_train)
        X_test_embed = kpca.transform(X_test)

        _, ax = plt.subplots(2, 2, figsize=(10, 5))
        ax[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train)
        ax[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        ax[1, 0].scatter(X_tr_embed[:, 0], X_tr_embed[:, 1], c=y_train)
        ax[1, 1].scatter(X_test_embed[:, 0], X_test_embed[:, 1], c=y_test)
        plt.show()
