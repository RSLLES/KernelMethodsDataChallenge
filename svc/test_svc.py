import unittest
from .svc import SVC
from data.generate_dumb import gen_data, gen_linearly_separable_data
from kernels.kernels import GaussianKernel
import numpy as np


class SVCTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        X, Y = gen_data(300)
        self.X = X
        self.y = Y

    def test_l2_hinge(self):
        kernel_params = {
            'poly_scale': 1.,
            'poly_offset': 0.1,
            'poly_degree': 2,
            'rbf_gamma': 0.5,
        }

        for kernel_type in ['linear', 'polynomial', 'rbf']:
            if kernel_type == 'linear':
                X, y = gen_linearly_separable_data(300)
            else:
                X, y = self.X, self.y
            model = SVC(loss='hinge', penalty='l2', kernel=kernel_type, **kernel_params, verbose=True)
            model.fit(X, y)
            self.assertEqual(model._opt_status, 'optimal')
            # uncomment for visual check (does not work well w/ linear kernel since data is not linearly separable)
            # from .plotting import plot_2d_classif
            # plot_2d_classif(X, y, model.predict(X), model, bound=((-2., 2.), (-2., 2.)))

    def test_precomp_kernel(self):
        """
        This just needs to run without errors.
        """
        kernel_func = GaussianKernel(sigma=2)
        K_full = kernel_func(self.X, self.X)
        K_train = K_full[:250, :250]
        X_test = self.X[250:]

        model = SVC(loss='hinge', penalty='l2', kernel=kernel_func, precomputed_kernel=K_train)
        model.fit(self.X[:250], self.y[:250])
        model.predict(X_test)
        # from .plotting import plot_2d_classif
        # plot_2d_classif(self.X, self.y, model.predict(self.X), model)
