import unittest
from .svc import SVC
from data.generate_dumb import gen_data, gen_linearly_separable_data


class SVCTest(unittest.TestCase):
    def setUp(self) -> None:
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
            # (do 'from plotting import plot_2d_classif')
            # plot_2d_classif(X, y, model.predict(X), model, bound=((-2., 2.), (-2., 2.)))
