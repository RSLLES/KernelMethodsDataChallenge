import unittest
from .svc import SVC
from data.generate_dumb import gen_data, gen_linearly_separable_data
from kernels.kernels import GaussianKernel, LinearKernel, PolynomialKernel
from dataset.dataset import Dataset
import numpy as np

class SVCTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.tests = [
            [LinearKernel(), gen_linearly_separable_data(300)],
            [GaussianKernel(sigma=1), gen_data(300)],
            [PolynomialKernel(scale=1.0, offset=0.1, degree=2), gen_data(300)],
        ]

    def test_svc(self):
        for kernel, (X, y) in self.tests:
            print(f"Test with {kernel.__class__.__name__}")
            ds = Dataset(X=X, y=y, k_folds=3, shuffle=True)
            for k, (X_train, y_train, X_test, y_test) in enumerate(ds):
                model = SVC(kernel=kernel)
                model.fit(X=X_train, y=y_train)
                self.assertEqual(model._opt_status, "optimal")
                accuracy, precision, recall = model.score(
                    X=X_test, y=np.array(y_test).astype(bool)
                )
                print(
                    f"Fold {k+1} : Accuracy = {100*accuracy:0.2f}%, "
                    f"Precision = {100*precision:0.2f}%, Recall = {100*recall:0.2f}%"
                )

            # uncomment for visual check (does not work well w/ linear kernel since data is not linearly separable)
            # from .plotting import plot_2d_classif

            # plot_2d_classif(
            #     X, y, model.predict(X), model, bound=((-1.0, 1.0), (-1.0, 1.0))
            # )
