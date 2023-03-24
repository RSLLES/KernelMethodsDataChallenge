import unittest
from .svc import SVC, score
from sklearn.svm import SVC as SVCSKLearn
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

    def test_vs_sklearn(self):
        for kernel, (X, y) in self.tests:
            print(f"Test with {kernel.__class__.__name__}")
            ds = Dataset(X=X, y=y, k_folds=3, shuffle=True)
            for k, (X_train, y_train, X_test, y_test) in enumerate(ds):
                # Ours
                model = SVC(kernel=kernel)
                model.fit(X=X_train, y=y_train)
                self.assertEqual(model._opt_status, "optimal")

                y_pred = model.predict(X_test)
                accuracy, precision, recall, f1 = score(
                    svc=model, X=X_test, y=np.array(y_test).astype(bool)
                )
                print(
                    f"[Ours] Fold {k+1}/3 : Accuracy = {100*accuracy:0.2f}%, "
                    f"Precision = {100*precision:0.2f}%, Recall = {100*recall:0.2f}%, "
                    f"f1 = {100 * f1:0.2f}%"
                )

                # Sklearn
                model_sk = SVCSKLearn(kernel=kernel)
                model_sk.fit(X=X_train, y=y_train)

                y_pred_sk = model_sk.predict(X_test)
                accuracy, precision, recall, f1 = score(
                    svc=model_sk, X=X_test, y=np.array(y_test).astype(bool)
                )
                print(
                    f"[SKLearn] Fold {k+1}/3 : Accuracy = {100*accuracy:0.2f}%, "
                    f"Precision = {100*precision:0.2f}%, Recall = {100*recall:0.2f}%, "
                    f"f1 = {100 * f1:0.2f}%"
                )

                # Compare predictions
                np.testing.assert_equal(
                    np.array(y_pred).astype(bool), np.array(y_pred_sk).astype(bool)
                )

    # def test_svc(self):
    #     for kernel, (X, y) in self.tests:
    #         print(f"Test with {kernel.__class__.__name__}")
    #         ds = Dataset(X=X, y=y, k_folds=3, shuffle=True)
    #         for k, (X_train, y_train, X_test, y_test) in enumerate(ds):
    #             model = SVC(kernel=kernel)
    #             model.fit(X=X_train, y=y_train)
    #             self.assertEqual(model._opt_status, "optimal")
    #             accuracy, precision, recall, f1 = score(
    #                 svc=model, X=X_test, y=np.array(y_test).astype(bool)
    #             )
    #             print(
    #                 f"Fold {k+1} : Accuracy = {100*accuracy:0.2f}%, "
    #                 f"Precision = {100*precision:0.2f}%, Recall = {100*recall:0.2f}%, "
    #                 f"f1 = {100 * f1:0.2f}%"
    #             )

    # uncomment for visual check (does not work well w/ linear kernel since data is not linearly separable)
    # from .plotting import plot_2d_classif

    # plot_2d_classif(
    #     X, y, model.predict(X), model, bound=((-1.0, 1.0), (-1.0, 1.0))
    # )
