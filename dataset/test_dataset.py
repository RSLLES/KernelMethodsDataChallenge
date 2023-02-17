import unittest
from .dataset import Dataset
from kernels.kernels import LinearKernel
import numpy as np


class TestDataset(unittest.TestCase):
    def setUp(self):
        x = np.arange(7)
        y = np.array([1, 0, 0, 1, 1, 0, 1])
        self.d = Dataset(X=x, Y=y, kernel=LinearKernel(), k_folds=3)
        print("Hello")
        self.d.compute_gram_matrix()

    def test_gram_matrix(self):
        np.testing.assert_almost_equal(
            self.d.K,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                    [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
                    [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
                    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
                    [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0],
                ]
            ),
            err_msg="Error with Gram Matrix",
        )

    def test_views(self):
        self.assertEqual(len(self.d), 3, "Dataset length problem.")
        d = iter(self.d)

        ### Fold 1 ###
        K_train, y_train, K_test, y_test = next(d)
        np.testing.assert_almost_equal(
            K_train,
            np.array(
                [
                    [4.0, 6.0, 8.0, 10.0, 12.0],
                    [6.0, 9.0, 12.0, 15.0, 18.0],
                    [8.0, 12.0, 16.0, 20.0, 24.0],
                    [10.0, 15.0, 20.0, 25.0, 30.0],
                    [12.0, 18.0, 24.0, 30.0, 36.0],
                ]
            ),
            err_msg="Error with fold 1.",
        )
        np.testing.assert_almost_equal(
            K_test, np.array([[0.0, 0.0], [0.0, 1.0]]), err_msg="Error with fold 1."
        )
        np.testing.assert_almost_equal(
            y_train, np.array([0, 1, 1, 0, 1]), err_msg="Error with fold 1."
        )
        np.testing.assert_almost_equal(
            y_test, np.array([1, 0]), err_msg="Error with fold 1."
        )

        ### Fold 2 ###
        K_train, y_train, K_test, y_test = next(d)
        np.testing.assert_almost_equal(
            K_train,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 4.0, 5.0, 6.0],
                    [0.0, 4.0, 16.0, 20.0, 24.0],
                    [0.0, 5.0, 20.0, 25.0, 30.0],
                    [0.0, 6.0, 24.0, 30.0, 36.0],
                ]
            ),
            err_msg="Error with fold 2.",
        )
        np.testing.assert_almost_equal(
            K_test, np.array([[4.0, 6.0], [6.0, 9.0]]), err_msg="Error with fold 2."
        )
        np.testing.assert_almost_equal(
            y_train, np.array([1, 0, 1, 0, 1]), err_msg="Error with fold 2."
        )
        np.testing.assert_almost_equal(
            y_test, np.array([0, 1]), err_msg="Error with fold 2."
        )

        ### Fold 3 ###
        K_train, y_train, K_test, y_test = next(d)
        np.testing.assert_almost_equal(
            K_train,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 2.0, 4.0, 6.0],
                    [0.0, 3.0, 6.0, 9.0],
                ]
            ),
            err_msg="Error with fold 3.",
        )
        np.testing.assert_almost_equal(
            K_test,
            np.array(
                [
                    [16.0, 20.0, 24.0],
                    [20.0, 25.0, 30.0],
                    [24.0, 30.0, 36.0],
                ]
            ),
            err_msg="Error with fold 3.",
        )
        np.testing.assert_almost_equal(
            y_train, np.array([1, 0, 0, 1]), err_msg="Error with fold 3."
        )
        np.testing.assert_almost_equal(
            y_test, np.array([1, 0, 1]), err_msg="Error with fold 3."
        )
