import unittest
from .dataset import Dataset
import numpy as np


class TestDataset(unittest.TestCase):
    def test_train_solo(self):
        x = np.arange(7)
        d = Dataset(X=x, k_folds=1)
        self.assertEqual(len(d), 1, "Dataset length problem.")
        d = iter(d)

        X_train, y_train, X_test, y_test = next(d)
        np.testing.assert_almost_equal(X_train, x)
        self.assertIsNone(y_train)
        self.assertIsNone(X_test)
        self.assertIsNone(y_test)

    def test_train_without_cv(self):
        x = np.arange(7)
        y = np.array([1, 0, 0, 1, 1, 0, 1])
        d = Dataset(X=x, y=y, k_folds=1)
        self.assertEqual(len(d), 1, "Dataset length problem.")
        d = iter(d)

        X_train, y_train, X_test, y_test = next(d)
        np.testing.assert_almost_equal(y_train, y)
        np.testing.assert_almost_equal(X_train, x)
        self.assertIsNone(X_test)
        self.assertIsNone(y_test)

    def test_train_and_cv(self):
        x = np.arange(7)
        y = np.array([1, 0, 0, 1, 1, 0, 1])
        d = Dataset(X=x, y=y, k_folds=3)
        self.assertEqual(len(d), 3, "Dataset length problem.")
        d = iter(d)

        ### Fold 1 ###
        X_train, y_train, X_test, y_test = next(d)
        np.testing.assert_almost_equal(X_train, np.array([2, 3, 4, 5, 6]))
        np.testing.assert_almost_equal(
            X_test, np.array([0, 1]), err_msg="Error with fold 1."
        )
        np.testing.assert_almost_equal(
            y_train, np.array([0, 1, 1, 0, 1]), err_msg="Error with fold 1."
        )
        np.testing.assert_almost_equal(
            y_test, np.array([1, 0]), err_msg="Error with fold 1."
        )

        ### Fold 2 ###
        X_train, y_train, X_test, y_test = next(d)
        np.testing.assert_almost_equal(
            X_train, np.array([0, 1, 4, 5, 6]), err_msg="Error with fold 2."
        )
        np.testing.assert_almost_equal(
            X_test, np.array([2, 3]), err_msg="Error with fold 2."
        )
        np.testing.assert_almost_equal(
            y_train, np.array([1, 0, 1, 0, 1]), err_msg="Error with fold 2."
        )
        np.testing.assert_almost_equal(
            y_test, np.array([0, 1]), err_msg="Error with fold 2."
        )

        ### Fold 3 ###
        X_train, y_train, X_test, y_test = next(d)
        np.testing.assert_almost_equal(
            X_train, np.array([0, 1, 2, 3]), err_msg="Error with fold 3."
        )
        np.testing.assert_almost_equal(
            X_test, np.array([4, 5, 6]), err_msg="Error with fold 3."
        )
        np.testing.assert_almost_equal(
            y_train, np.array([1, 0, 0, 1]), err_msg="Error with fold 3."
        )
        np.testing.assert_almost_equal(
            y_test, np.array([1, 0, 1]), err_msg="Error with fold 3."
        )
