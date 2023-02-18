import unittest
from .svc import SVC
import cvxpy as cvx
import numpy as np


class SVCTest(unittest.TestCase):
    def setUp(self) -> None:
        x1 = [
            [1.0, 1.0],
            [0.5, 0.5],
            [0., 0.5]
        ]
        y1 = [1, 1, 1]

        x2 = [
            [-1.0, -1.0],
            [-0.5, -0.5],
            [-0., -0.5]
        ]
        y2 = [-1, -1, -1]

        self.X = np.array(x1 + x2)
        self.y = np.array(y1 + y2)

    def test_l2_hinge(self):
        model = SVC(loss='hinge', penalty='l2', kernel='linear')
        model.fit(self.X, self.y)
        self.assertEqual(model._opt_status, 'optimal')
