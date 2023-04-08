import numpy as np
import cvxpy as cvx
import sys
from typing import Union, Callable, Optional, List
from kernels.kernels import LinearKernel, GaussianKernel, PolynomialKernel
import itertools
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import warnings


class SVC:
    """
    Support Vector Classifier implementation.
    """

    def __init__(
        self,
        kernel: Callable,
        C: float = 1.0,
        loss: str = "hinge",
        penalty: str = "l2",
        epsilon: float = 10 * sys.float_info.epsilon,
        verbose: bool = False,
    ):
        """
        Initialize the classifier.

        :param kernel: callable taking two NumPy arrays as input and computing pairwise positive definite kernel
        :param C: inverse regularization parameter (lower is more regularization)
        :param loss: the loss to use in primal formulation, one of 'hinge', 'squared_hinge'
        :param penalty: 'l1' or 'l2'
        :param epsilon: a numerical tolerance parameter on solutions of the dual problem
        :param verbose: if True, will print some info when fitting
        """
        self.C = C
        self.loss = loss.lower()
        self.kernel = kernel.lower() if isinstance(kernel, str) else kernel
        self.penalty = penalty.lower()
        self.epsilon = epsilon
        self.verbose = verbose
        if not isinstance(kernel, str):
            self.kernel.set_verbose(verbose)
        warnings.simplefilter("ignore")

    def fit(self, X: List, y: np.ndarray):
        # assumption : X *always* contains samples (i.e. X[0] is the first sample and so on)
        # however, if we have a precomputed kernel, we don't recompute it
        assert len(X) == len(
            y
        ), f"Unmatched sizes : len(X) = {len(X)}, len(y) = {len(y)}"
        if self.loss == "hinge" and self.penalty == "l2":
            self._fit_l2_hinge(X, y)
        else:
            raise AttributeError(
                f"Method using loss {self.loss} and penalty {self.penalty} is not implemented yet."
            )

    def predict(self, X: list) -> np.ndarray:
        """
        Compute 0/1 labels for samples.

        :param X: The data points to classify
        :return: a vector containing the label value (0/1) for each data point
        """
        return (self.decision_function(X) > 0).astype(int)

    def decision_function(self, X: list) -> np.ndarray:
        """
        Compute the classification decision function.

        :param X: The data points to classify
        :return: a vector containing the decision function value for each data point
        """
        if self.kernel == "precomputed":
            kernel_eval = X[:, self.nonzero_alpha_idx].squeeze()
            return np.sum(self._separating_weights * kernel_eval, axis=1) + self._offset
        kernel_eval = self.kernel(self._separating_vecs, X)
        return (
            np.sum(self._separating_weights[:, None] * kernel_eval, axis=0)
            + self._offset
        )

    @staticmethod
    def _check_kernel(K):
        if len(K.shape) == 2 and K.shape[0] == K.shape[1]:
            return
        raise ValueError(
            f"The computed kernel should be a sqaure matrix (NxN), found {K.shape}"
        )

    @staticmethod
    def _check_labels(y):
        """
        Make sure labels are -1 and 1.

        :param y: scalar or 1D array of labels
        :return: array of the same shape as y but with values in {-1, 1}
        """
        if isinstance(y, list):
            y = np.array(y)
        if np.all(y**2 == 1.0):
            return y
        bool_y = y > 0
        return 1.0 * bool_y - 1.0 * ~bool_y

    def _fit_l2_hinge(self, X: List, y: np.ndarray) -> None:
        """
        Fit the SVC when loss='hinge' and penalty='l2' using the dual formulation (Homework 2).

        :param X: the matrix of samples
        :param y: the ground-truth labels (can be boolean, 0/1, or -1/1)
        """
        n = len(y)
        y = self._check_labels(y)

        # Compute Kernel
        K = self.kernel(X) if self.kernel != "precomputed" else X
        SVC._check_kernel(K)

        # Dual problem definition
        alpha = cvx.Variable(
            shape=n, name="alpha"
        )  # alpha represents the dual variables
        if self.verbose:
            print(
                "[SVC.fit] Computing the Hessienne and checking if it is semi positive definite..."
            )
        hess = cvx.Parameter(
            shape=(n, n),
            name="hessian",
            value=y[:, None] * K * y[None, :],  # diag(y) @ K @ diag(y)
            PSD=True,
        )
        dual_objective = cvx.Minimize(
            0.5 * cvx.quad_form(x=alpha, P=hess) - cvx.sum(alpha)
        )

        # Inequality constraints in vector form
        _y = cvx.Parameter(shape=n, name="y", value=y)
        _C = cvx.Parameter(name="C", value=self.C, nonneg=True)
        dual_constraints = [
            -alpha <= 0.0,
            alpha <= _C,
            alpha @ _y == 0.0,
        ]

        if self.verbose:
            print("[SVC.fit] Solving dual problem...")
        dual_problem = cvx.Problem(dual_objective, dual_constraints)
        dual_problem.solve(
            solver=cvx.OSQP,
            eps_abs=max(self.epsilon / 100, 10 * sys.float_info.epsilon),
        )

        if self.verbose:
            if dual_problem.status != "optimal":
                print(
                    f"[WARNING] : SVC fit may be bad, CVXPy returned status '{dual_problem.status}'"
                )
            print(f"Done solving dual problem for hinge-l2 SVC")
        self._opt_status = (
            dual_problem.status
        )  # store the opt status for debugging purposes

        self._alpha = alpha.value
        # now, get vectors needed for separation
        self.nonzero_alpha_idx = self._alpha > self.epsilon
        self.support_idx = self.nonzero_alpha_idx & (
            self._alpha < (self.C - self.epsilon)
        )
        self._support_vecs = [X[i] for i in range(len(X)) if self.support_idx[i]]
        self._separating_weights = (
            self._alpha[self.nonzero_alpha_idx] * y[self.nonzero_alpha_idx]
        )
        self._separating_vecs_idx = [
            i for i in range(len(X)) if self.nonzero_alpha_idx[i]
        ]
        self._separating_vecs = [X[i] for i in self._separating_vecs_idx]

        # f is the RKHS function optimal for the primal problem
        def f(x_idx):
            if self.kernel == "precomputed":
                _kern_eval = K[self.nonzero_alpha_idx, :][:, x_idx]
            else:
                _kern_eval = self.kernel(self._separating_vecs, X[x_idx])
            return np.dot(self._separating_weights, _kern_eval)

        if len(self._support_vecs) == 0:
            print("WARNING : no support vectors")
            # this can happen when all dual coefficients are close to C
            neg_idx = np.argmax(
                [
                    f(self._separating_vecs_idx[i])
                    for i in range(len(self._separating_vecs))
                    if y[i] == -1
                ]
            )
            pos_idx = np.argmin(
                [
                    f(self._separating_vecs_idx[i])
                    for i in range(len(self._separating_vecs))
                    if y[i] == 1
                ]
            )
            self._support_vecs = [
                self._separating_vecs[neg_idx],
                self._separating_vecs[pos_idx],
            ]
            self._offset = -0.5 * (
                f(self._separating_vecs_idx[neg_idx])
                + f(self._separating_vecs_idx[pos_idx])
            )
        else:
            # compute b (hyperplane offset) : for x0 a support vector, y0 (f(x0) + b) = -1
            idx = np.where(self.support_idx)[0][0]
            f_x0 = f(idx)
            self._offset = y[self.support_idx][0] - f_x0

        if self.verbose:
            print(
                f"# Support vectors    : {len(self._support_vecs)}\n"
                f"# Separating vectors : {len(self._separating_vecs)}"
            )
        self._rkhs_norm = np.sqrt(np.dot(self._alpha, hess.value @ self._alpha))


def score(svc, X: list, y: np.ndarray, score_type: Optional[str] = None):
    """
    Compute accuracy, precision and recall, or one of them.

    :param X: the samples to predict labels of
    :param y: the ground-truth labels of X (should be a binary array)
    :param score_type: 'accuracy', 'precision' or 'recall' to select which score to compute
    :return: a tuple of length 3 : (accuracy, precision, recall) or one of them if 'score_type' was passed
    """
    if y.dtype != bool:
        raise ValueError(f"y should be a binary array, found type {y.dtype}.")

    score = score_type.lower() if score_type else None
    y_score = svc.decision_function(X)
    y_pred = y_score > 0

    tp = np.sum(y_pred & y)
    tn = np.sum(~y_pred & ~y)
    fp = np.sum(y_pred & ~y)
    fn = np.sum(~y_pred & y)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) >= 1 else 0.0
    recall = tp / (tp + fn) if (tp + fn) >= 1 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    rocauc = roc_auc_score(y_true=y, y_score=y_score)

    if score == "accuracy":
        return accuracy
    if score == "precision":
        return precision
    if score == "recall":
        return recall
    if score == "f1":
        return f1
    if score == "rocauc":
        return rocauc

    return accuracy, precision, recall, f1, rocauc
