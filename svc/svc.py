import numpy as np
import cvxpy as cvx
import sys
from typing import Union, Callable, Optional
from kernels.kernels import LinearKernel, GaussianKernel, PolynomialKernel


class SVC:
    """
    Support Vector Classifier implementation.
    Follows sklearn's API.
    """
    def __init__(self,
                 C: float = 1.,
                 loss: str = 'hinge',
                 kernel: Union[Callable, str] = 'linear',
                 penalty: str = 'l2',
                 epsilon: float = 10 * sys.float_info.epsilon,
                 rbf_gamma: Optional[float] = None,
                 poly_scale: Optional[float] = None,
                 poly_offset: Optional[float] = None,
                 poly_degree: Optional[float] = None,
                 verbose: bool = False
                 ):
        """
        Initialize the classifier.

        :param C: inverse regularization parameter (lower is more regularization)
        :param loss: the loss to use in primal formulation, one of 'hinge', 'squared_hinge'
        :param kernel: callable taking two NumPy arrays as input and computing pairwise positive definite kernel
            evaluations, or a string : one of 'linear', 'polynomial', 'rbf', 'precomputed'.
                - 'linear': linear kernel x1.dot(x2)
                - 'polynomial': polynomial kernel (x1.dot(x2) / a + c)**d
                - 'rbf': Radial Basis Function kernel exp(-gamma * norm(x1 - x2)**2) (Euclidian norm)
                - 'precomputed': a matrix is used as a precomputed kernel. Must be provided when calling fit()
        :param penalty: 'l1' or 'l2'
        :param epsilon: a numerical tolerance parameter on solutions of the dual problem
        :param rbf_gamma: the 'gamma' factor in the RBF kernel. Lower means distances get shrunk,
         resulting in softer classification. Ignored if kernel is not 'rbf'
        :param poly_scale: the 'a' factor in the polynomial kernel. Must be non-negative. Values < 1 dilate the
         distance, values > 1 shrink the distance. Ignored if kernel is not 'polynomial'
        :param poly_offset: the 'c' factor in the polynomial kernel. Must be non-negative. Higher values tend to
         smoother classification (prevents overfitting but may smoothen "too much").
         Ignored if kernel is not 'polynomial'
        :param poly_degree: the 'd' factor in the polynomial kernel. Higher means small distances get shrunk and large
         distances get dilated. Ignored if kernel is not 'polynomial'
        """
        self.C = C
        self.loss = loss.lower()
        self.kernel = kernel.lower() if isinstance(kernel, str) else kernel
        self.penalty = penalty.lower()
        self.epsilon = epsilon
        self.rbf_gamma = rbf_gamma
        self.poly_scale = poly_scale
        self.poly_offset = poly_offset
        self.poly_degree = poly_degree
        self.verbose = verbose

        if self.kernel == 'linear':
            self.kernel = LinearKernel()
        if self.kernel == 'rbf':
            if self.rbf_gamma is None:
                raise ValueError('Requested kernel \'rbf\', but argument \'rbf_gamma\' was not provided')
            self.kernel = GaussianKernel(sigma=1. / np.sqrt(2 * self.rbf_gamma))
        if self.kernel == 'polynomial':
            if self.poly_scale is None or self.poly_offset is None or self.poly_degree is None:
                raise ValueError('Requested kernel \'polynomial\', but arguments '
                                 '\'poly_scale\', \'poly_offset\' and \'poly_degree\' were not provided')
            self.kernel = PolynomialKernel(self.poly_scale, self.poly_offset, self.poly_degree)

    def fit(self, X, y):
        if self.loss == 'hinge' and self.penalty == 'l2':
            self._fit_l2_hinge(X, y)
        # todo : other cases !

    def predict(self, X) -> np.ndarray:
        """
        Compute 0/1 labels for samples.

        :param X: The data points to classify.
        If kernel='precomputed', X should be a precomputed kernel of the form (k(xt_i, xp_j)) where xt_i are
        the elements of the training sets and xp_j are the elements of the set to predict labels on.

        :return: a vector containing the label value (0/1) for each data point
        """
        return 2 * ((self.decision_function(X) + self._offset) > 0) - 1

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the classification decision function.

        :param X: The data points to classify.
        If kernel='precomputed', X should be a precomputed kernel of the form (k(xt_i, xp_j)) where xt_i are
        the elements of the training sets and xp_j are the elements of the set to predict labels on.

        :return: a vector containing the decision function value for each data point
        """
        if self.kernel != 'precomputed':
            kernel_eval = self.kernel(self._separating_vecs, X)
        else:
            kernel_eval = X[self._separating_vecs]  # if kernel='precomputed', _separating_vecs contains the indexes

        return np.sum(self._separating_weights[:, None] * kernel_eval, axis=0)

    def _fit_l2_hinge(self, 
                      X: np.ndarray,
                      y: np.ndarray) -> None:
        """
        Fit the SVC when loss='hinge' and penalty='l2' using the dual formulation (Homework 2).
        
        :param X: the matrix of samples, or the precomputed kernel if kernel='precomputed'
        :param y: the ground-truth labels
        """
        n = len(y)
        K = X if self.kernel == 'precomputed' else self.kernel(X, X)

        # Dual problem definition
        alpha = cvx.Variable(shape=n, name='alpha')  # alpha represents the dual variables
        hess = cvx.Parameter(shape=(n, n),
                             name='hessian',
                             value=y[:, None] * K * y[None, :],  # diag(y) @ K @ diag(y)
                             PSD=True)
        dual_objective = cvx.Minimize(0.5 * cvx.quad_form(x=alpha, P=hess) - cvx.sum(alpha))

        # Inequality constraints in vector form
        _y = cvx.Parameter(shape=n, name='y', value=y)
        _C = cvx.Parameter(name='C', value=self.C, nonneg=True)
        dual_constraints = [
            -alpha <= 0.,
            alpha <= _C,
            alpha @ _y == 0.,
        ]

        dual_problem = cvx.Problem(dual_objective, dual_constraints)
        dual_problem.solve()

        if self.verbose and dual_problem.status != 'optimal':
            print(f'[WARNING] : SVC fit may be bad, CVXPy returned status \'{dual_problem.status}\'')
        self._opt_status = dual_problem.status  # store the opt status for debugging purposes

        self._alpha = alpha.value
        # now, get vectors needed for separation
        nonzero_alpha_idx = self._alpha > self.epsilon
        support_idx = nonzero_alpha_idx & (self._alpha < (self.C - self.epsilon))
        # if kernel='precomputed', store the indices instead of the vectors (we don't have the vectors)
        self._support_vecs = X[support_idx] if self.kernel != 'precomputed' else support_idx
        self._separating_weights = self._alpha[nonzero_alpha_idx] * y[nonzero_alpha_idx]
        self._separating_vecs = X[nonzero_alpha_idx] if self.kernel != 'precomputed' else nonzero_alpha_idx

        # compute b (hyperplane offset) : for x0 a support vector, y0 (f(x0) + b) = -1
        if self.kernel != 'precomputed':
            f_x0 = np.dot(self._separating_weights, self.kernel(self._separating_vecs, self._support_vecs[0][None, :]))
        else:
            f_x0 = np.dot(self._separating_weights, self.kernel[support_idx][0][nonzero_alpha_idx])  # todo : check this
        self._offset = y[support_idx][0] - f_x0
        self._rkhs_norm = np.sqrt(np.dot(self._alpha, hess.value @ self._alpha))

    def score(self, X, y):
        raise NotImplementedError
