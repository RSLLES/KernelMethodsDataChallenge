import unittest
from .svc import SVC
import numpy as np
from data.generate_dumb import gen_data, gen_linearly_separable_data
import matplotlib.pyplot as plt


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
                plt.scatter(X[y == 1, 0], X[y == 1, 1], c='tab:green')
                plt.scatter(X[y == -1, 0], X[y == -1, 1], c='tab:red')
                plt.show()
            else:
                X, y = self.X, self.y
            model = SVC(loss='hinge', penalty='l2', kernel=kernel_type, **kernel_params, verbose=True)
            model.fit(X, y)
            self.assertEqual(model._opt_status, 'optimal')
            # uncomment for visual check (does not work well w/ linear kernel since data is not linearly separable)
            plot_2d_classif(X, y, model.predict(X), model, bound=((-2., 2.), (-2., 2.)))


def plot_hypersurface(ax, x_extent, model: SVC, intercept, color='grey', linestyle='-', alpha=1.):
    x_extent = np.linspace(x_extent[0], x_extent[1], 100)
    xx, yy = np.meshgrid(x_extent, x_extent)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    decision = model.decision_function(xy).reshape(xx.shape) - model._offset + intercept
    ax.contour(xx, yy, decision, colors=color, levels=[0.], alpha=alpha, linestyles=[linestyle])


def plot_2d_classif(X, ytrue, ypred, model: SVC, ax=None, bound=((-1., 1.), (-1., 1.))):
    """
    Plot the SVC separation and margin for dummy 2D data
    """
    if ytrue.dtype == bool:
        ytrue = 1. * ytrue - 1. * ~ytrue
    tp_mask = (ytrue == 1.) & (ypred == 1.)
    fp_mask = (ytrue == -1.) & (ypred == 1.)
    tn_mask = (ytrue == -1.) & (ypred == -1.)
    fn_mask = (ytrue == 1.) & (ypred == -1.)
    masks = [tp_mask, fp_mask, tn_mask, fn_mask]
    mask_labels = ['TP', 'FP', 'TN', 'FN']
    colors = ('tab:green', 'tab:orange', 'tab:blue', 'tab:red')
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(11, 7))
    for i, mask in enumerate(masks):
        ax.scatter(X[mask, 0], X[mask, 1], alpha=0.5, label=mask_labels[i], c=colors[i])

    if model is not None:
        xx = np.array(bound[0])
        plot_hypersurface(ax, bound[0], model, 0.)
        # Plot margin
        ax.scatter(model._support_vecs[:, 0], model._support_vecs[:, 1], label='Support', s=80, facecolors='none',
                   edgecolors='r', color='r')
        print("Number of support vectors = %d" % (len(model._support_vecs)))
        decision_supp = model.decision_function(model._support_vecs) - model._offset
        plot_hypersurface(ax, xx, model, -np.min(decision_supp), linestyle='-.', alpha=0.8)
        plot_hypersurface(ax, xx, model, -np.max(decision_supp), linestyle='--', alpha=0.8)
        # Plot points on the wrong side of the margin
        decision_data = model.decision_function(X) - model._offset
        supp_min = X[(decision_data > np.min(decision_supp)) * (ytrue == -1)]
        supp_max = X[(decision_data < np.max(decision_supp)) * (ytrue == 1)]
        wrong_side_points = np.concatenate([supp_min, supp_max], axis=0)
        ax.scatter(wrong_side_points[:, 0], wrong_side_points[:, 1], label='Beyond the margin', s=80,
                   facecolors='none',
                   edgecolors='grey', color='grey')

    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlim(bound[0])
    ax.set_ylim(bound[1])
    plt.show()
