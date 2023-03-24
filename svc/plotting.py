from .svc import SVC
import numpy as np
from matplotlib import pyplot as plt


def plot_hypersurface(
    ax, x_extent, model: SVC, intercept, color="grey", linestyle="-", alpha=1.0
):
    x_extent = np.linspace(x_extent[0], x_extent[1], 100)
    xx, yy = np.meshgrid(x_extent, x_extent)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    decision = model.decision_function(xy).reshape(xx.shape) - model._offset + intercept
    ax.contour(
        xx,
        yy,
        decision,
        colors=color,
        levels=[0.0],
        alpha=alpha,
        linestyles=[linestyle],
    )


def plot_2d_classif(
    X, ytrue, ypred, model: SVC, ax=None, bound=((-1.0, 1.0), (-1.0, 1.0))
):
    """
    Plot the SVC separation and margin for dummy 2D data
    """
    if ytrue.dtype == bool:
        ytrue = 1.0 * ytrue - 1.0 * ~ytrue
    else:
        ytrue = SVC._check_labels(ytrue)
    if ypred.dtype == bool:
        ypred = 1.0 * ypred - 1.0 * ~ypred
    else:
        ypred = SVC._check_labels(ypred)
    tp_mask = (ytrue == 1.0) & (ypred == 1.0)
    fp_mask = (ytrue == -1.0) & (ypred == 1.0)
    tn_mask = (ytrue == -1.0) & (ypred == -1.0)
    fn_mask = (ytrue == 1.0) & (ypred == -1.0)
    masks = [tp_mask, fp_mask, tn_mask, fn_mask]
    mask_labels = ["TP", "FP", "TN", "FN"]
    colors = ("tab:green", "tab:orange", "tab:blue", "tab:red")
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(11, 7))
    for i, mask in enumerate(masks):
        ax.scatter(X[mask, 0], X[mask, 1], alpha=0.5, label=mask_labels[i], c=colors[i])

    if model is not None:
        xx = np.array(bound[0])
        plot_hypersurface(ax, bound[0], model, 0.0)
        # Plot margin
        ax.scatter(
            np.array(model._support_vecs)[:, 0],
            np.array(model._support_vecs)[:, 1],
            label="Support",
            s=80,
            facecolors="none",
            edgecolors="r",
            color="r",
        )
        print("Number of support vectors = %d" % (len(model._support_vecs)))
        decision_supp = model.decision_function(model._support_vecs) - model._offset
        plot_hypersurface(
            ax, xx, model, -np.min(decision_supp), linestyle="-.", alpha=0.8
        )
        plot_hypersurface(
            ax, xx, model, -np.max(decision_supp), linestyle="--", alpha=0.8
        )
        # Plot points on the wrong side of the margin
        decision_data = model.decision_function(X) - model._offset
        supp_min = X[(decision_data > np.min(decision_supp)) * (ytrue == -1)]
        supp_max = X[(decision_data < np.max(decision_supp)) * (ytrue == 1)]
        wrong_side_points = np.concatenate([supp_min, supp_max], axis=0)
        ax.scatter(
            wrong_side_points[:, 0],
            wrong_side_points[:, 1],
            label="Beyond the margin",
            s=80,
            facecolors="none",
            edgecolors="grey",
            color="grey",
        )

    ax.legend(loc="upper left")
    ax.grid()
    ax.set_xlim(bound[0])
    ax.set_ylim(bound[1])
    plt.show()
