import os

import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs


def main(method_dir: str, show: bool = True, N: int = 1000) -> None:
    """
    Creates a heatmap of the Bayesian Optimization results.

    Args:
        method_dir (str): The directory containing the logs file.
        show (bool): Whether or not to display the plot. Defaults to True.
        N (int): Number of points for plotting the heatmap. Defaults to 1000.
    """

    path = os.path.join(method_dir, "logs.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Please provide a folder containing a BaysianOPtimization log file."
        )

    # Load the logs from the file and create BayesianOptimization object with the optimization parameters
    pbounds = {
        "depth": (1, 9),
        "log_lambd": (-1.0, 1.0),
    }
    opt = BayesianOptimization(f=None, pbounds=pbounds)
    load_logs(opt, logs=[path])

    # Check whether there are only two parameters in the logged data
    if len(opt.space.keys) != 2:
        raise ValueError("This script can only handle 2 parameter functions.")

    print("{} points loaded.".format(len(opt.space)))

    # Fitting the gaussian process with params and target
    opt._gp.fit(opt.space.params, opt.space.target)

    # Get bounds of the parameters and create a grid using linspace
    bounds = opt.space.bounds
    xbounds, ybounds = bounds[0], bounds[1]
    x = np.linspace(xbounds[0], xbounds[1], N)
    y = np.linspace(ybounds[0], ybounds[1], N)

    # Create a meshgrid of parameters and predict the target function
    X, Y = np.meshgrid(x, y)
    Z = 100 * opt._gp.predict(np.vstack([X.ravel(), Y.ravel()]).T).T

    # Reshape the arrays for plotting
    X = X.reshape((N, N))
    Y = Y.reshape((N, N))
    Z = Z.reshape((N, N))

    # Get the index with the highest value of target function
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)

    # Plot the heatmap with the points and maximum index
    plt.pcolormesh(X, Y, Z, cmap="inferno")
    plt.colorbar()
    plt.scatter(opt.space.params[:, 0], opt.space.params[:, 1], c="#3185FC")
    plt.scatter(
        X[max_idx],
        Y[max_idx],
        s=100,
        marker="x",
        c="#38A3A5",
        label="Best theoretical point",
    )
    plt.xlabel(opt.space.keys[0])
    plt.ylabel(opt.space.keys[1])

    # Set the mouse cursor
    ax = plt.gca()

    def format_coord(x, y):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        col = int(np.floor((x - x0) / float(x1 - x0) * X.shape[1]))
        row = int(np.floor((y - y0) / float(y1 - y0) * Y.shape[0]))
        if col >= 0 and col < Z.shape[1] and row >= 0 and row < Z.shape[0]:
            z = Z[row, col]
            return "x=%1.4f, y=%1.4f, z=%1.4f" % (x, y, z)
        else:
            return "x=%1.4f, y=%1.4f" % (x, y)

    ax.format_coord = format_coord

    # add the star annotation to the plot
    max_point = opt.max["params"]
    xy = (max_point[opt.space.keys[0]], max_point[opt.space.keys[1]])
    plt.scatter(xy[0], xy[1], s=100, marker="*", color="red", label="Best tested point")

    # Set the title of plot with maximum value obtained
    plt.title(
        f"Max tested at ({opt.space.keys[0]}={xy[0]:.2f}, {opt.space.keys[1]} = {xy[1]:.2f}) : f = {opt.max['target']*100:.3f}%"
    )

    # Add legend
    plt.legend()

    # Save the plot to a png file
    plt.savefig(os.path.join(method_dir, "heatmap.png"))

    print("Plot saved at {}".format(os.path.join(method_dir, "heatmap.png")))

    if show:
        # Show the plot if asked by the user
        plt.show()


import fire

if __name__ == "__main__":
    fire.Fire(main)
