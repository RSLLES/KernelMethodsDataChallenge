import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def sgm(x, t, sigma):
    """
    This function is an implementation of the sigmoid function
    taking three arguments: x, t, and sigma.
    It is defined as 1/(1+e^(-sigma*(x-t))).
    """
    return 1.0 / (1.0 + np.exp(-sigma * (x - t)))


def gen_data(N):
    """
    This function generates data for a binary classification problem.
    It takes in an integer N as an argument, which determines the number of data points to generate.
    """
    X = 2 * np.random.rand(N, 2) - 1
    dist = np.einsum("Nd,Nd->N", X, X)
    p = sgm(x=dist, t=0.4, sigma=10)
    Y = np.random.rand(N) > p
    return [x for x in X], Y


def gen_linearly_separable_data(
    N,
    mu_positive: np.ndarray = np.array([0.5, 0.0]),
    sigma_positive: float = 0.2,
    mu_negative: np.ndarray = np.array([-0.5, 0.0]),
    sigma_negative: float = 0.2,
):
    """
    This function generates linearly separable Gaussian data for a binary classification problem.

    :param N: number of points to generate (if odd, will generate N-1 points)
    :param mu_positive: mean of the positive class
    :param mu_negative: mean of the negative class
    :param sigma_positive: 1D deviation of the positive class gaussian
    :param sigma_negative: 1D deviation of the negative class gaussian
    """
    points_per_class = N // 2
    positive_points = np.zeros(shape=(points_per_class, 2))
    negative_points = np.zeros(shape=(points_per_class, 2))
    for i in range(points_per_class):
        positive_points[i] = mu_positive + np.random.randn(2) * sigma_positive
        negative_points[i] = mu_negative + np.random.randn(2) * sigma_negative

    y = np.concatenate((np.ones(points_per_class), np.zeros(points_per_class)))
    X = np.concatenate((positive_points, negative_points))
    return [x for x in X], y.astype(bool)


def gen_circles(N, n_circles):
    """
    Generate N 2D data points in the shape of concentric circles.

    :param N: number of points
    :param n_circles: number of circles to draw
    :return: an array of shape (N, 2) containing the points, and an array of integers of shape (N,)
     containing the labels (index of circle of the corresponding point)
    """
    points = np.zeros((N, 2), dtype=np.float32)
    labels = np.zeros(N, dtype=int)
    start_radius = 1.0
    points_per_circle = N // n_circles
    for i in range(n_circles):
        radius = start_radius + i
        theta = np.linspace(0.0, 2 * np.pi, num=points_per_circle, endpoint=False)
        theta += 0.01 * np.random.randn(len(theta))
        r = np.full(
            shape=points_per_circle, fill_value=radius
        ) + 0.05 * np.random.randn(points_per_circle)
        points[i * points_per_circle : (i + 1) * points_per_circle, 0] = r * np.cos(
            theta
        )
        points[i * points_per_circle : (i + 1) * points_per_circle, 1] = r * np.sin(
            theta
        )
        labels[i * points_per_circle : (i + 1) * points_per_circle] = i

    # just forward fill the end of the array
    remaining_points = N % n_circles
    for i in range(remaining_points):
        curr_idx = n_circles * points_per_circle + i
        points[curr_idx, 0] = points[curr_idx - 1, 0]
        points[curr_idx, 1] = points[curr_idx - 1, 1]
        labels[curr_idx] = n_circles - 1

    return [x for x in points], labels


def show(X, Y):
    plt.scatter(X[Y][:, 0], X[Y][:, 1], label="1")
    plt.scatter(X[np.invert(Y)][:, 0], X[np.invert(Y)][:, 1], label="0")
    plt.legend()
    plt.show()


def save(data, file_path):
    """ """
    print(f"Saving {file_path} ...")
    if not os.path.isdir(os.path.dirname(file_path)):
        raise FileNotFoundError(f"{os.path.dirname(file_path)} does not exist.")

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print("Saved.")


if __name__ == "__main__":
    import configs.dumb as config

    X, Y = gen_data(300)
    save(X, os.path.join(config.data_directory, config.x))
    save(Y.astype(int), os.path.join(config.data_directory, config.y))
    show(X, Y)
