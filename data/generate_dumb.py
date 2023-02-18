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
    return X, Y


def gen_linearly_separable_data(N,
                                mu_positive: np.ndarray = np.array([1., 1.]),
                                sigma_positive: float = 0.2,
                                mu_negative: np.ndarray = np.array([-1., 0.]),
                                sigma_negative: float = 0.2):
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

    y = np.concatenate((np.ones(points_per_class), -np.ones(points_per_class)))

    return np.concatenate((positive_points, negative_points)), y


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
