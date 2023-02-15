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
