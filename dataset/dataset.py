import numpy as np
from tqdm import tqdm
from typing import List


def split_list(L: list, k: int):
    """
    This function takes in a list, L, and an integer, k, and returns a list of sublists.
    The sublists are created by splitting the original list into k equal parts.
    It returns a list containing only the original list if k=1.
    """
    assert k >= 1, "K must be >= 1"
    if k == 1:
        return [L]
    div = len(L) / k
    res = []
    for i in range(k):
        # create a sub list from i * div to (i + 1) * div
        sub_list = L[int(i * div) : int((i + 1) * div)]
        res.append(sub_list)
    return res


def all_except(L: List[list]):
    """
    This function takes in a list of lists (L) and returns a list containing the sum of all elements in L except for the elements at each index.
    For example, if L is [[1,2], [3,4], [5,6]], the function will return [[3,4,5,6], [1,2,5,6], [1,2,3,4]].
    Note that if the given list contains only 1 element, this function returns [].
    """
    if len(L) == 1:
        return []
    rest = []
    for i in range(len(L)):
        rest.append(sum(L[:i] + L[i + 1 :], []))
    return rest


class Dataset:
    """
    This function defines a dataset that can be iterated over.
    It returns two matrices, K_train and K_test, and two label vectors, y_train and y_test,
    for each fold in the cross-validation process.
    Note that it can be initialized with Y=None to support unsupervised learning,
    and with k_folds=1 to return only training data.
    (in this case, K_test and y_test will be set to None).
    """

    def __init__(self, X, Y=None, kernel=None, k_folds=1) -> None:
        assert Y is None or len(X) == len(Y), "X and Y must have the same length."
        self.n = len(X)
        self.x = X
        self.y = Y
        self.kernel = kernel
        self.k_folds = k_folds

        # Cross validation
        idxs = list(range(len(X)))
        self.idxs_test = split_list(idxs, k=self.k_folds)
        self.idxs_train = all_except(self.idxs_test)

    def compute_gram_matrix(self, kernel=None):
        if kernel is not None and self.kernel is not None and self.kernel != kernel:
            print(
                f"Warning : provide kernel '{kernel}' as argument will replace kernel '{self.kernel}' specified during initialization."
            )
            self.kernel = kernel

        self.K = np.zeros((self.n, self.n))

        with tqdm(
            list(range(self.n * (self.n + 1) // 2)), desc="Computing Gram Matrix"
        ) as pbar:
            for i in range(self.n):
                for j in range(i + 1):
                    self.K[i, j] = self.kernel(self.x[i], self.x[j])
                    self.K[j, i] = self.K[i, j]
                    pbar.update(1)

    def check_gram(self):
        if not hasattr(self, "K"):
            self.compute_gram_matrix()

    def __len__(self):
        return self.k_folds

    def __getitem__(self, fold_idx):
        """
        Returns for the given fold
        the Gram matrix for training K_train,
        the ground-truth labels for training y_train,
        the Gram matrix for testing K_train and
        the ground-truth labels for testing y_train.
        """
        self.check_gram()
        if not (0 <= fold_idx and fold_idx < self.__len__()):
            raise IndexError(f"Index {fold_idx} is incorrect.")

        # Define idxs
        if self.k_folds == 1:  # Test become train in this scenario
            idxs_train = self.idxs_test[fold_idx]
        else:
            idxs_test = self.idxs_test[fold_idx]
            idxs_train = self.idxs_train[fold_idx]

        # Around train
        K_train = self.K[np.ix_(idxs_train, idxs_train)]
        y_train = self.y[idxs_train] if self.y is not None else None

        if self.k_folds == 1:
            return K_train, y_train, None, None

        # Around test
        K_test = self.K[np.ix_(idxs_test, idxs_train)]
        y_test = self.y[idxs_test] if self.y is not None else None

        return K_train, y_train, K_test, y_test

    def __iter__(self):
        self.check_gram()
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.__len__():
            raise StopIteration
        a = self.__getitem__(self.idx)
        self.idx += 1
        return a
