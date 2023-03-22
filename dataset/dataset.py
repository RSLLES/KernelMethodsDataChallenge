import random
from typing import List
import numpy as np


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
    It returns two data vectors, X_train and X_test, and two label vectors, y_train and y_test,
    for each fold in the cross-validation process.
    Note that it can be initialized with Y=None to support unsupervised learning,
    and with k_folds=1 to return only training data.
    (in this case, X_test and y_test will be set to None).
    """

    def __init__(self, X, y=None, k_folds=1, shuffle=False) -> None:
        assert isinstance(X, list)
        assert y is None or len(X) == len(y), "X and Y must have the same length."
        assert y is None or isinstance(y, np.ndarray)
        self.n = len(X)
        self.X = X
        self.y = y
        self.k_folds = k_folds

        # Cross validation
        idxs = list(range(len(X)))
        if shuffle:
            random.shuffle(idxs)
        self.idxs_test = split_list(idxs, k=self.k_folds)
        self.idxs_train = all_except(self.idxs_test)

    def __len__(self):
        return self.k_folds

    def __getitem__(self, fold_idx):
        if not (0 <= fold_idx and fold_idx < self.__len__()):
            raise IndexError(f"Index {fold_idx} is incorrect.")

        # Define idxs
        if self.k_folds == 1:  # Test become train in this scenario
            idxs_train = self.idxs_test[fold_idx]
        else:
            idxs_test = self.idxs_test[fold_idx]
            idxs_train = self.idxs_train[fold_idx]

        # Around train
        X_train = [self.X[i] for i in idxs_train]
        y_train = self.y[idxs_train] if self.y is not None else None

        if self.k_folds == 1:
            return X_train, y_train, None, None

        # Around test
        X_test = [self.X[i] for i in idxs_test]
        y_test = self.y[idxs_test] if self.y is not None else None

        return X_train, y_train, X_test, y_test

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.__len__():
            raise StopIteration
        a = self.__getitem__(self.idx)
        self.idx += 1
        return a
