import numpy as np
from tqdm import tqdm


def split_list(L: list, k: int):
    """
    This function takes in a list, L, and an integer, k, and returns a list of sublists. The sublists are created by splitting the original list into k equal parts.
    """
    div = len(L) / k
    res = []
    for i in range(k):
        # create a sub list from i * div to (i + 1) * div
        sub_list = L[int(i * div) : int((i + 1) * div)]
        res.append(sub_list)
    return res


def all_except(L: list[list]):
    """
    This function takes in a list of lists (L) and returns a list containing the sum of all elements in L except for the elements at each index.

    For example, if L is [[1,2], [3,4], [5,6]], the function will return [[3,4,5,6], [1,2,5,6], [1,2,3,4]].
    """
    rest = []
    for i in range(len(L)):
        rest.append(sum(L[:i] + L[i + 1 :], []))
    return rest


class Dataset:
    def __init__(self, X, Y, kernel=None, k_folds=2) -> None:
        assert len(X) == len(Y), "X and Y must have the same length."
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

        idxs_test = self.idxs_test[fold_idx]
        K_test = self.K[np.ix_(idxs_test, idxs_test)]
        y_test = self.y[idxs_test]

        idxs_train = self.idxs_train[fold_idx]
        K_train = self.K[np.ix_(idxs_train, idxs_train)]
        y_train = self.y[idxs_train]

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


### Unit Test ###
if __name__ == "__main__":
    n = 10
    k_folds = 3

    x = np.arange(10)
    y = np.arange(10) % 2
    print(f"x = {x}")
    print(f"y = {y}")

    kernel = lambda x, y: x * y
    print("Kernel is linear")

    print(f"k_folds = {k_folds}")

    d = Dataset(X=x, Y=y, kernel=kernel, k_folds=k_folds)
    d.compute_gram_matrix()
    print("Full gram Matrix : ")
    print(d.K)
    for fold, (K_train, y_train, K_test, y_test) in enumerate(d):
        print(f"##### Fold {fold+1} #####")
        print("K_train :")
        print(K_train)
        print("K_test :")
        print(K_test)
