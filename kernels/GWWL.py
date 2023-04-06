from typing import List
from collections import Counter
import warnings
from kernels.kernel import Kernel
import networkx as nx
import numpy as np
import ot

Graph = nx.classes.graph.Graph


#########################
### Weisfeiler Lehman ###
#########################


def compute_neighbors_based_label(G: Graph, labels: dict, n: int) -> int:
    label = hash(labels[n])
    neighbors = tuple(
        hash((labels[k], G.edges[k, n]["labels"][0]))
        if (k, n) in G.edges
        else hash(labels[k])
        for k in G.neighbors(n)
    )

    return label, neighbors


def WL_iterations(G: Graph, labels: dict, depth: int) -> int:
    assert depth >= 1

    new_labels = {
        n: compute_neighbors_based_label(G=G, labels=labels, n=n) for n in G.nodes()
    }

    R = [(new_labels[n][0], Counter(new_labels[n][1])) for n in G.nodes()]
    if depth == 1:
        return [R]
    return [R] + WL_iterations(G=G, labels=new_labels, depth=depth - 1)


class GeneralizedWassersteinWeisfeilerLehmanKernel(Kernel):
    def __init__(
        self, depth: int, lambd: float = 1.0, weight: float = 0.5, *args, **kargs
    ) -> None:
        """
        Implementation of the Wasserstein Weisfeiler-Lehman Graph Kernel.
        See the following paper : https://arxiv.org/pdf/1906.01277.pdf

        Parameters
        ----------
        depth : int
            An integer representing the depth of iterations to be performed by Weisfeiler Lehman
            algorithm. This should always be a non-negative integer.
        """
        assert (
            isinstance(depth, int) and depth >= 1
        ), "depth should be a non-negative integer"
        assert isinstance(weight, float) and weight <= 1.0 and weight >= 0.0
        self.depth = depth
        self.l = lambd
        self.w = weight
        self.max_labels = 7  # See pad function
        super().__init__(*args, **kargs)

    def phi(self, x: Graph, *args, **kargs):
        """
        Generates the feature vector representation of given directed graph using Weisfeiler-Lehman graph
        kernel.

        Parameters
        ----------
        x : networkx.classes.graph.Graph
            Input undirected graph to be represented in feature space

        Returns
        -------
        list(collections.Counter) :
            A list of counter objects where each counter representa the frequency of each labeled
            substructure observed at `i-th` iteration.
        """
        labels = {n: x.nodes[n]["labels"][0] for n in x.nodes}
        res = WL_iterations(G=x, labels=labels, depth=self.depth)
        labels = np.array([[label for label, neighbors in batch] for batch in res])
        neighbors = [[neighbors for label, neighbors in batch] for batch in res]
        return labels, neighbors

    def inner(self, X1, X2):
        ## Labels
        L1, L2 = X1[0], X2[0]
        D_labels = np.zeros((self.depth, L1.shape[1], L2.shape[1]))
        for batch in range(self.depth):
            D_labels[batch] = np.not_equal.outer(L1[batch], L2[batch])

        ## neighbors
        N1, N2 = X1[1], X2[1]
        D_neighbors = np.zeros((self.depth, L1.shape[1], L2.shape[1]))
        for batch in range(self.depth):
            for i in range(len(N1[batch])):
                for j in range(len(N2[batch])):
                    c1, c2 = N1[batch][i], N2[batch][j]
                    div = min(c1.total(), c2.total())
                    D_neighbors[batch, i, j] = (
                        1.0 - (c1 & c2).total() / div if div > 0 else 0.0
                    )

        # Merge
        D = D_labels * (1.0 + self.w * D_neighbors) / (1 + self.w)
        # D = D_labels
        D = D.mean(axis=0)
        wasserstein = ot.emd2([], [], D)
        return round(np.exp(-self.l * wasserstein), 7)


def count_common(arr, blank=-1, a=7):  # max_size
    arr1, arr2 = arr[:a], arr[a:]
    arr1, arr2 = arr1[arr1 != blank], arr2[arr2 != blank]
    a1 = sum(np.isin(arr1, arr2, invert=True)) / len(arr1) if len(arr1) > 0 else 0.0
    a2 = sum(np.isin(arr2, arr1, invert=True)) / len(arr2) if len(arr2) > 0 else 0.0
    return 0.5 * (a1 + a2)


def not_equal_or_nan(a, b, blank_token=-1):
    equal_mask = a == b
    none_mask = (a == blank_token) & (b == blank_token)
    result = np.where(equal_mask, False, True)
    result = np.where(none_mask, np.nan, result)
    return result


def main():
    import timeit
    import kernels.WWL
    from preprocessing.load import load_file

    G1 = load_file("data/training_data.pkl")[0]
    G2 = load_file("data/training_data.pkl")[1]

    KernelG = GeneralizedWassersteinWeisfeilerLehmanKernel
    Kernel = kernels.WWL.WassersteinWeisfeilerLehmanKernel

    kernelg, kernel = KernelG(8, weight=1.0), Kernel(7)
    K1g, K2g = kernelg.phi(G1), kernelg.phi(G2)
    K1, K2 = kernel.phi(G1), kernel.phi(G2)
    i = kernel.inner(K1, K2)
    ig = kernelg.inner(K1g, K2g)
    print(i, ig)

    timer = timeit.Timer(lambda: kernelg.inner(K1g, K2g))
    print(f"Time for gwwl : {timer.timeit(1000)}")

    timer = timeit.Timer(lambda: kernel.inner(K1, K2))
    print(f"Time for wwl : {timer.timeit(1000)}")

    pass


if __name__ == "__main__":
    main()
