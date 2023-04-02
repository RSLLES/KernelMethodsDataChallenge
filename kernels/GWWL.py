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


def pad(labels, size=1 + 6, blank_label=-1):  # one_label + Max neighbors
    return labels + (blank_label,) * (size - len(labels))


def compute_neighbors_based_label(G: Graph, labels: dict, n: int) -> int:
    label = hash(labels[n])
    neighbors = tuple(
        hash((labels[k], G.edges[k, n]["labels"][0]))
        if (k, n) in G.edges
        else hash(labels[k])
        for k in G.neighbors(n)
    )

    return pad((label,) + (neighbors))


def WL_iterations(G: Graph, labels: dict, depth: int) -> int:
    assert depth >= 1

    new_labels = {
        n: compute_neighbors_based_label(G=G, labels=labels, n=n) for n in G.nodes()
    }

    if depth == 1:
        return [new_labels]
    return [new_labels] + WL_iterations(G=G, labels=new_labels, depth=depth - 1)


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

        batch_labels = []
        for labels in res:
            batch_labels.append(np.array([labels[n] for n in x.nodes]))
        return np.stack(batch_labels)

    def inner(self, X1, X2):
        len1, len2 = len(X1[0]), len(X2[0])
        shape = (self.depth, len1, len2, self.max_labels)
        D1 = np.broadcast_to(X1[:, :, None, :], shape)
        D2 = np.broadcast_to(X2[:, None, :, :], shape)
        D = not_equal_or_nan(D1, D2)

        D_labels = D[..., 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            D_neighbors = np.nanmean(D[..., 1:], axis=-1)
        D_neighbors = np.nan_to_num(D_neighbors, 0.0)

        D = self.w * D_labels + (1.0 - self.w) * D_neighbors
        D = D.mean(axis=0)
        wasserstein = ot.emd2([], [], D)
        return round(np.exp(-self.l * wasserstein), 7)


def not_equal_or_nan(a, b, blank_token=-1):
    equal_mask = a == b
    none_mask = (a == blank_token) & (b == blank_token)
    result = np.where(equal_mask, False, True)
    result = np.where(none_mask, np.nan, result)
    return result


# def main():
#     import kernels.WWL
#     from preprocessing.load import load_file

#     G1 = load_file("data/training_data.pkl")[0]
#     G2 = load_file("data/training_data.pkl")[1]

#     KernelG = GeneralizedWassersteinWeisfeilerLehmanKernel
#     Kernel = kernels.WWL.WassersteinWeisfeilerLehmanKernel

#     kernelg, kernel = KernelG(1, weight=1.0), Kernel(0)
#     K1g, K2g = kernelg.phi(G1), kernelg.phi(G2)
#     K1, K2 = kernel.phi(G1), kernel.phi(G2)
#     i = kernel.inner(K1, K2)
#     ig = kernelg.inner(K1g, K2g)
#     print(i, ig)


# if __name__ == "__main__":
#     main()
