from typing import List
from collections import Counter
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
    neighbors_labels = tuple(hash(labels[k]) for k in G.neighbors(n))
    neighbors_edges = tuple(hash(G.edges[n, k]["labels"][0]) for k in G.neighbors(n))
    return (label, neighbors_labels, neighbors_edges)


def WL_iterations(G: Graph, labels: dict, depth: int) -> int:
    assert depth >= 1

    new_labels = {
        n: compute_neighbors_based_label(G=G, labels=labels, n=n) for n in G.nodes()
    }

    if depth == 1:
        return [new_labels]
    return [new_labels] + WL_iterations(G=G, labels=new_labels, depth=depth - 1)


class GeneralizedWassersteinWeisfeilerLehmanKernel(Kernel):
    def __init__(self, depth: int, lambd: float = 1.0, *args, **kargs) -> None:
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

        self.depth = depth
        self.l = lambd
        self.w_labels, self.w_neighbors, self.w_edges = 1 / 2, 1 / 4, 1 / 4
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
        return WL_iterations(G=x, labels=labels, depth=self.depth)

    def inner(self, X1, X2):
        len1, len2 = len(X1[0]), len(X2[0])
        D = np.zeros((self.depth, len1, len2))

        for batch in range(self.depth):
            dic1, dic2 = X1[batch], X2[batch]
            for i in range(len1):
                for j in range(len2):
                    label1, neighbors1, edges1 = dic1[i]
                    label2, neighbors2, edges2 = dic2[j]

                    D[batch, i, j] += self.w_labels if label1 != label2 else 0.0
                    D[batch, i, j] += self.w_neighbors * unshared_elements(
                        neighbors1, neighbors2
                    )
                    D[batch, i, j] += self.w_edges * unshared_elements(edges1, edges2)

        D = D.mean(axis=0)
        wasserstein = ot.emd2([], [], D)
        # return wasserstein
        return np.exp(-self.l * wasserstein)


def unshared_elements(T1, T2):
    L1, L2 = list(T1), list(T2)
    base = len(L1) + len(L2)
    if base == 0:
        return 0.0
    i = 0
    while i < len(L1):
        if L1[i] in L2:
            L2.remove(L1[i])
            del L1[i]
        else:
            i += 1
    return (len(L1) + len(L2)) / base


def main():
    from preprocessing.load import load_file

    G1 = load_file("data/training_data.pkl")[0]
    G2 = load_file("data/training_data.pkl")[1]
    kernel = GeneralizedWassersteinWeisfeilerLehmanKernel(1)
    K1, K2 = kernel.phi(G1), kernel.phi(G2)
    D = kernel.inner(K1, K2)


if __name__ == "__main__":
    main()
