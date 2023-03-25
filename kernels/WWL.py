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


def compute_neighbors_based_label(
    G: Graph, labels: dict, n: int, edges_labels: dict = None
) -> int:
    """
    Computes a new label based on labels of a node and its direct neighbors.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The input graph to which this node belongs.
    labels : dict
        A dictionary where key represents the node number and value is its corresponding node label
    n : int
        The node whose feature has to be computed.

    Returns
    -------
    int :
        A new label based on labels of a node and its direct neighbors.
    """
    if edges_labels is None:
        label = [labels[n]] + sorted([labels[k] for k in G.neighbors(n)])
    else:
        label = [labels[n]] + sorted(
            [
                hash((labels[k], edges_labels[k, n]["labels"][0]))
                if (k, n) in edges_labels
                else labels[k]
                for k in G.neighbors(n)
            ]
        )
    return hash(tuple(label))


def WL_iterations(G: Graph, labels: dict, depth: int, edges_labels: dict = None) -> int:
    """
    Computes the Weisfeiler Lehman reduction of a given level.

    Parameters
    ----------
    G : networkx.classes.graph.Graph
        The input graph which needs to be reduced
    labels : dict
        A dictionary where key represents the node number and value is its corresponding node label
    depth : int
        An integer representing the depth of iterations to be performed by Weisfeiler Lehman.

    Returns
    -------
    list(collections.Counter) :
        A list of counter objects where each counter representa the frequency of each labeled
        substructure observed at `i-th` iteration.
    """
    assert depth >= 0, "depth should be a non-negative integer"

    c = list(labels.values())
    if depth == 0:
        return [c]

    new_labels = {
        n: compute_neighbors_based_label(
            G=G, labels=labels, edges_labels=edges_labels, n=n
        )
        for n in G.nodes()
    }

    for n in G.nodes():
        labels[n] = new_labels[n]

    return [c] + WL_iterations(G=G, labels=labels, depth=depth - 1)


class WassersteinWeisfeilerLehmanKernel(Kernel):
    def __init__(
        self,
        depth: int,
        lambd: float = 1.0,
        enable_edges_labels: bool = False,
        *args,
        **kargs
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
            isinstance(depth, int) and depth >= 0
        ), "depth should be a non-negative integer"

        self.depth = depth
        self.enable_edges_labels = enable_edges_labels
        self.l = lambd
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
        if self.enable_edges_labels:
            return np.array(
                WL_iterations(
                    G=x, labels=labels, depth=self.depth, edges_labels=x.edges
                )
            )
        return np.array(WL_iterations(G=x, labels=labels, depth=self.depth))

    def inner(self, X1, X2):
        D = np.zeros((self.depth + 1, X1.shape[1], X2.shape[1]))
        for batch in range(self.depth + 1):
            D[batch] = np.not_equal.outer(X1[batch], X2[batch])
        D = D.mean(axis=0)
        wasserstein = ot.emd2([], [], D)
        return np.exp(-self.l * wasserstein)
