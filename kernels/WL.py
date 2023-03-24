from typing import List
from collections import Counter
from kernels.kernel import Kernel
import networkx as nx
import numpy as np

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

    counts = Counter(labels.values())
    if depth == 0:
        return [counts]

    new_labels = {
        n: compute_neighbors_based_label(
            G=G, labels=labels, edges_labels=edges_labels, n=n
        )
        for n in G.nodes()
    }

    for n in G.nodes():
        labels[n] = new_labels[n]

    return [counts] + WL_iterations(G=G, labels=labels, depth=depth - 1)


class WeisfeilerLehmanKernel(Kernel):
    def __init__(
        self, depth: int, enable_edges_labels: bool = False, *args, **kargs
    ) -> None:
        """
        Implementation of the Weisfeiler-Lehman Graph Kernel at a given depth.
        See the following paper : https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

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
        super().__init__(*args, **kargs)

    def phi(self, x: Graph, *args, **kargs) -> List[Counter]:
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
            return WL_iterations(
                G=x, labels=labels, depth=self.depth, edges_labels=x.edges
            )
        return WL_iterations(G=x, labels=labels, depth=self.depth)

    def inner(self, counts1, counts2):
        s = 0
        for c1, c2 in zip(counts1, counts2):
            s += sum([c1[k] * c2[k] for k in c1 if k in c2])
        return s


class RBFWeisfeilerLehmanKernel(Kernel):
    def __init__(self, depth: int, sigma: float = 1.0, *args, **kargs) -> None:
        """
        Implementation of the Weisfeiler-Lehman Graph Kernel at a given depth.
        See the following paper : https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

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
        self.gamma = 1.0 / (2 * sigma**2)
        super().__init__(*args, **kargs)

    def phi(self, x: Graph, *args, **kargs) -> List[Counter]:
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

    def inner(self, counts1, counts2):
        v = []
        for c1, c2 in zip(counts1, counts2):
            v += [(c1[k], c2[k]) for k in c1 if k in c2]
        if len(v) == 0:
            return 1.0
        v = np.array(v).T
        d = v[1] - v[0]
        return np.exp(-self.gamma * np.dot(d, d))
