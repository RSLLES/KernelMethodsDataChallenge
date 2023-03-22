from typing import Union, List
from collections import Counter
from functools import wraps
import networkx as nx

Graph = nx.classes.graph.Graph

#########################
### Weisfeiler Lehman ###
#########################


def get_neighbours_based_label(G: Graph, n: int) -> tuple:
    """
    This function returns a tuple of labels based on the neighbors of node n in graph G.

    Args:
    - G (Graph): the input graph
    - n (int): the node index

    Returns:
    - tuple: a tuple of labels based on the neighbors of node n
    """
    labels = [G.nodes[k]["labels"][0] for k in G.neighbors(n)]
    labels = sorted(labels)
    return tuple(G.nodes[n]["labels"] + labels)


def assign_neighbours_based_labeling(G: Graph) -> None:
    """
    This function assigns labels to nodes based on neighboring labels

    Args:
    - G (Graph): the input graph

    Returns:
    - None
    """
    labels = {n: get_neighbours_based_label(G, n) for n in G.nodes()}
    for n in G.nodes():
        G.nodes[n]["labels"] = [labels[n]]


def label_compression(G: Graph, lookup: dict = {}) -> None:
    """
    This function compresses or renames the labels of a graph to integer values.

    Args:
    - G (Graph): the input graph
    - loops (dict): dictionary containing mappings from labels to integers. It defaults to an empty dictionary.

    Returns:
    - None
    """
    for n in G.nodes():
        if G.nodes[n]["labels"][0] not in lookup:
            lookup[G.nodes[n]["labels"][0]] = len(lookup)
        G.nodes[n]["labels"] = [lookup[G.nodes[n]["labels"][0]]]


def inner(G1: Graph, G2: Graph) -> int:
    """
    This function calculates the inner product of the two graphs using their labels.

    Args:
    - G1 (Graph): first input graph
    - G2 (Graph): second input graph

    Returns:
    - int: the inner product of the two graphs using their labels
    """
    counts1 = Counter([G1.nodes[n]["labels"][0] for n in G1.nodes()])
    counts2 = Counter([G2.nodes[n]["labels"][0] for n in G2.nodes()])
    s = 0
    for label in counts1:
        if label in counts2:
            s += counts1[label] * counts2[label]
    return s


# @cache_WL
def WL(G1: Graph, G2: Graph, depth: int) -> int:
    """
    This function implements the Weisfeiler-Lehman Graph Kernel at a given depth.
    See the following paper : https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

    Args:
    - G1 (Graph): first input graph
    - G2 (Graph): second input graph
    - depth (int): the maximum depth of the Weisfeiler-Lehman algorithm. It should be a non-negative integer.

    Returns:
    - int: whether the two graphs are isomorphic (if the returned value equals zero), or how many times they differ
            in the number of nodes with a specific label in their neighborhood (if the returned value is greater than zero).
    """
    assert isinstance(depth, int), "depth should be a non-negative integer"
    assert depth >= 0, "depth should be a non-negative integer"

    s = inner(G1, G2)

    if depth == 0:
        return s

    assign_neighbours_based_labeling(G1)
    assign_neighbours_based_labeling(G2)

    lookup = {}
    label_compression(G1, lookup=lookup)
    label_compression(G2, lookup=lookup)

    return s + WL(G1=G1, G2=G2, depth=depth - 1)


class WeisfeilerLehmanKernel:
    def __init__(self, depth: int) -> None:
        """
        Weisfeiler Lehman Kernel

        Parameters:
            depth (int): depth of kernel
        """
        assert (
            isinstance(depth, int) and depth >= 0
        ), "depth should be a non-negative integer"
        self.depth = depth
        self.cache = {}

    def __call__(
        self,
        x1: Union[Graph, List[Graph]],
        x2: Union[Graph, List[Graph]],
        use_cache: bool = True,
    ) -> Union[int, List[int]]:
        """
        Compute the Weisfeiler Lehman Kernel at a given depth for a single or several pairs of graphs.

        Args:
            x1 (Union[Graph, List[Graph]]): A single graph or a list of graphs.
            x2 (Union[Graph, List[Graph]]): A single graph or a list of graphs.
            use_cache (bool): Whether to use caching. If True, the kernel values will be stored in a cache dictionary for
                faster computation next time `__call__` is called with the same input. Defaults to True.

        Returns:
             The Weisfeiler Lehman Kernel of two graphs if given two graphs.
             A list of results or a list of list of results if either `x1` or `x2` is a list.
        """
        if isinstance(x1, list):
            results = []
            for item in x1:
                results.append(self.__call__(x1=item, x2=x2, use_cache=use_cache))
            return results
        if isinstance(x2, list):
            results = []
            for item in x2:
                results.append(self.__call__(x1=x1, x2=item, use_cache=use_cache))
            return results

        if use_cache:
            h1, h2 = hash(x1), hash(x2)
            h = (h2, h1) if h1 > h2 else (h1, h2)
            if h in self.cache:
                return self.cache[h]

        G1, G2 = x1.copy(), x2.copy()
        kernel_value = WL(G1=G1, G2=G2, depth=self.depth)

        if use_cache:
            self.cache[h] = kernel_value

        return kernel_value
