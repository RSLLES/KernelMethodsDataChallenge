from typing import Union, List
from collections import Counter
from functools import wraps
from kernels.kernel import Kernel
import networkx as nx

Graph = nx.classes.graph.Graph

#########################
### Weisfeiler Lehman ###
#########################


def get_neighbours_based_label(G: Graph, n: int) -> int:
    """
    This function returns a new label based on the neighbors of node n in graph G.

    Args:
    - G (Graph): the input graph
    - n (int): the node index

    Returns:
    - int: a new label for the node n of G
    """
    labels = G.nodes[n]["labels"] + sorted(
        [G.nodes[k]["labels"][0] for k in G.neighbors(n)]
    )
    return hash(tuple(labels))


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


def label_counts_over_WL_iteration(G: Graph, depth: int) -> int:
    assert isinstance(depth, int), "depth should be a non-negative integer"
    assert depth >= 0, "depth should be a non-negative integer"

    counts = Counter([G.nodes[n]["labels"][0] for n in G.nodes()])

    if depth == 0:
        return [counts]

    assign_neighbours_based_labeling(G)
    return [counts] + label_counts_over_WL_iteration(G, depth - 1)


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

    all_counts1 = label_counts_over_WL_iteration(G1, depth)
    all_counts2 = label_counts_over_WL_iteration(G2, depth)

    s = 0
    for counts1, counts2 in zip(all_counts1, all_counts2):
        for label in counts1:
            if label in counts2:
                s += counts1[label] * counts2[label]
    return s


class WeisfeilerLehmanKernel(Kernel):
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
        super().__init__()

    def kernel(
        self,
        x1: Union[Graph, List[Graph]],
        x2: Union[Graph, List[Graph]],
    ) -> Union[int, List[int]]:
        """
        Compute the Weisfeiler Lehman Kernel at a given depth for a single or several pairs of graphs.

        Args:
            x1 (Union[Graph, List[Graph]]): A single graph or a list of graphs.
            x2 (Union[Graph, List[Graph]]): A single graph or a list of graphs.

        Returns:
             The Weisfeiler Lehman Kernel of two graphs if given two graphs.
             A list of results or a list of list of results if either `x1` or `x2` is a list.
        """

        G1, G2 = x1.copy(), x2.copy()
        return WL(G1=G1, G2=G2, depth=self.depth)
