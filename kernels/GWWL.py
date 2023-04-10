from typing import List
from collections import Counter
import warnings
from kernels.kernel import Kernel
import networkx as nx
import numpy as np
from ot import emd2
from treeEditDistance import treeEditDistance

Graph = nx.classes.graph.Graph


def chain_representation(G, n, depth, cache=None, prev=None):
    if cache is None:
        cache = {}
    alone = True

    if (n, depth) in cache:
        return cache[(n, depth)]

    label = G.nodes[n]["labels"][0]
    label = label if prev is None else hash((label, G.edges[n, prev]["labels"][0]))
    node_str = str(label % 1000170000)
    neighbors_result = []

    if depth > 0:
        for k in G.neighbors(n):
            alone = False
            neighbor_result = chain_representation(G, k, depth - 1, cache, prev=n)
            neighbors_result.append(neighbor_result)
        if alone:
            neighbor_result = chain_representation(G, n, depth - 1, cache)
            neighbors_result.append(neighbor_result)

    result = (
        f"{node_str}({','.join(neighbors_result)})"
        if len(neighbors_result) > 0
        else f"{node_str}"
    )
    cache[(n, depth)] = result
    return result


class GeneralizedWassersteinWeisfeilerLehmanKernel(Kernel):
    def __init__(self, depth: int, lambd: float = 1.0, *args, **kargs) -> None:
        assert (
            isinstance(depth, int) and depth >= 0
        ), "depth should be a non-negative integer"
        self.depth = depth
        self.l = lambd
        super().__init__(*args, **kargs)

    def phi(self, x: Graph, *args, **kargs):
        return [chain_representation(x, n, depth=self.depth) for n in x.nodes()]

    def inner(self, X1, X2):
        D = treeEditDistance(X1, X2)
        wasserstein = emd2([], [], D)
        # return wasserstein
        return np.exp(-self.l * wasserstein)


def main():
    import timeit
    from preprocessing.load import load_file

    G1 = load_file("data/training_data.pkl")[0]
    G2 = load_file("data/training_data.pkl")[1]

    kernel = GeneralizedWassersteinWeisfeilerLehmanKernel(depth=3)

    K1g, K2g = kernel.phi(G1), kernel.phi(G2)
    print("Start")
    print(kernel.inner(K1g, K2g))
    print(kernel.inner(K1g, K1g))
    print(kernel.inner(K2g, K2g))

    timer = timeit.Timer(lambda: kernel.inner(K1g, K2g))
    print(f"Time for gwwl : {timer.timeit(1000)}")


if __name__ == "__main__":
    main()
