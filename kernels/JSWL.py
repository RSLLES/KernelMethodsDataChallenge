from typing import List
from collections import Counter
from kernels.kernel import Kernel
import networkx as nx
import numpy as np
import sortednp as snp

Graph = nx.classes.graph.Graph


def MidProb(idx1, p1, idx2, p2):
    if len(idx1) == 0:
        return p2 * 0.5
    if len(idx2) == 0:
        return p1 * 0.5

    idx = np.unique(snp.merge(idx1, idx2))
    p = np.zeros((idx.shape[0],), dtype=float)

    np.add.at(p, np.searchsorted(idx, idx1), p1)
    np.add.at(p, np.searchsorted(idx, idx2), p2)

    return p * 0.5


def SplitProbaIndex(X):
    if len(X) == 0:
        return X, X
    x = X[:, 1].astype(float)
    return X[:, 0], x / x.sum()


def Entropy(p):
    return -np.dot(p, np.log(p))


def compute_neighbors_based_label(
    G: Graph, labels: dict, n: int, edges_labels: dict = None
) -> int:
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
    assert depth >= 0, "depth should be a non-negative integer"

    to_count = [
        [(labels[n], edges_labels[k, n]["labels"][0]) for k in G.neighbors(n)]
        for n in G
    ]
    to_count = sum(to_count, [])
    counts = Counter(to_count)
    counts = [(hash(key), value) for key, value in counts.items()]
    counts.sort(key=lambda x: x[0])
    counts = np.array(counts, dtype=np.int64)
    idx, p = SplitProbaIndex(counts)
    h = Entropy(p)
    if depth == 0:
        return [(idx, p, h)]

    new_labels = {
        n: compute_neighbors_based_label(
            G=G, labels=labels, edges_labels=edges_labels, n=n
        )
        for n in G.nodes()
    }

    for n in G.nodes():
        labels[n] = new_labels[n]

    return [(idx, p, h)] + WL_iterations(
        G=G, labels=labels, edges_labels=edges_labels, depth=depth - 1
    )


class JensenShannonWeisfeilerLehmanKernel(Kernel):
    def __init__(
        self,
        depth: int,
        enable_edges_labels: bool = False,
        lambd: float = 1.0,
        *args,
        **kargs,
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
        self.lambd = lambd
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
        for (idx1, p1, h1), (idx2, p2, h2) in zip(counts1, counts2):
            pM = MidProb(idx1, p1, idx2, p2)
            h = Entropy(pM)
            s += h - 0.5 * (h1 + h2)
        return np.exp(-self.lambd * s / self.depth)


def main():
    import timeit
    from preprocessing.load import load_file

    G1 = load_file("data/training_data.pkl")[0]
    G2 = load_file("data/training_data.pkl")[1]

    kernel = JensenShannonWeisfeilerLehmanKernel(depth=5, enable_edges_labels=True)
    K1, K2 = kernel.phi(G1), kernel.phi(G2)
    print(kernel.inner(K1, K1))
    print(kernel.inner(K2, K2))
    print(kernel.inner(K1, K2))

    timer = timeit.Timer(lambda: kernel.inner(K1, K2))
    N = 20000
    print(f"Duration for {N} runs {timer.timeit(N)}")

    pass


if __name__ == "__main__":
    main()
