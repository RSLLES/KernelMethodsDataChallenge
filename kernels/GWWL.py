from typing import List
from kernels.kernel import Kernel
import networkx as nx
import numpy as np
from ot import emd2
import ctypes

Graph = nx.classes.graph.Graph


def chain_representation(G, n, depth, cache=None, prev=None):
    if cache is None:
        cache = {}
    alone = True

    if (n, depth) in cache:
        return cache[(n, depth)]

    label = G.nodes[n]["labels"][0]
    label_hashed = (
        hash((label, -1))
        if prev is None
        else hash((label, G.edges[n, prev]["labels"][0]))
    )
    label_hashed = ctypes.c_uint32(label_hashed).value
    node_str = str(label_hashed)
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


import wgwl


class GeneralizedWassersteinWeisfeilerLehmanKernel(Kernel):
    def __init__(self, depth: int, lambd: float = 1.0, *args, **kargs) -> None:
        assert (
            isinstance(depth, int) and depth >= 0
        ), "depth should be a non-negative integer"
        self.depth = depth
        self.l = lambd
        super().__init__(*args, **kargs)

    def _solo_phi(self, x):
        print("Custom solo phi")
        # z = [chain_representation(x, n, depth=self.depth) for n in x.nodes()]

        def work_with_cache(x):
            if self.use_cache:
                h = hash(x)
                if h not in self.cache:
                    self.cache[h] = self.phi(x)
                    self.nb_heavy_call += 1
                return self.cache[h]
            self.nb_heavy_call += 1
            return self.phi(x)

        z = [
            work_with_cache(x)
            for x in self._tqdm(x, desc="[Fit_Phi] Computing Embedding")
        ]

        K = wgwl.wgwlVec(z)
        np.save("wgwl.npy", K)
        return np.exp(-self.l * K)

    def phi(self, x: Graph, *args, **kargs):
        return [chain_representation(x, n, depth=self.depth) for n in x.nodes()]

    def inner(self, X1, X2):
        pass


import os


class GeneralizedWassersteinWeisfeilerLehmanKernelSum(Kernel):
    def __init__(self, root: str, lambd: float = 1.0, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        self.root = root
        self.l = lambd

    def _multi_inner_solo(self, H):
        try:
            r, c = H
            K = np.zeros((len(r),))
            for k, (i, j) in enumerate(zip(r, c)):
                K[k] = self.inner(i, j)  # And not self._z[i], self._z[j]
                if k % self.granularite == 0:
                    self.q.put(True)
            return K
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            return None

    def phi(self, x, *args, **kargs):
        return x

    def inner(self, X1, X2):
        X1, X2 = min(X1, X2), max(X1, X2)
        path = os.path.join(self.root, str(X1), str(X2) + ".npy")
        D = np.load(path)
        return np.exp(-self.l * D.sum())


class GeneralizedWassersteinWeisfeilerLehmanKernelImport(Kernel):
    def __init__(self, root: str, lambd: float, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        self.root = root
        self.l = lambd

    def _multi_inner_solo(self, H):
        try:
            r, c = H
            K = np.zeros((len(r),))
            for k, (i, j) in enumerate(zip(r, c)):
                K[k] = self.inner(i, j)  # And not self._z[i], self._z[j]
                if k % self.granularite == 0:
                    self.q.put(True)
            return K
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            return None

    def phi(self, x, *args, **kargs):
        return None

    def inner(self, X1, X2):
        X1, X2 = min(X1, X2), max(X1, X2)
        path = os.path.join(self.root, str(X1), str(X2) + ".npy")
        D = np.load(path)
        wasserstein = emd2([], [], D)
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
