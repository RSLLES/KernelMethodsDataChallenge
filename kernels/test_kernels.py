import unittest
import networkx as nx
import numpy as np
import random as rd

from kernels.WL import WeisfeilerLehmanKernel


class TestWeisfeilerLehmanKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = [G1(), G2()]

    def test_kernel_computation_single_graph(self):
        kernel = WeisfeilerLehmanKernel(depth=1)
        g1, g2 = self.graphs[0], self.graphs[1]

        computed_kernel_value = kernel(g1, g2)
        self.assertEqual(computed_kernel_value, 11)

    def test_kernel_computation_list_of_graphs(self):
        kernel = WeisfeilerLehmanKernel(depth=1)
        kernel_values = kernel(self.graphs, self.graphs)
        np.testing.assert_almost_equal(kernel_values, np.array([[16, 11], [11, 14]]))

    def test_cache(self):
        graphs = [randomGraph() for _ in range(100)]
        kernel = WeisfeilerLehmanKernel(depth=1, use_cache=True)
        kernel_wto_cache = WeisfeilerLehmanKernel(depth=1, use_cache=False)

        results = kernel_wto_cache(graphs, graphs)
        results_cached = kernel(graphs, graphs)

        np.testing.assert_almost_equal(results, results_cached)
        print(
            f"{kernel.nb_heavy_call} heavy calls with cache VS {kernel_wto_cache.nb_heavy_call} without."
        )


def randomGraph(p_edges=0.3, nb_nodes=10, nb_labels=5):
    edges = []
    for i in range(nb_nodes):
        for j in range(i + 1, nb_nodes):
            edges.append((i, j))
    rd.shuffle(edges)
    edges = edges[int(len(edges) * p_edges) :]

    g = nx.Graph(edges)
    for n in g.nodes():
        g.nodes[n]["labels"] = [rd.randint(0, nb_labels)]

    return g


def G1():
    g = nx.Graph(
        [
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 4),
            (3, 4),
            (3, 5),
            (3, 6),
        ]
    )

    g.nodes[1]["labels"] = [5]
    g.nodes[2]["labels"] = [2]
    g.nodes[3]["labels"] = [4]
    g.nodes[4]["labels"] = [3]
    g.nodes[5]["labels"] = [1]
    g.nodes[6]["labels"] = [1]

    return g


def G2():
    g = nx.Graph(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 6),
        ]
    )

    g.nodes[1]["labels"] = [2]
    g.nodes[2]["labels"] = [5]
    g.nodes[3]["labels"] = [4]
    g.nodes[4]["labels"] = [3]
    g.nodes[5]["labels"] = [1]
    g.nodes[6]["labels"] = [2]

    return g
