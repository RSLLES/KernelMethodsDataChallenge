import unittest
import networkx as nx

from kernels.WL import WeisfeilerLehmanKernel


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


class TestWeisfeilerLehmanKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = [G1(), G2()]
        self.kernel = WeisfeilerLehmanKernel(depth=1)

    def test_kernel_computation_single_graph(self):
        g1, g2 = self.graphs[0], self.graphs[1]
        computed_kernel_value = self.kernel(g1, g2, use_cache=False)
        self.assertEqual(computed_kernel_value, 11)

    def test_kernel_computation_list_of_graphs(self):
        expected_kernel_values = [[16, 11], [11, 14]]

        computed_kernel_values_without_cache = self.kernel(
            self.graphs, self.graphs, use_cache=False
        )
        computed_kernel_values = self.kernel(self.graphs, self.graphs)
        self.assertEqual(computed_kernel_values_without_cache, expected_kernel_values)
        self.assertEqual(computed_kernel_values, expected_kernel_values)
