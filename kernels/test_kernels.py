import unittest
import networkx as nx
import numpy as np
import random as rd
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel
from sklearn.gaussian_process.kernels import RBF

from data.generate_dumb import gen_data

from kernels.kernels import GaussianKernel, LinearKernel, PolynomialKernel

from kernels.WL import WeisfeilerLehmanKernel

# import grakel


class TestBasicsKernels(unittest.TestCase):
    def setUp(self):
        self.X, _ = gen_data(20)

    def test_linear(self):
        kernel = LinearKernel()
        kernel_sk = linear_kernel

        results_ours = kernel(self.X, self.X)
        results_sk = kernel_sk(self.X, self.X)
        np.testing.assert_almost_equal(results_ours, results_sk)

    def test_gaussian(self):
        kernel = GaussianKernel(sigma=1.5)
        kernel_sk = RBF(length_scale=1.5)

        results_ours = kernel(self.X, self.X)
        results_sk = kernel_sk(self.X, self.X)
        np.testing.assert_almost_equal(results_ours, results_sk)

    def test_polynomial(self):
        kernel = PolynomialKernel(scale=2.0, offset=2.0, degree=3)
        kernel_sk = lambda x, y: polynomial_kernel(
            X=x, Y=y, gamma=0.5, coef0=2.0, degree=3
        )

        results_ours = kernel(self.X, self.X)
        results_sk = kernel_sk(self.X, self.X)
        np.testing.assert_almost_equal(results_ours, results_sk)


class TestKernel(unittest.TestCase):
    def setUp(self) -> None:
        self.nb_data = 20
        self.X = np.random.random(size=self.nb_data).tolist()
        self.graphs = [randomGraph() for _ in range(20)]

    def test_cache_with_phi_method(self):
        kernel = GaussianKernel(sigma=1.0, use_cache=True)
        kernel_wto_cache = GaussianKernel(sigma=1.0, use_cache=False)

        results = kernel_wto_cache(self.X, self.X)
        results_cached = kernel(self.X, self.X)

        np.testing.assert_almost_equal(results, results_cached)
        np.testing.assert_equal(
            kernel.nb_heavy_call, self.nb_data * (self.nb_data + 1) / 2
        )
        np.testing.assert_equal(kernel_wto_cache.nb_heavy_call, self.nb_data**2)

    def test_cache_with_phi_method(self):
        kernel = WeisfeilerLehmanKernel(depth=7, use_cache=True)
        kernel_wto_cache = WeisfeilerLehmanKernel(depth=7, use_cache=False)

        results = kernel_wto_cache(self.graphs, self.graphs)
        results_cached = kernel(self.graphs, self.graphs)

        np.testing.assert_almost_equal(results, results_cached)
        np.testing.assert_equal(kernel.nb_heavy_call, self.nb_data)
        np.testing.assert_equal(kernel_wto_cache.nb_heavy_call, 2 * self.nb_data**2)

    def test_multiprocess(self):
        kernel = WeisfeilerLehmanKernel(depth=3)
        K1 = kernel(self.graphs, self.graphs)
        K2 = kernel(self.graphs)
        np.testing.assert_allclose(K1, K2)


class TestWeisfeilerLehmanKernel(unittest.TestCase):
    def setUp(self) -> None:
        rd.seed(0)
        self.ex_graphs = [G1(), G2()]
        # self.graphs = [G1(), G2()]
        self.graphs = [randomGraph() for _ in range(5)]

    def test_kernel_computation_single_graph(self):
        kernel = WeisfeilerLehmanKernel(depth=1)
        g1, g2 = self.ex_graphs[0], self.ex_graphs[1]

        computed_kernel_value = kernel(g1, g2)
        self.assertEqual(computed_kernel_value, 11)

    def test_kernel_computation_list_of_graphs(self):
        kernel = WeisfeilerLehmanKernel(depth=1)
        kernel_values = kernel(self.ex_graphs, self.ex_graphs)
        np.testing.assert_almost_equal(kernel_values, np.array([[16, 11], [11, 14]]))

    # @staticmethod
    # def from_ours_to_grakels_graph(g):
    #     edges = list(g.edges)
    #     labels = {n: g.nodes[n]["labels"][0] for n in g.nodes()}
    #     return grakel.Graph(edges, node_labels=labels)

    # def test_vs_grakel(self):
    #     kernel = WeisfeilerLehmanKernel(depth=1)
    #     kernel_values = kernel(self.graphs, self.graphs)

    #     graphs_grakel = [
    #         TestWeisfeilerLehmanKernel.from_ours_to_grakels_graph(g)
    #         for g in self.graphs
    #     ]

    #     kernel_grakel = grakel.kernels.WeisfeilerLehman(n_iter=1)
    #     grakel_values = kernel_grakel.fit_transform(graphs_grakel)

    #     np.testing.assert_almost_equal(kernel_values, grakel_values)


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
    g = nx.Graph()

    edges = [(1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 5), (3, 6)]
    labels = [5, 2, 4, 3, 1, 1, 1]

    g.add_edges_from(edges)

    for node in g.nodes():
        g.nodes[node]["labels"] = [labels[node - 1]]

    return g


def G2():
    g = nx.Graph()

    edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 6)]
    labels = [2, 5, 4, 3, 1, 2, 2]

    g.add_edges_from(edges)

    for node in g.nodes():
        g.nodes[node]["labels"] = [labels[node - 1]]

    return g
