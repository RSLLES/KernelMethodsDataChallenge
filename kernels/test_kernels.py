import unittest

from kernels.WL import WeisfeilerLehmanKernel


class TestWeisfeilerLehmanKernel(unittest.TestCase):
    def setUp(self) -> None:
        from preprocessing.load import load_file

        self.graphs = load_file("data/training_data.pkl")[0:10]
        self.kernel = WeisfeilerLehmanKernel(depth=3)

    def test_kernel_computation_single_graph(self):
        g1 = self.graphs[0]
        g2 = self.graphs[2]
        computed_kernel_value = self.kernel(g1, g2, use_cache=False)
        self.assertEqual(computed_kernel_value, 215)

    def test_kernel_computation_list_of_graphs(self):
        expected_kernel_values = [
            [264, 322, 215, 420, 385, 255, 248, 284, 175, 312],
            [322, 654, 328, 767, 622, 496, 478, 478, 271, 484],
            [215, 328, 274, 441, 382, 271, 253, 278, 189, 346],
            [420, 767, 441, 1212, 799, 624, 596, 595, 379, 680],
            [385, 622, 382, 799, 858, 510, 480, 512, 384, 652],
            [255, 496, 271, 624, 510, 494, 387, 365, 234, 412],
            [248, 478, 253, 596, 480, 387, 422, 381, 221, 390],
            [284, 478, 278, 595, 512, 365, 381, 524, 261, 452],
            [175, 271, 189, 379, 384, 234, 221, 261, 380, 558],
            [312, 484, 346, 680, 652, 412, 390, 452, 558, 1056],
        ]

        computed_kernel_values_without_cache = self.kernel(
            self.graphs, self.graphs, use_cache=False
        )
        computed_kernel_values = self.kernel(self.graphs, self.graphs)
        self.assertEqual(computed_kernel_values_without_cache, expected_kernel_values)
        self.assertEqual(computed_kernel_values, expected_kernel_values)
