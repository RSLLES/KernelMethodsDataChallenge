import itertools
import numpy as np
import multiprocessing
import tqdm
from time import sleep
import os


class Kernel:
    """
    A base class for kernel methods.

    To create a subclass of `Kernel`, implement either a kernel method or a phi and an inner function.
    - a kernel method should take as input two elements x1 and x2, and return the scalar value output of
      the kernel function: k(x1,x2)
    - a phi method should take only one element x and may return whatever object x', embedding of x in the RKHS.
      The method inner then takes two x'1 and x'2 and should output their inner product.
    If both cases are implemented, the defaut method would be the second one.
    Note that caching and recursion if inputs are lists are automatically handled by this class.
    """

    def __init__(self, use_cache: bool = False, multiprocess: bool = False):
        """
        Initialize the Kernel instance.

        Parameters:
            use_cache (bool): Whether to use caching for phi or kernel calls.
        """
        self.use_cache = use_cache
        self.multiprocess = multiprocess
        self.reset_cache()
        self.nb_heavy_call = 0

    def reset_cache(self):
        """Reset the cache dictionary."""
        if self.use_cache:
            self.cache = {}

    def __call__(self, x1, x2=None, *args, **kwargs):
        """
        Call the Kernel instance.

        Parameters:
            x1: First input of the kernel method.
            x2: Second input of the kernel method.
            *args and **kwargs: Any additionnal parameters
            for the kernel or the phi method.

        Returns:
            A scalar output if x1 and x2 are scalars.
            A 1D array if either x1 or x2 is a list, and the other a scalar.
            A 2d array if both x1 and x2 are list.
        Exceptions:
            If either a kernel function or a phi and an inner functions are defined,
            it will raise a NotImplementedError.
        """
        if hasattr(self, "phi") and hasattr(self, "inner"):
            if x2 is None:
                return self._solo_phi(x1)
            else:
                return self._phi_call(x1, x2, *args, **kwargs)
        elif hasattr(self, "kernel"):
            return self._kernel_call(x1, x2, *args, **kwargs)
        else:
            raise NotImplementedError(
                "A subclass of `Kernel` should implement either a kernel or a phi and an inner functions."
            )

    def _solo_phi(self, x):
        assert isinstance(x, list)
        n = len(x)
        self.granularite = max(1, n * (n + 1) // 2 // 2000 // 4)

        # Compute embedding
        z = [self.phi(x) for x in tqdm.tqdm(x, desc="Computing Embedding")]

        # Compute inner products with multiprocessing
        m = n // 2

        K = np.zeros((n, n))

        # Processess
        pool = multiprocessing.Pool(4)
        manager = multiprocessing.Manager()
        self.q = manager.Queue()

        nord_ouest = pool.apply_async(self.solve_triangular, (z[:m], z[:m]))
        sud_est = pool.apply_async(self.solve_triangular, (z[m:], z[m:]))
        nord_est = pool.apply_async(self.solve_triangular, (z[:m], z[m:]))
        center = pool.apply_async(self.solve_triangular, (z[:m], z[m:], False))

        last_v, v = 0, 0
        with tqdm.tqdm(desc="Inner product", total=n * (n + 1) // 2) as pbar:
            while (
                not nord_ouest.ready()
                and not sud_est.ready()
                and not nord_est.ready()
                and not center.ready()
            ):
                v = self.q.qsize()
                if v > last_v:
                    pbar.update(self.granularite * (v - last_v))
                    last_v = v
                sleep(0.01)
            pbar.update(pbar.total - pbar.n)

        K[:m, :m] += nord_ouest.get()
        K[m:, m:] += sud_est.get()
        K[:m, m:] += nord_est.get()
        K[:m, m:] += center.get()

        pool.close()

        # Symmetry
        r, c = np.triu_indices(n, k=1)
        K[c, r] = K[r, c]

        return K

    def solve_triangular(self, X, Y, upper=True):
        try:
            assert len(X) == len(Y)
            n = len(X)
            K = np.zeros((n, n))
            if upper:
                gen = list(
                    (i, j)
                    for i, j in itertools.product(range(len(X)), range(len(Y)))
                    if i <= j
                )
            else:
                gen = list(
                    (i, j)
                    for i, j in itertools.product(range(len(X)), range(len(Y)))
                    if i > j
                )
            for k, (i, j) in enumerate(gen):
                K[i, j] = self.inner(X[i], Y[j])
                if k % self.granularite == 0:
                    self.q.put_nowait(True)

            return K
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            return None

    def _phi_call(self, x1, x2, *args, **kwargs):
        """
        Handle computation for the phi and inner case.

        Parameters:
            x1: First input of the kernel method.
            x2: Second input of the kernel method.
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            An array of embedded points.
        """
        if isinstance(x1, list):
            results = [self._phi_call(x1=item, x2=x2, *args, **kwargs) for item in x1]
            return np.array(results)
        elif isinstance(x2, list):
            results = [self._phi_call(x1=x1, x2=item, *args, **kwargs) for item in x2]
            return np.array(results)

        if self.use_cache:
            h1, h2 = hash(x1), hash(x2)
            if h1 in self.cache:
                xp1 = self.cache[h1]
            else:
                self.nb_heavy_call += 1
                xp1 = self.phi(x1, *args, **kwargs)
                self.cache[h1] = xp1

            if h2 in self.cache:
                xp2 = self.cache[h2]
            else:
                self.nb_heavy_call += 1
                xp2 = self.phi(x2, *args, **kwargs)
                self.cache[h2] = xp2
        else:
            self.nb_heavy_call += 2
            xp1 = self.phi(x1, *args, **kwargs)
            xp2 = self.phi(x2, *args, **kwargs)

        kernel_value = self.inner(xp1, xp2)

        return np.array(kernel_value)

    def _kernel_call(self, x1, x2, *args, **kwargs):
        """
        Handle computation for the kernel case.

        Parameters:
            x1: First input of the kernel method.
            x2: Second input of the kernel method.
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A scalar output of the kernel function.
        """
        if isinstance(x1, list):
            results = [
                self._kernel_call(x1=item, x2=x2, *args, **kwargs) for item in x1
            ]
            return np.array(results)
        elif isinstance(x2, list):
            results = [
                self._kernel_call(x1=x1, x2=item, *args, **kwargs) for item in x2
            ]
            return np.array(results)

        if self.use_cache:
            h1, h2 = hash(x1), hash(x2)
            h = (h2, h1) if h1 > h2 else (h1, h2)
            if h in self.cache:
                return self.cache[h]

        self.nb_heavy_call += 1
        kernel_value = self.kernel(x1=x1, x2=x2, *args, **kwargs)

        if self.use_cache:
            self.cache[h] = kernel_value

        return np.array(kernel_value)
