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

    def __init__(
        self,
        use_cache: bool = False,
        multiprocess: bool = False,
        verbose: bool = True,
        processess: int = None,
    ):
        """
        Initialize the Kernel instance.

        Parameters:
            use_cache (bool): Whether to use caching for phi or kernel calls.
        """
        self.use_cache = use_cache
        self.multiprocess = multiprocess
        self.reset_cache()
        self.nb_heavy_call = 0
        self.verbose = verbose
        self.processes = processess or max(os.cpu_count() - 4, os.cpu_count() // 2, 1)
        if verbose:
            print(f"[Init] Using {self.processes} processes.")

    def set_verbose(self, verbose):
        assert isinstance(verbose, bool)
        if verbose != self.verbose:
            print(f"Verbosity set to {verbose}")
            self.verbose = verbose

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
                return self._phi(x1, x2, *args, **kwargs)
        elif hasattr(self, "kernel"):
            if x2 is None:
                return self._solo_kernel(x1, *args, **kwargs)
            else:
                return self._kernel(x1, x2, *args, **kwargs)
        else:
            raise NotImplementedError(
                "A subclass of `Kernel` should implement either a kernel or a phi and an inner functions."
            )

    def _tqdm(self, *args, **kargs):
        return tqdm.tqdm(*args, **kargs, disable=not self.verbose)

    def _solo_phi(self, x):
        assert isinstance(x, list)
        n = len(x)
        self.granularite = max(100, n * (n + 1) // 2 // 2000 // 4)

        # Compute embedding
        def work_with_cache(x):
            if self.use_cache:
                h = hash(x)
                if h not in self.cache:
                    self.cache[h] = self.phi(x)
                    self.nb_heavy_call += 1
                return self.cache[h]
            self.nb_heavy_call += 1
            return self.phi(x)

        self._z = [
            work_with_cache(x)
            for x in self._tqdm(x, desc="[Fit_Phi] Computing Embedding")
        ]

        # Compute inner products with multiprocessing
        K = np.zeros((n, n))
        m = n // 2

        R, C = np.triu_indices(n)
        R, C = np.array_split(R, self.processes), np.array_split(C, self.processes)

        with multiprocessing.Pool(processes=self.processes) as p:
            m = multiprocessing.Manager()
            self.q = m.Queue(maxsize=n * (n + 1) // 2)
            res = p.map_async(self._multi_inner_solo, zip(R, C))

            # Wait
            with self._tqdm(
                total=n * (n + 1) // 2, desc="[Fit_Phi] Computing Inner products"
            ) as pbar:
                last_v, v = 0, 0
                while not res.ready():
                    v = self.q.qsize()
                    if v > last_v:
                        pbar.update(self.granularite * (v - last_v))
                        last_v = v
                    sleep(0.01)
                pbar.update(pbar.total - pbar.n)
        for values, r, c in zip(res.get(), R, C):
            K[r, c] = values

        # Symmetry
        r, c = np.triu_indices(n, k=1)
        K[c, r] = K[r, c]

        return K

    def _multi_inner_solo(self, H):
        try:
            r, c = H
            K = np.zeros((len(r),))
            for k, (i, j) in enumerate(zip(r, c)):
                K[k] = self.inner(self._z[i], self._z[j])
                if k % self.granularite == 0:
                    self.q.put(True)
            return K
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            return None

    def _phi(self, x1, x2):
        """
        Handle computation for the phi and inner case.

        Parameters:
            x1: First input of the kernel method.
            x2: Second input of the kernel method.

        Returns:
            An array of embedded points.
        """
        if not isinstance(x1, list):
            x1 = [x1]
        if not isinstance(x2, list):
            x2 = [x2]

        n, m = len(x1), len(x2)
        self.granularite = max(100, n * m // 2000 // 4)

        # Compute embedding
        def work_with_cache(x):
            if self.use_cache:
                h = hash(x)
                if h not in self.cache:
                    self.nb_heavy_call += 1
                    self.cache[h] = self.phi(x)
                return self.cache[h]
            self.nb_heavy_call += 1
            return self.phi(x)

        self._z1 = [
            work_with_cache(x)
            for x in self._tqdm(x1, desc="[Transform_Phi] Computing Embedding 1")
        ]

        self._z2 = [
            work_with_cache(x)
            for x in self._tqdm(x2, desc="[Transform_Phi] Computing Embedding 2")
        ]

        # Compute inner products with multiprocessing
        K = np.zeros((n, m))

        R, C = np.array(list(np.ndindex(K.shape))).T
        R, C = np.array_split(R, self.processes), np.array_split(C, self.processes)

        with multiprocessing.Pool(processes=self.processes) as p:
            man = multiprocessing.Manager()
            self.q = man.Queue(maxsize=n * m)
            res = p.map_async(self._multi_inner, zip(R, C))

            # Wait
            with self._tqdm(
                total=n * m, desc="[Transform_Phi] Computing Inner products"
            ) as pbar:
                last_v, v = 0, 0
                while not res.ready():
                    v = self.q.qsize()
                    if v > last_v:
                        pbar.update(self.granularite * (v - last_v))
                        last_v = v
                    sleep(0.01)
                pbar.update(pbar.total - pbar.n)
        for values, r, c in zip(res.get(), R, C):
            K[r, c] = values

        return K.squeeze()

    def _multi_inner(self, H):
        try:
            r, c = H
            K = np.zeros((len(r),))
            for k, (i, j) in enumerate(zip(r, c)):
                K[k] = self.inner(self._z1[i], self._z2[j])
                if k % self.granularite == 0:
                    self.q.put(True)
            return K
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            return None

    def _solo_kernel(self, x):
        assert isinstance(x, list)
        n = len(x)
        self._X = x
        self.granularite = max(100, n * (n + 1) // 2 // 2000 // 4)

        # Compute inner products with multiprocessing
        K = np.zeros((n, n))
        m = n // 2

        R, C = np.triu_indices(n)
        R, C = np.array_split(R, self.processes), np.array_split(C, self.processes)

        with multiprocessing.Pool(processes=self.processes) as p:
            m = multiprocessing.Manager()
            self.q = m.Queue(maxsize=n * (n + 1) // 2)
            res = p.map_async(self._multi_kernel, zip(R, C))

            # Wait
            with self._tqdm(total=n * (n + 1) // 2, desc="[Fit] Kernel") as pbar:
                last_v, v = 0, 0
                while not res.ready():
                    v = self.q.qsize()
                    if v > last_v:
                        pbar.update(self.granularite * (v - last_v))
                        last_v = v
                    sleep(0.01)
                pbar.update(pbar.total - pbar.n)
        for values, r, c in zip(res.get(), R, C):
            K[r, c] = values

        # Symmetry
        r, c = np.triu_indices(n, k=1)
        K[c, r] = K[r, c]

        return K

    def _multi_kernel(self, H):
        try:
            r, c = H
            K = np.zeros((len(r),))
            for k, (i, j) in enumerate(zip(r, c)):
                K[k] = self.kernel(self._X[i], self._X[j])
                if k % 1500 == 0:
                    self.q.put(True)
            return K
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            return None

    def _kernel(self, x1, x2, *args, **kwargs):
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
            results = [self._kernel(x1=item, x2=x2, *args, **kwargs) for item in x1]
            return np.array(results)
        elif isinstance(x2, list):
            results = [self._kernel(x1=x1, x2=item, *args, **kwargs) for item in x2]
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
