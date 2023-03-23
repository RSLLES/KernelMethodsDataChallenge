import numpy as np


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

    def __init__(self, use_cache: bool = False):
        """
        Initialize the Kernel instance.

        Parameters:
            use_cache (bool): Whether to use caching for phi or kernel calls.
        """
        self.use_cache = use_cache
        self.reset_cache()
        self.nb_heavy_call = 0

    def reset_cache(self):
        """Reset the cache dictionary."""
        if self.use_cache:
            self.cache = {}

    def __call__(self, x1, x2, *args, **kwargs):
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
            return self._phi_call(x1, x2, *args, **kwargs)
        elif hasattr(self, "kernel"):
            return self._kernel_call(x1, x2, *args, **kwargs)
        else:
            raise NotImplementedError(
                "A subclass of `Kernel` should implement either a kernel or a phi and an inner functions."
            )

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

        kernel_value = self.kernel(x1=x1, x2=x2, *args, **kwargs)

        if self.use_cache:
            self.cache[h] = kernel_value

        return np.array(kernel_value)