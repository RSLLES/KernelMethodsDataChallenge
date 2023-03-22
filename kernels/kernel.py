import numpy as np


class Kernel:
    def __init__(self) -> None:
        self.cache = {}

    def __call__(self, x1, x2, *args, use_cache=False, **kargs):
        if isinstance(x1, list):
            results = []
            for item in x1:
                results.append(
                    self.__call__(x1=item, x2=x2, *args, use_cache=use_cache, **kargs)
                )
            return np.array(results)
        if isinstance(x2, list):
            results = []
            for item in x2:
                results.append(
                    self.__call__(x1=x1, x2=item, *args, use_cache=use_cache, **kargs)
                )
            return np.array(results)

        if use_cache:
            h1, h2 = hash(x1), hash(x2)
            h = (h2, h1) if h1 > h2 else (h1, h2)
            if h in self.cache:
                return self.cache[h]

        kernel_value = self.kernel(x1=x1, x2=x2)

        if use_cache:
            self.cache[h] = kernel_value

        return np.array(kernel_value)

    def kernel(self, x1, x2):
        raise NotImplementedError(
            "The `kernel` method should be redefined in sub classes."
        )
