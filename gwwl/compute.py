from preprocessing.load import load_data
import configs.gwwl_depth3 as config
from tqdm import tqdm
import numpy as np
import multiprocessing
from time import sleep
import os
from treeEditDistance import treeEditDistance
from functools import partial


def compute(const, var):
    try:
        q, Z, granularite, path = const
        r, c = var

        for k, (i, j) in enumerate(zip(r, c)):
            directory = os.path.join(path, str(i))
            if not os.path.exists(directory):
                os.makedirs(directory)

            file = os.path.join(directory, f"{j}.npy")
            if not os.path.isfile(file):
                D = treeEditDistance(Z[i], Z[j])
                np.save(file, D)

            if k % granularite == 0:
                q.put(True)
        return 0.0
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        return None


def main(path, processes=None, flip=False):
    print("Loading data...")
    ds, ds_val = load_data(config=config)
    X = ds.X
    n = len(X)

    kernel = config.kernel
    kernel.set_processes(processes)
    processes = kernel.processes
    granularite = max(100, n * (n + 1) // 2 // 2000 // 4)

    Z = [kernel.phi(x) for x in tqdm(X, desc="Computing Embedding ...")]

    R, C = np.triu_indices(n)
    if flip:
        R = np.flip(R)
        C = np.flip(C)
    R, C = np.array_split(R, processes), np.array_split(C, processes)
    with multiprocessing.Pool(processes=processes) as p:
        m = multiprocessing.Manager()
        q = m.Queue(maxsize=n * (n + 1) // 2)
        func = partial(compute, (q, Z, granularite, path))
        res = p.map_async(func, zip(R, C))

        # Wait
        with tqdm(total=n * (n + 1) // 2, desc="Inner prods ...") as pbar:
            last_v, v = 0, 0
            while not res.ready():
                v = q.qsize()
                if v > last_v:
                    pbar.update(granularite * (v - last_v))
                    last_v = v
                sleep(0.01)
            pbar.update(pbar.total - pbar.n)

    res.wait()


import fire

if __name__ == "__main__":
    fire.Fire(main)
