from preprocessing.load import load_data
import configs.gwwl_depth3 as config
from tqdm import tqdm
import numpy as np
import multiprocessing
from time import sleep
import os
from treeEditDistance import treeEditDistance
from functools import partial


def search_for(const, var):
    try:
        path = const
        i, j = var
        file = os.path.join(path, str(i), f"{j}.npy")
        return os.path.isfile(file)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        return None


def compute(const, var):
    try:
        Z, path = const
        r, c = var

        for i, j in zip(r, c):
            directory = os.path.join(path, str(i))
            if not os.path.exists(directory):
                os.makedirs(directory)

            file = os.path.join(directory, f"{j}.npy")
            if not os.path.isfile(file):
                D = treeEditDistance(Z[i], Z[j])
                np.save(file, D)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")


def main(path, processes=None, flip=False):
    print("Loading data...")
    ds, _ = load_data(config=config)
    X = ds.X
    n = len(X)
    granularite = max(100, n * (n + 1) // 2 // 2000 // 4)

    kernel = config.kernel
    kernel.set_processes(processes)
    processes = kernel.processes

    Z = [kernel.phi(x) for x in tqdm(X, desc="Computing Embedding ...")]

    R, C = np.triu_indices(n)
    if flip:
        R = np.flip(R)
        C = np.flip(C)

    R = np.array_split(R, len(R) // granularite)
    C = np.array_split(C, len(C) // granularite)
    with multiprocessing.Pool(processes=processes) as p:
        func = partial(compute, (Z, path))
        s = 0
        for _ in tqdm(
            p.imap_unordered(func, zip(R, C)),
            total=len(R) // granularite,
            desc="Inner prod...",
        ):
            s += 1


import fire

if __name__ == "__main__":
    fire.Fire(main)
