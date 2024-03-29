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
        Z1, Z2, path = const
        r, c = var

        for i, j in zip(r, c):
            directory = os.path.join(path, str(i))
            if not os.path.exists(directory):
                os.makedirs(directory)

            file = os.path.join(directory, f"{j}.npy")
            compute = True
            if os.path.isfile(file):
                try:
                    _ = np.load(file)
                    compute = False
                except:
                    print(f"{file} corrupted.")
                    os.remove(file)

            if compute:
                D = treeEditDistance(Z1[i], Z2[j])
                np.save(file, D)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")


def main(path, processes=None, flip=False):
    print("Loading data...")
    ds, ds_val = load_data(config=config)
    X1 = ds_val.X
    X2 = ds.X
    granularite = max(100, len(X1) * len(X2) // 10000)

    kernel = config.kernel
    kernel.set_processes(processes)
    processes = kernel.processes

    Z1 = [kernel.phi(x) for x in tqdm(X1, desc="Computing Embedding ...")]
    Z2 = [kernel.phi(x) for x in tqdm(X2, desc="Computing Embedding ...")]

    R, C = np.array(list(np.ndindex((len(X1), len(X2))))).T

    R = np.array_split(R, len(R) // granularite)
    C = np.array_split(C, len(C) // granularite)
    with multiprocessing.Pool(processes=processes) as p:
        func = partial(compute, (Z1, Z2, path))
        s = 0
        for _ in tqdm(
            p.imap_unordered(func, zip(R, C)),
            total=len(R),
            desc="Inner prod...",
        ):
            s += 1


import fire

if __name__ == "__main__":
    fire.Fire(main)
