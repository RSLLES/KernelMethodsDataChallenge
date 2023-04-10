import fire
import numpy as np
from tqdm import tqdm
import os


def main(root_folder, n=6000):
    print("Checking completeness ...")
    R, C = np.triu_indices(n)
    for i, j in tqdm(zip(R, C), total=n * (n + 1) // 2):
        path = os.path.join(root_folder, str(i), str(j) + ".npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}.")
    print("Complete.")


if __name__ == "__main__":
    # main(
    #     root_folder="D:/Documents_D/Mines/Cours/Master/Kernel/cost_matrices/",
    # )
    fire.Fire(main)
