import os
import multiprocessing
from tqdm import tqdm


# define function to move a single file
def move_file(file_path):
    try:
        orig_path, new_path = file_path
        # print(f"{orig_path} -> {new_path}")
        directory = os.path.dirname(new_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.rename(orig_path, new_path)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")


def move_files(nb_processes):
    # get list of files
    files = [
        f
        # for f in os.listdir("D:/Documents_D/Mines/Cours/Master/Kernel/cost_matrices")
        for f in os.listdir(".")
        if f.endswith(".npy")
    ]

    # create list of tuples containing original file path and new file path
    file_paths = [(f, "{}/{}".format(*f.split("_"))) for f in files]

    # use multiprocessing to move the files in parallel
    with multiprocessing.Pool(nb_processes) as pool:
        results = []
        for _ in tqdm(
            pool.imap_unordered(move_file, file_paths), total=len(file_paths)
        ):
            results.append(_)


if __name__ == "__main__":
    move_files(1)
