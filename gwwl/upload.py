import multiprocessing
import os
import fire
import subprocess
from tqdm import tqdm
from functools import partial


def process_folder(const, folder):
    archive_folder, urls_folder = const
    name = os.path.basename(folder)
    archive_file = os.path.join(archive_folder, name + ".tar.bz2")
    url_file = os.path.join(urls_folder, name + ".txt")
    if os.path.isfile(url_file):
        return None
    # compress
    with open(os.devnull, "w") as devnull:
        subprocess.run(
            ["tar", "-cjf", archive_file, folder],
            stdout=devnull,
            stderr=subprocess.PIPE,
        )
    # upload
    with open(os.devnull, "w") as devnull:
        curl_cmd = ["curl", "--upload-file", archive_file, "https://transfer.sh/"]
        url = subprocess.check_output(curl_cmd, stderr=devnull).decode().strip()
    # write
    with open(url_file, "w") as f:
        f.write(url)
    return url


def main(root_folder, archive_folder, urls_folder, nb_processes):
    folders = [
        os.path.join(root_folder, folder)
        for folder in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, folder))
    ]
    with multiprocessing.Pool(nb_processes) as pool:
        results = []
        func = partial(process_folder, (archive_folder, urls_folder))
        for _ in tqdm(pool.imap_unordered(func, folders), total=len(folders)):
            results.append(_)


if __name__ == "__main__":
    fire.Fire(main)
    # main(
    #     root_folder="D:/Documents_D/Mines/Cours/Master/Kernel/cost_matrices/",
    #     archive_folder="D:/Documents_D/Mines/Cours/Master/Kernel/archives/",
    #     urls_folder="D:/Documents_D/Mines/Cours/Master/Kernel/urls/",
    #     nb_processes=8,
    # )
