import os
import multiprocessing
import subprocess
from tqdm import tqdm
from functools import partial
import fire


def download_and_extract(const, url_file):
    archive_folder, root_folder = const
    folder_name = os.path.splitext(os.path.basename(url_file))[
        0
    ]  # Get the name of the folder
    archive_file = os.path.join(
        archive_folder, folder_name + ".tar.bz2"
    )  # Get the path to the archive file
    with open(url_file, "r") as f:
        url = f.read().strip()  # Read the URL from the file
    download_cmd = ["curl", "-L", "-o", archive_file, url]  # Download command
    extract_cmd = ["tar", "-xjf", archive_file, "-C", root_folder]  # Extract command
    with open(os.devnull, "w") as devnull:
        subprocess.run(
            download_cmd, stdout=devnull, stderr=subprocess.PIPE
        )  # Download the file
        subprocess.run(
            extract_cmd, stdout=devnull, stderr=subprocess.PIPE
        )  # Extract the archive file to the root folder
    # os.remove(archive_file)  # Remove the archive file


def main(urls_folder, archive_folder, root_folder, nb_processes):
    urls_files = [
        os.path.join(urls_folder, f)
        for f in os.listdir(urls_folder)
        if os.path.isfile(os.path.join(urls_folder, f)) and f.endswith(".txt")
    ]  # Get all the URL files from the urls_folder
    with multiprocessing.Pool(nb_processes) as pool:
        results = []
        func = partial(download_and_extract, (archive_folder, root_folder))
        for _ in tqdm(pool.imap_unordered(func, urls_files), total=len(urls_files)):
            results.append(_)


if __name__ == "__main__":
    fire.Fire(main)
    # main(
    #     urls_folder="D:/Documents_D/Mines/Cours/Master/Kernel/urls/",
    #     archive_folder="D:/Documents_D/Mines/Cours/Master/Kernel/archives/",
    #     root_folder="D:/Documents_D/Mines/Cours/Master/Kernel/retrieve/",
    #     nb_processes=1,
    # )
