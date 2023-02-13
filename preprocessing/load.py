import os
import pickle
import numpy as np


def load_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if not (isinstance(data, list) or isinstance(data, np.ndarray)):
        raise ValueError(f"{file_path} should contains a list")
    return data


def load_data(data_directory="data/"):
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(f"{data_directory} does not exist.")
    files_list = {
        "x_test": "test_data.pkl",
        "x": "training_data.pkl",
        "y": "training_labels.pkl",
    }

    data = {}
    for file_name in files_list:
        data[file_name] = load_file(os.path.join(data_directory, files_list[file_name]))

    if len(data["x"]) == len(data["y"]):
        raise ValueError(f"There is not the same amount of data in x than labels in y.")

    return data


### Unit test ###
if __name__ == "__main__":
    a = load_data()
    print("Data loading is successful.")
    print(a["y"])
