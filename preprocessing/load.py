import os
import pickle
import numpy as np

from dataset.dataset import Dataset


def load_file(file_path):
    """
    This function takes in a file path and attempts to load the file
    using pickle.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if not (isinstance(data, list) or isinstance(data, np.ndarray)):
        raise ValueError(f"{file_path} should contains a list or an array.")
    return data


def load_data(config):
    """
    This function loads the data from a given directory. It takes in a configuration object as an argument.
    It returns either two dataset objects, respectively ds and ds_validation, or only ds if there is no x_val in
    the configuration file.
    """
    if not os.path.isdir(config.data_directory):
        raise FileNotFoundError(f"{config.data_directory} does not exist.")
    if not (hasattr(config, "x") and hasattr(config, "y")):
        raise ValueError(
            f"config module {config} should have an 'x' and 'y' attribute specifying the training files."
        )

    x = load_file(os.path.join(config.data_directory, config.x))
    y = load_file(os.path.join(config.data_directory, config.y))
    ds = Dataset(x=x, y=y, prop_test=config.prop_test, k_folds=config.k_folds_cross_val)

    if hasattr(config, "x_val"):
        x_val = load_file(os.path.join(config.data_directory, config.x_val))
        ds_val = Dataset(x=x_val)
        return ds, ds_val

    return ds


### Unit test ###
if __name__ == "__main__":
    import configs.challenge as config

    ds, _ = load_data(config=config)
    print("Data loaded with success.")
    print(ds.y[:20])
