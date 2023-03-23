from svc.svc import SVC
from preprocessing.load import load_data
import numpy as np
import fire
import importlib.util
import sys
import os
import pandas as pd


def print_metrics(score_output):
    s = ""
    acc, prec, rec, f1 = score_output
    s += f"Accuracy = {acc*100:0.1f}%, "
    s += f"Precision = {prec*100:0.1f}%, "
    s += f"Recall = {rec*100:0.1f}%, "
    s += f"F1 = {f1*100:0.1f}%"

    print(s)


def run(config):
    # Loading
    print("Loading data...")
    ds, ds_val = load_data(config=config)
    kernel = config.kernel

    # Training
    print("Training ...")
    scores = []
    for fold, (X, y, Xv, yv) in enumerate(ds):
        print(f"### Fold {fold+1}/{len(ds)} ###")

        svc = SVC(kernel=kernel, verbose=True)
        svc.fit(X=X, y=np.array(y))

        score = svc.score(X=Xv, y=yv.astype(bool))
        print_metrics(score)
        scores.append(score)

    print("### Summary ###")
    logs = [(f"Fold {fold+1}",) + score for fold, score in enumerate(scores)]
    scores = sum([np.array(score) for score in scores]) / len(scores)
    logs += [("Average",) + tuple(scores)]
    df = pd.DataFrame(logs, columns=["/", "Accuracy", "Precision", "Recall", "F1"])
    print(df.to_markdown(index=False))


def main(config_path):
    """
    This Python script trains and validates a Support Vector Classifier.
    It uses the configuration provided by the file provided as argument.
    This configuration file must contains:

    ### Data ###
    - data_directory (str): Path to directory containing data files.
    - x_val (str): File name of pickled validation dataset.
    - x (str): File name of pickled training dataset.
    - y (str): File name of pickled training labels.
    - k_folds_cross_val (int): Number of folds for cross validation.

    ### Kernel ###
    - kernel: Initialized kernel object.

    Args:
        config_path (str): Path to python module describing configuration for current run.

    """
    filename, _ = os.path.splitext(os.path.basename(config_path))
    config_name = filename
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config
    spec.loader.exec_module(config)

    run(config=config)


if __name__ == "__main__":
    fire.Fire(main)
