from svc.svc import SVC
from preprocessing.load import load_data
import numpy as np
import fire
import importlib.util
import sys
import os
import pandas as pd


def print_metrics(score_output):
    """Prints metrics produced by SVC.score function"""
    acc, prec, rec, f1 = score_output
    s = f"Accuracy = {acc*100:.1f}%, Precision = {prec*100:.1f}%, Recall = {rec*100:.1f}%, F1 = {f1*100:.1f}%"
    print(s)


def train_and_score(kernel, X, y, Xv, yv):
    """
    Trains a classifier on a training set and returns the score on a holdout set.

    Args:
        kernel (str): The kernel to be used by the classifier.
        X (numpy.array): The features for the training set.
        y (numpy.array): The labels for the training set.
        Xv (numpy.array): The features for the holdout set.
        yv (numpy.array): The labels for the holdout set.

    Returns:
        float: The score of the trained classifier on the holdout set.
    """
    svc = SVC(kernel=kernel, verbose=True)
    svc.fit(X=X, y=np.array(y))
    score = svc.score(X=Xv, y=yv.astype(bool))
    print_metrics(score)
    return score


def test(config, ds, ds_val, kernel):
    """
    Tests the performance of the classifier on the validation set.

    Args:
        config: The configuration module.
        ds (iterable): An iterable of training/validation data.
        ds_val (iterable): An iterable of validation data.
        kernel: The kernel to be used by the classifier.
    """
    print("### On validation data ###")
    print("Training full SVC...")
    X, y = ds.full()
    X_val, _ = ds_val.full()
    svc = SVC(kernel=kernel, verbose=True)
    svc.fit(X=X, y=np.array(y))
    print("Predicting values...")
    y_pred = svc.predict(X_val)
    print("Exporting...")
    if not os.path.isdir(config.save_directory):
        os.mkdir(config.save_directory)
    module_name = [key for key in sys.modules if sys.modules[key] == config].pop()
    df = pd.DataFrame(
        y_pred, columns=["Predicted"], index=pd.RangeIndex(1, len(y_pred) + 1)
    )
    df.index.name = "Id"
    df.to_csv(os.path.join(config.save_directory, module_name + ".csv"))


def run(config, predict):
    """
    Runs the entire pipeline for training and testing a classifier.

    Args:
        config (object): A configuration object.
        predict (bool): A flag indicating whether to predict on a holdout set.

    Returns:
        None
    """
    # Loading
    print("Loading data...")
    ds, ds_val = load_data(config=config)
    kernel = config.kernel

    # Training
    print("Training ...")
    scores = []
    for fold, (X, y, Xv, yv) in enumerate(ds):
        print(f"### Fold {fold+1}/{len(ds)} ###")
        score = train_and_score(kernel, X, y, Xv, yv)
        scores.append(score)

    # Summary
    print("### Summary ###")
    logs = [(f"Fold {fold+1}", *score) for fold, score in enumerate(scores)]

    scores = [np.array(score) for score in scores]
    average_scores = sum(scores) / len(scores)
    average_log = ("Average", *tuple(average_scores))

    df = pd.DataFrame(
        [*logs, average_log], columns=["", "Accuracy", "Precision", "Recall", "F1"]
    )

    cols_to_convert = ["Accuracy", "Precision", "Recall", "F1"]
    df[cols_to_convert] = (df[cols_to_convert] * 100).round(1).astype(str) + "%"
    print(df.to_markdown(index=False))

    # Test
    if predict:
        test(config, ds, ds_val, kernel)


def main(config_path: str, predict=False):
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

    ### Save ###
    - save_directory (str) : Path to directory where the output csv will be saved
    if the predict flag is enabled.

    Args:
        config_path (str): Path to python module describing configuration for current run.
        predict (bool, default : False) : Wether or not to predict outputs for validation data.

    """
    filename, _ = os.path.splitext(os.path.basename(config_path))
    config_name = filename
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config
    spec.loader.exec_module(config)

    run(config=config, predict=predict)


if __name__ == "__main__":
    fire.Fire(main)
