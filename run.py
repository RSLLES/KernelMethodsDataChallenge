from svc.svc import SVC
from svc.svc import score as score_metric

# from sklearn.svm import SVC
from preprocessing.load import load_data
import numpy as np
import fire
import importlib.util
import sys
import os
import pandas as pd


def print_metrics(score_output):
    """Prints metrics produced by SVC.score function"""
    acc, prec, rec, f1, rocauc = score_output
    s = f"Accuracy = {acc*100:.1f}%, Precision = {prec*100:.1f}%, Recall = {rec*100:.1f}%, F1 = {f1*100:.1f}%, ROCAUC = {rocauc*100:.1f}%"
    print(s)


def train_and_score(kernel, X, y, Xv, yv, verbose):
    """
    Trains a classifier on a training set and returns the score on a holdout set.
    """
    svc = SVC(kernel=kernel, verbose=verbose)
    svc.fit(X=X, y=np.array(y))
    score = score_metric(svc=svc, X=Xv, y=yv.astype(bool))
    print_metrics(score)
    return score


def test(config, ds, ds_val, kernel, filename):
    """
    Tests the performance of the classifier on the validation set.
    """
    print("### On validation data ###")
    print("Training full SVC...")
    X, y = ds.full()
    X_val, _ = ds_val.full()
    svc = SVC(kernel=kernel, verbose=True)
    svc.fit(X=X, y=np.array(y))
    print("Predicting values...")
    y_pred = svc.decision_function(X_val)
    print("Exporting...")
    if not os.path.isdir(config.export_directory):
        os.mkdir(config.export_directory)
    df = pd.DataFrame(
        y_pred,
        columns=["Predicted"],
        index=pd.RangeIndex(1, len(y_pred) + 1, name="Id"),
    )
    df.to_csv(os.path.join(config.export_directory, filename + ".csv"))


def get_summary(scores):
    logs = [(f"Fold {fold+1}", *score) for fold, score in enumerate(scores)]

    scores = [np.array(score) for score in scores]
    average_scores = sum(scores) / len(scores)
    average_log = ("Average", *tuple(average_scores))

    cols = ["Accuracy", "Precision", "Recall", "F1", "ROCAUC"]
    df = pd.DataFrame([*logs, average_log], columns=[""] + cols)

    df[cols] = (df[cols] * 100).round(1).astype(str) + "%"
    return df.to_markdown(index=False)


def run(config, performance, predict, filename, verbose):
    """
    Runs the entire pipeline for training and testing a classifier.
    """
    print(f"Config : {filename}")

    # Loading
    print("Loading data...")
    ds, ds_val = load_data(config=config)
    kernel = config.kernel

    # Training
    if performance:
        print("Estimating performances ...")
        scores = []
        for fold, (X, y, Xv, yv) in enumerate(ds):
            print(f"### Fold {fold+1}/{len(ds)} ###")
            score = train_and_score(kernel, X, y, Xv, yv, verbose=verbose)
            scores.append(score)

        # Summary
        print("### Summary ###")
        summary = get_summary(scores)
        print(summary)

        if not os.path.isdir(config.results_directory):
            os.mkdir(config.results_directory)
        with open(os.path.join(config.results_directory, filename + ".md"), "w") as f:
            f.write(summary)

    # Test
    if predict:
        test(config, ds, ds_val, kernel, filename)


def main(
    config_path: str,
    performance: bool = True,
    predict: bool = False,
    verbose: bool = True,
):
    """
    Trains and validates a Support Vector Classifier using the provided configuration file.

    Args:
        config_path (str): Path to Python module describing the configuration for the current run
        performance (bool, optional): Whether or not to estimate the performance of the model (default True)
        predict (bool, optional): Whether or not to predict outputs for validation data (default False)
        verbose (bool, optional): Whether or not to display progress bars during performance estimation or final prediction (default True)
    """

    filename, _ = os.path.splitext(os.path.basename(config_path))
    config_name = filename
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config
    spec.loader.exec_module(config)

    run(
        config=config,
        predict=predict,
        filename=filename,
        verbose=verbose,
        performance=performance,
    )


if __name__ == "__main__":
    fire.Fire(main)
