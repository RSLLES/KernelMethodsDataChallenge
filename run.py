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
import multiprocessing


def print_metrics(fold, score_output):
    """Prints metrics produced by SVC.score function"""
    acc, prec, rec, f1, rocauc = score_output
    s = f"Fold {fold} -> Accuracy = {acc*100:.1f}%, Precision = {prec*100:.1f}%, Recall = {rec*100:.1f}%, F1 = {f1*100:.1f}%, ROCAUC = {rocauc*100:.1f}%"
    print(s)


def get_summary(scores):
    logs = [(f"Fold {fold+1}", *score) for fold, score in enumerate(scores)]

    scores = [np.array(score) for score in scores]
    average_scores = sum(scores) / len(scores)
    average_log = ("Average", *tuple(average_scores))

    cols = ["Accuracy", "Precision", "Recall", "F1", "ROCAUC"]
    df = pd.DataFrame([*logs, average_log], columns=[""] + cols)

    df[cols] = (df[cols] * 100).round(1).astype(str) + "%"
    return df.to_markdown(index=False)


def perf(h):
    try:
        fold, idx, K, y = h
        idx_train, idx_test = idx
        K_train = K[idx_train][:, idx_train]
        y_train = y[idx_train]
        K_test = K[idx_test][:, idx_train]
        y_test = y[idx_test]

        print(f"[{fold+1}] Solving ...")
        svc = SVC(kernel="precomputed", verbose=True)
        svc.fit(X=K_train, y=y_train)
        score = score_metric(svc=svc, X=K_test, y=y_test.astype(bool))
        print_metrics(fold + 1, score)
        return score
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        return None


def evaluate_perfs(ds, K, processes):
    k = ds.k_folds
    with multiprocessing.Pool(processes=processes) as p:
        print(f"Starting {k} SVC solvers on {processes} processes...")
        res = p.map_async(perf, zip(range(k), ds.iter_indexes(), [K] * k, [ds.y] * k))
        scores = list(res.get())
    return scores


def run(config, performance, predict, filename, verbose, processes):
    """
    Runs the entire pipeline for training and testing a classifier.
    """
    print(f"Config : {filename}")

    # Loading
    print("Loading data...")
    ds, ds_val = load_data(config=config)
    kernel = config.kernel
    kernel.set_processes(processes)

    # Computing kernel
    print("Computing kernel")
    K = kernel(ds.X)

    # Training
    if performance:
        scores = evaluate_perfs(ds=ds, K=K, processes=2)
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
        print("### On validation data ###")
        print("Training full SVC...")
        svc = SVC(kernel="precomputed")
        svc.fit(X=K, y=np.array(ds.y))
        print("Predicting values...")
        K_val = kernel(ds_val.X, ds.X)
        y_pred = svc.decision_function(K_val)
        print("Exporting...")
        if not os.path.isdir(config.export_directory):
            os.mkdir(config.export_directory)
        df = pd.DataFrame(
            y_pred,
            columns=["Predicted"],
            index=pd.RangeIndex(1, len(y_pred) + 1, name="Id"),
        )
        df.to_csv(os.path.join(config.export_directory, filename + ".csv"))


def main(
    config_path: str,
    performance: bool = True,
    predict: bool = False,
    verbose: bool = True,
    processes: int = None,
) -> None:
    """
    Trains and validates a Support Vector Classifier using the provided configuration file.

    Args:
        config_path (str): Path to Python module describing the configuration for the current run
        performance (bool, optional): Whether or not to estimate the performance of the model (default True)
        predict (bool, optional): Whether or not to predict outputs for validation data (default False)
        verbose (bool, optional): Whether or not to display progress bars during performance estimation or final prediction (default True)
        processes (int, optional): Number of processes to use for training. If set to None(default), it is set to max(1, your_cpu_cores - 4, your_cpu_cores //2). If set to an integer, program will use that many number of processes. If set to -1, processes is set to the maximum number of CPU available on the device. Note: Setting it to -1 on personal laptops may cause CPU overruns.

    Returns:
        None
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
        processes=processes,
    )


if __name__ == "__main__":
    fire.Fire(main)
