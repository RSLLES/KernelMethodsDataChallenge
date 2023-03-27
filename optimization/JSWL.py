import configs.wl_depth4 as config
from preprocessing.load import load_data
from kernels.JSWL import JensenShannonWeisfeilerLehmanKernel as Kernel
from run import train_and_score
import numpy as np
import pandas as pd
import os

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def test(depth, log_lambd):
    print(f"Computing for depth={depth}, log_lambd = {log_lambd}")
    ds, ds_val = load_data(config=config)

    kernel = Kernel(
        depth=int(depth),
        enable_edges_labels=True,
        lambd=np.power(10, log_lambd),
        use_cache=True,
    )
    kernel.set_processes(-1)
    print(f"Processes : {kernel.processes}")

    # Training
    scores = []
    for fold, (X, y, Xv, yv) in enumerate(ds):
        print(f"### Fold {fold+1}/{len(ds)} ###")
        score = train_and_score(kernel, X, y, Xv, yv, verbose=False)
        scores.append(score)

    scores = [np.array(score) for score in scores]
    average_scores = sum(scores) / len(scores)
    f1, auc = average_scores[-2], average_scores[-1]
    return auc


def main():
    path = f"./optimization/{Kernel.__name__}"
    if not os.path.isdir(path):
        os.mkdir(path)

    json_path = os.path.join(path, "logs.json")
    results_path = os.path.join(path, "results.csv")

    pbounds = {
        "depth": (1, 9),
        "log_lambd": (-1.0, 1.0),
    }

    optimizer = BayesianOptimization(
        f=test,
        pbounds=pbounds,
        random_state=1,
    )

    if os.path.isfile(json_path):
        print(f"Loading logs at {json_path}")
        load_logs(optimizer, logs=json_path)
        print("Optimizer is now aware of {} points.".format(len(optimizer.space)))

    logger = JSONLogger(path=json_path, reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.set_gp_params(alpha=1e-3)
    # optimizer.probe(
    #     params=[3.5, 0.0],
    # )
    optimizer._gp.fit(optimizer.space.params, optimizer.space.target)
    optimizer.maximize(n_iter=40, init_points=15)

    df = pd.DataFrame(list(optimizer.res))
    df.index.name = "Iteration"
    df.columns.name = "Result"
    for param in optimizer.max["params"]:
        df[param] = df["params"].apply(lambda x: x[param])
    df = df.drop("params", axis=1)
    df = df.sort_values(["target"], ascending=False)

    print(df)
    df.to_csv(results_path)


if __name__ == "__main__":
    main()
