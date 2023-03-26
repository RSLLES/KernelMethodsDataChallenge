import configs.wwl_edges_depth1 as config
from preprocessing.load import load_data
from kernels.WWL import WassersteinWeisfeilerLehmanKernel
from run import train_and_score
import numpy as np
import pandas as pd
import os

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


def test(depth, lambd):
    ds, ds_val = load_data(config=config)

    kernel = WassersteinWeisfeilerLehmanKernel(
        depth=int(depth), enable_edges_labels=True, lambd=lambd, use_cache=True
    )
    kernel.set_processes(-1)

    # Training
    scores = []
    for fold, (X, y, Xv, yv) in enumerate(ds):
        print(f"### Fold {fold+1}/{len(ds)} ###")
        score = train_and_score(kernel, X, y, Xv, yv, verbose=True)
        scores.append(score)

    scores = [np.array(score) for score in scores]
    average_scores = sum(scores) / len(scores)
    f1, auc = average_scores[-2], average_scores[-1]
    return auc + 0.01 * f1


def black_box_function(x, y, d):
    d = int(d)
    return ((x + y + d) // (1 + d)) / (1 + (x + y) ** 2)


def main():
    if not os.path.isdir("./optimization/"):
        os.mkdir("./optimization/")

    logger = JSONLogger(path="./optimization/logs.json")

    pbounds = {
        "depth": (1, 8),
        "lambd": (0.5, 10.0),
    }

    optimizer = BayesianOptimization(
        f=test,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
    )

    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.set_gp_params(alpha=1e-3)
    optimizer.maximize(n_iter=30)
    import pandas as pd

    df = pd.DataFrame(list(optimizer.res))
    df.index.name = "Iteration"
    df.columns.name = "Result"
    for param in optimizer.max["params"]:
        df[param] = df["params"].apply(lambda x: x[param])
    df = df.drop("params", axis=1)
    df = df.sort_values(["target"], ascending=False)
    print(df)
    df.to_csv("./optimization/optimization_result.csv")


if __name__ == "__main__":
    main()
