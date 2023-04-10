import configs.wl_depth4 as config
from preprocessing.load import load_data
from kernels.GWWL import GeneralizedWassersteinWeisfeilerLehmanKernel as Kernel
from run import evaluate_perfs
import numpy as np
import pandas as pd
import os

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def test(depth, lambd, weight):
    print(f"Params for this run : depth={depth}, lambd={lambd}, weight={weight}")
    ds, _ = load_data(config=config)

    kernel = Kernel(
        depth=int(depth),
        lambd=lambd,
        weight=weight,
        use_cache=True,
    )
    kernel.set_processes(None)
    print(f"Processes : {kernel.processes}")

    # Computing kernel
    print("Computing kernel")
    K = kernel(ds.X)

    # Training
    scores = evaluate_perfs(ds=ds, K=K, processes=kernel.processes)

    scores = [np.array(score) for score in scores]
    average_scores = sum(scores) / len(scores)
    return average_scores[-1]


def main():
    path = f"./optimization/{Kernel.__name__}"
    if not os.path.isdir(path):
        os.mkdir(path)

    json_path = os.path.join(path, "logs.json")
    results_path = os.path.join(path, "results.csv")

    pbounds = {
        "depth": (3, 9),
        "lambd": (0.5, 5),
        "weight": (0.0, 0.5),
    }

    optimizer = BayesianOptimization(
        f=test,
        pbounds=pbounds,
        random_state=2,
    )

    if os.path.isfile(json_path):
        print(f"Loading logs at {json_path}")
        load_logs(optimizer, logs=json_path)
        print("Optimizer is now aware of {} points.".format(len(optimizer.space)))

    logger = JSONLogger(path=json_path, reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.set_gp_params(alpha=1e-3)
<<<<<<< HEAD
    optimizer.probe(params=[4, 3.0, 0.0])
=======
    optimizer.probe(params=[5, 3.0, 0.])
>>>>>>> 4d7ef36 ([~] Final result optimization GWWL)
    if len(optimizer.space.target) > 0:
        optimizer._gp.fit(optimizer.space.params, optimizer.space.target)
    optimizer.maximize(n_iter=60, init_points=15)

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
