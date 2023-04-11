import configs.wl_depth4 as config
from preprocessing.load import load_data
from kernels.GWWL import GeneralizedWassersteinWeisfeilerLehmanKernelImport as Kernel
from run import evaluate_perfs
import numpy as np
import pandas as pd
import os

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def test(log_lambd):
    try:
        print(f"Params for this run : log_lambd={log_lambd}")
        ds, _ = load_data(config=config)

        kernel = Kernel(
            root="../cost_matrices/",
            lambd=np.power(10, log_lambd),
            use_cache=True,
        )
        kernel.set_processes(None)
        print(f"Processes : {kernel.processes}")

        # Computing kernel
        print("Computing kernel")
        K = kernel(ds.X)

        # Training
        scores = evaluate_perfs(ds=ds, K=K, processes=kernel.processes // 2)

        scores = [np.array(score) for score in scores]
        average_scores = sum(scores) / len(scores)
        print(f"Score = {100*average_scores[-1]:0.3}%")
        return average_scores[-1]
    except:
        return 0.85


def main():
    path = f"./optimization/{Kernel.__name__}"
    if not os.path.isdir(path):
        os.mkdir(path)

    json_path = os.path.join(path, "logs.json")

    pbounds = {
        "log_lambd": (-2.0, 0.0),
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
    # optimizer.set_gp_params(alpha=1e-3)
    if len(optimizer.space.target) > 0:
        optimizer._gp.fit(optimizer.space.params, optimizer.space.target)
    optimizer.maximize(n_iter=25, init_points=7)

    df = pd.DataFrame(list(optimizer.res))
    df.index.name = "Iteration"
    df.columns.name = "Result"
    for param in optimizer.max["params"]:
        df[param] = df["params"].apply(lambda x: x[param])
    df = df.drop("params", axis=1)
    df = df.sort_values(["target"], ascending=False)

    print(df)


if __name__ == "__main__":
    main()
