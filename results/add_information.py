import numpy as np
import re
import pandas as pd
import seaborn as sns


def ranking(data, cols=["Method", "Accuracy", "Precision", "Recall", "F1", "ROCAUC"]):
    tables = re.findall(r"##\s*(.+?)\n\|(?:.*?\|)+\n((?:\|.*?\|\n)+)", data)

    results_df = []
    for title, table in tables:
        # replace whitespace characters and split the table into rows
        table = table.replace("\n ", "").split("\n")

        # extract the column names and their indices

        # extract the values from the table
        row_values = None
        for last_row in reversed(table):
            if last_row.strip() != "":
                row_values = [
                    v.strip().replace("%", "") for v in last_row.split("|")[1:-1]
                ]
            if row_values is not None:
                break
        row_values[0] = title
        results_df.append(row_values)

    # sort the dataframe by average ROCAUC and average F1 scores in descending order
    results_df = pd.DataFrame(results_df, columns=cols)
    results_df = results_df.sort_values(by=["ROCAUC", "F1"], ascending=False)
    return results_df


def corr_matrix(results_df, path="./results/corr_matrix.png"):
    def import_results(method_name):
        df = pd.read_csv(f"./export/{method_name}.csv")
        return df["Predicted"].to_numpy()

    methods = list(results_df["Method"])
    # methods.sort()
    values = np.array([import_results(method) for method in methods]).T
    df_values = pd.DataFrame(values, columns=methods)
    corr = df_values.corr()

    plot = sns.heatmap(corr, cmap="mako")
    fig = plot.get_figure()
    # fig.subplots_adjust(top=0.95)
    fig.savefig(path, bbox_inches="tight")


def main():
    with open("./results/README.md", "r") as f:
        data = f.read()
    # Computing
    results_df = ranking(data)
    corr_matrix(results_df)

    # Creating text
    text = ["## Ranking"] + results_df.to_markdown(index=False).split("\n")
    text += [
        "## Correlation between methods",
        f"![Correlation matrix](./corr_matrix.png)",
    ]

    # Inserting
    lines = data.splitlines()
    for line in reversed(text):
        lines.insert(1, line)

    # Saving
    with open("./results/README.md", "w") as f:
        f.writelines("%s\n" % l for l in lines)


if __name__ == "__main__":
    main()
