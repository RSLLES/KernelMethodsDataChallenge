import re
import pandas as pd


def main(cols=["Method", "Accuracy", "Precision", "Recall", "F1", "ROCAUC"]):
    with open("./results/README.md", "r") as f:
        data = f.read()
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

    # Inserting
    text = results_df.to_markdown(index=False).split("\n")
    lines = data.splitlines()
    lines.insert(1, "## Ranking")
    for line in reversed(text):
        lines.insert(2, line)

    # Saving
    with open("./results/README.md", "w") as f:
        f.writelines("%s\n" % l for l in lines)


if __name__ == "__main__":
    main()
