import os

import pandas as pd  # type: ignore


def merge_wandb_csvs(dir: str) -> pd.DataFrame:
    csvs = []
    for f in os.listdir(dir):
        if f[-4:] == ".csv":
            csvs.append(os.path.join(dir, f))
    df = pd.DataFrame({"Step": []})

    for csv in csvs:
        csv_df = pd.read_csv(csv)

        main_k = ""
        for k in csv_df.keys():
            if "__MIN" not in k and "__MAX" not in k and k != "Step":
                main_k = k

        assert " - " in main_k
        csv_df = csv_df[["Step", main_k]]
        csv_df = csv_df[csv_df["Step"] % 100 == 0]

        df = df.merge(csv_df, on="Step", how="outer")

    return df
