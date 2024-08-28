import os

import pandas as pd  # type: ignore


def merge_wandb_csvs(dir: str, step_size=100) -> pd.DataFrame:
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

        csv_df_copy = csv_df.copy()

        csv_df["Step"] //= step_size
        csv_df = csv_df.groupby("Step", as_index=False).mean().reset_index(drop=True)
        csv_df["Step"] *= step_size

        if "train_loss" not in main_k and step_size == 100:
            assert (csv_df["Step"] == csv_df_copy["Step"]).all()
            assert (csv_df[main_k] == csv_df_copy[main_k]).all()

        df = df.merge(csv_df, on="Step", how="outer")

    return df
