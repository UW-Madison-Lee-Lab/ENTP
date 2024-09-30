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
        for k in csv_df.keys():
            if k != "Step":
                key_df = csv_df[["Step", k]].copy()
                key_df["Step"] //= step_size
                key_df = (
                    key_df.groupby("Step", as_index=False).mean().reset_index(drop=True)
                )
                key_df["Step"] *= step_size
                df = df.merge(key_df, on="Step", how="outer")

    return df


def combine_columns(df, x, y):
    df[x] = df.apply(
        lambda row: min(row[x], row[y])
        if pd.notna(row[x]) and pd.notna(row[y])
        else row[x]
        if pd.notna(row[x])
        else row[y],
        axis=1,
    )
