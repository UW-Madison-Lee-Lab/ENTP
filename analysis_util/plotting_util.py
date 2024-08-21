from collections import defaultdict
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore


def reduce_results(
    df: pd.DataFrame,
    key: str,
    reduction: Callable,
    n_trains=[1250, 2500, 3750, 5000, 10000, 15000, 20000],
) -> dict:
    results = defaultdict(list)

    for task in df["task"].unique():
        for decoder in [True, False]:
            for n_train in n_trains:
                subset = df[
                    (df["task"] == task)
                    & (df["n_train"] == n_train)
                    & (df["decoder"] == decoder)
                ]
                if len(subset) > 0:
                    results_key = f"{task}_{'decoder' if decoder else 'encoder'}"
                    results[results_key].append(reduction(subset[key].to_numpy()))

    return results


def plot_results(
    df: pd.DataFrame,
    task: str,
    reduction: Callable,
    title_prefix="",
    n_trains=[1250, 2500, 3750, 5000, 10000, 15000, 20000],
) -> None:
    plt.figure(figsize=(8, 4), dpi=128)

    reduced = reduce_results(df, "accuracy_test", reduction)

    for k in [f"{task}_decoder", f"{task}_encoder"]:
        plt.plot(
            n_trains,
            reduced[k],
            marker="o",
            ms=4,
            label=k[-7:],
        )

    plt.ylabel(f"{reduction.__name__} test accuracy", fontsize=8)
    plt.ylabel("test accuracy", fontsize=8)
    plt.xlabel("training examples", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title_prefix + task.replace("_", " "), fontsize=8)
    plt.legend(fontsize=8)
    plt.show()


def plot_results_errorbar(
    df: pd.DataFrame,
    task: str,
    title_prefix="",
    n_trains=[1250, 2500, 3750, 5000, 10000, 15000, 20000],
) -> None:
    plt.figure(figsize=(8, 4), dpi=128)

    means = reduce_results(df, "accuracy_test", np.mean)
    stds = reduce_results(df, "accuracy_test", np.std)

    for k in [f"{task}_decoder", f"{task}_encoder"]:
        plt.errorbar(
            n_trains,
            means[k],
            yerr=stds[k],
            fmt="-o",
            ms=4,
            capsize=4,
            label=k[-7:],
        )

    plt.ylabel("test accuracy", fontsize=8)
    plt.xlabel("training examples", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title_prefix + task.replace("_", " "), fontsize=8)
    plt.legend(fontsize=8)
    plt.show()


def plot_results_scatter(
    df: pd.DataFrame,
    task: str,
    title_prefix="",
    n_trains=[1250, 2500, 3750, 5000, 10000, 15000, 20000],
) -> None:
    x_decoder: list[int] = []
    x_encoder: list[int] = []
    y_decoder: list[float] = []
    y_encoder: list[float] = []

    for decoder, x, y in [(True, x_decoder, y_decoder), (False, x_encoder, y_encoder)]:
        for n_train in n_trains:
            df_subset = df[(df["decoder"] == decoder) & (df["n_train"] == n_train)]
            for acc in df_subset["accuracy_test"]:
                x.append(n_train)
                y.append(acc)

    plt.scatter(x_decoder, y_decoder, s=5, alpha=0.5, label="decoder")
    plt.scatter(x_encoder, y_encoder, s=5, alpha=0.5, label="encoder")
    plt.ylabel("test accuracy")
    plt.xlabel("training examples")
    plt.title(title_prefix + task.replace("_", " "))
    plt.legend()
    plt.show()


def plot_results_bar(
    df: pd.DataFrame,
    task: str,
    n_trains=[1250, 2500, 3750, 5000, 10000, 15000, 20000],
) -> None:
    _, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, (ax, n_train) in enumerate(zip(axes.reshape(-1), n_trains)):
        keys = [
            "accuracy_2d_0c",
            "accuracy_2d_1c",
            "accuracy_2d_2c",
            "accuracy_3d_0c",
            "accuracy_3d_1c",
            "accuracy_3d_2c",
            "accuracy_3d_3c",
        ]

        data = [reduce_results(df, k, np.median) for k in keys]
        x_labels = [k[-5:].replace("_", "-") for k in keys]

        y_decoder = [d[f"{task}_decoder"][i] for d in data]
        y_encoder = [d[f"{task}_encoder"][i] for d in data]

        x = np.arange(len(x_labels))

        bar_width = 0.35

        ax.bar(x - bar_width / 2, y_decoder, width=bar_width, label="decoder")
        ax.bar(x + bar_width / 2, y_encoder, width=bar_width, label="encoder")

        ax.set_title(
            task.replace("_", " ") + f" {n_train} training examples", fontsize=8
        )
        ax.set_xticks(x, x_labels)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.show()


def plot_results_from_two_df(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    task: str,
    reduction: Callable,
    decoder: bool,
    label1="",
    label2="",
    n_trains=[1250, 2500, 3750, 5000, 10000, 15000, 20000],
) -> None:
    plt.figure(figsize=(8, 4), dpi=128)

    reduced1 = reduce_results(df1, "accuracy_test", reduction)
    reduced2 = reduce_results(df2, "accuracy_test", reduction)

    plt.plot(
        n_trains,
        reduced1[task + ("_decoder" if decoder else "_encoder")],
        marker="o",
        ms=4,
        label=label1,
    )

    plt.plot(
        n_trains,
        reduced2[task + ("_decoder" if decoder else "_encoder")],
        marker="o",
        ms=4,
        label=label2,
    )

    plt.ylabel(f"{reduction.__name__} test accuracy", fontsize=8)
    plt.xlabel("training examples", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(
        task.replace("_", " ") + (" decoder" if decoder else " encoder"), fontsize=8
    )
    plt.legend(fontsize=8)
    plt.show()


def plot_multiple_keys(
    df: pd.DataFrame,
    x: str,
    ys: list[str],
    x_label: Optional[str] = None,
    y_labels: Optional[list[Optional[str]]] = None,
    title: Optional[str] = None,
) -> None:
    plt.figure(figsize=(8, 4), dpi=128)

    if y_labels is None:
        y_labels = [None] * len(ys)
    else:
        assert len(ys) == len(y_labels)

    for y, label in zip(ys, y_labels):
        plt.plot(
            df[x],
            df[y],
            marker="o",
            ms=4,
            label=label,
        )

    if x_label is not None:
        plt.xlabel(x_label, fontsize=8)

    if title is not None:
        plt.title(title, fontsize=8)

    if y_labels[0] is not None:
        plt.legend(fontsize=8)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()


def plot_two_sets_of_multiple_keys(
    df: pd.DataFrame,
    x1: str,
    ys1: list[str],
    x2: str,
    ys2: list[str],
    x_label1: Optional[str] = None,
    y_labels1: Optional[list[Optional[str]]] = None,
    title1: Optional[str] = None,
    x_label2: Optional[str] = None,
    y_labels2: Optional[list[Optional[str]]] = None,
    title2: Optional[str] = None,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), dpi=128)

    if y_labels1 is None:
        y_labels1 = [None] * len(ys1)
    else:
        assert len(ys1) == len(y_labels1)

    for y, label in zip(ys1, y_labels1):
        ax1.plot(
            df[x1],
            df[y],
            marker="o",
            ms=4,
            label=label,
        )

    if x_label1 is not None:
        ax1.set_xlabel(x_label1, fontsize=8)

    if title1 is not None:
        ax1.set_title(title1, fontsize=8)

    if y_labels1[0] is not None:
        ax1.legend(fontsize=8)

    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    if y_labels2 is None:
        y_labels2 = [None] * len(ys2)
    else:
        assert len(ys2) == len(y_labels2)

    for y, label in zip(ys2, y_labels2):
        ax2.plot(
            df[x2],
            df[y],
            marker="o",
            ms=4,
            label=label,
        )

    if x_label2 is not None:
        ax2.set_xlabel(x_label2, fontsize=8)

    if title2 is not None:
        ax2.set_title(title2, fontsize=8)

    if y_labels2[0] is not None:
        ax2.legend(fontsize=8)

    ax2.tick_params(axis="x", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.show()
