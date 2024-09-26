from collections import defaultdict
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"  # Use serif font
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage{amsmath}"  # Use additional packages if needed

FONTSIZE: int = 8


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
    save_path: Optional[str] = None,
    figsize=(8, 4),
    dpi=128,
) -> None:
    plt.figure(figsize=figsize, dpi=dpi)

    reduced = reduce_results(df, "accuracy_test", reduction)

    for k in [f"{task}_decoder", f"{task}_encoder"]:
        plt.plot(
            n_trains,
            reduced[k],
            marker="o",
            ms=4,
            label=k[-7:],
        )

    plt.ylabel(f"{reduction.__name__} test accuracy", fontsize=FONTSIZE)
    plt.ylabel("test accuracy", fontsize=FONTSIZE)
    plt.xlabel("training examples", fontsize=FONTSIZE)
    plt.xticks(fontsize=int(0.6667 * FONTSIZE))
    plt.yticks(fontsize=int(0.6667 * FONTSIZE))
    plt.title(title_prefix + task.replace("_", " "), fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_results_errorbar(
    df: pd.DataFrame,
    task: str,
    title_prefix="",
    title=True,
    n_trains=[1250, 2500, 3750, 5000, 10000, 15000, 20000],
    save_path: Optional[str] = None,
    figsize=(8, 4),
    dpi=128,
) -> None:
    plt.figure(figsize=figsize, dpi=dpi)

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
            label=k[-7:].capitalize() + "-only",
        )

    plt.ylabel("\\textbf{Test Accuracy}", fontsize=FONTSIZE)
    plt.xlabel("\\textbf{Number of Training Examples}", fontsize=FONTSIZE)
    plt.xticks(fontsize=int(0.6667 * FONTSIZE))
    plt.yticks(fontsize=int(0.6667 * FONTSIZE))
    if title:
        plt.title(title_prefix + task.replace("_", " "), fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

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
    save_path: Optional[str] = None,
    figsize=(8, 4),
    dpi=128,
) -> None:
    plt.figure(figsize=figsize, dpi=dpi)

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

    plt.ylabel(f"{reduction.__name__} test accuracy", fontsize=FONTSIZE)
    plt.xlabel("training examples", fontsize=FONTSIZE)
    plt.xticks(fontsize=int(0.6667 * FONTSIZE))
    plt.yticks(fontsize=int(0.6667 * FONTSIZE))
    plt.title(
        task.replace("_", " ") + (" decoder" if decoder else " encoder"),
        fontsize=FONTSIZE,
    )
    plt.legend(fontsize=FONTSIZE)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_multiple_keys(
    df: pd.DataFrame,
    x: str,
    ys: list[str],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    y_labels: Optional[list[Optional[str]]] = None,
    transform: Optional[Callable[[Any], Any]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize=(8, 4),
    dpi=128,
) -> None:
    plt.figure(figsize=figsize, dpi=dpi)

    if y_labels is None:
        y_labels = [None] * len(ys)
    else:
        assert len(ys) == len(y_labels)

    for y, label in zip(ys, y_labels):
        plt.plot(
            df[x],
            df[y] if transform is None else [transform(a) for a in df[y]],
            marker="o",
            ms=2,
            label=label,
        )

    if x_label is not None:
        plt.xlabel("\\textbf{" + x_label.capitalize() + "}", fontsize=FONTSIZE)

    if y_label is not None:
        plt.ylabel("\\textbf{" + y_label.capitalize() + "}", fontsize=FONTSIZE)

    if title is not None:
        plt.title("\\textbf{" + title.capitalize() + "}", fontsize=FONTSIZE)

    if y_labels[0] is not None:
        plt.legend(fontsize=FONTSIZE)

    plt.xticks(fontsize=int(0.6667 * FONTSIZE))
    plt.yticks(fontsize=int(0.6667 * FONTSIZE))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

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
    save_path: Optional[str] = None,
    figsize=(12, 3),
    dpi=128,
) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    if y_labels1 is None:
        y_labels1 = [None] * len(ys1)
    else:
        assert len(ys1) == len(y_labels1)

    for y, label in zip(ys1, y_labels1):
        ax1.plot(
            df[x1],
            df[y],
            marker="o",
            ms=2,
            label=label,
        )

    if x_label1 is not None:
        ax1.set_xlabel("\\textbf{" + x_label1.capitalize() + "}", fontsize=FONTSIZE)

    if title1 is not None:
        ax1.set_title("\\textbf{" + title1.capitalize() + "}", fontsize=FONTSIZE)

    if y_labels1[0] is not None:
        ax1.legend(fontsize=FONTSIZE)

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
            ms=2,
            label=label,
        )

    if x_label2 is not None:
        ax2.set_xlabel("\\textbf{" + x_label2.capitalize() + "}", fontsize=FONTSIZE)

    if title2 is not None:
        ax2.set_title("\\textbf{" + title2.capitalize() + "}", fontsize=FONTSIZE)

    if y_labels2[0] is not None:
        ax2.legend(fontsize=FONTSIZE)

    ax2.tick_params(axis="x", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
