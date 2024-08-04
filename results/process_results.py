import json
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python process_results.py results_dir")
        exit(1)

    results_dir = sys.argv[1]

    # results: dict[str, dict[str, list[float]]] = {}

    # for model in ["decoder", "encoder"]:
    #     results[model] = {}
    #     for task in ["plain_addition", "reversed_addition"]:
    #         results[model][task] = []
    #         prefix = f"{task}_{model}"
    #         n_seeds = len([f for f in os.listdir(results_dir) if prefix in f])
    #         for seed in range(n_seeds):
    #             path = os.path.join(results_dir, f"{task}_{model}_{seed}_results.txt")
    #             with open(path, "r") as f:
    #                 text = f.read()

    #             line = text.split("\n")[0]
    #             n_correct = int(line[: line.find("/")])
    #             n_total = int(line[line.find("/") + 1 :])
    #             results[model][task].append(n_correct / n_total)

    # results_table: dict[str, list[Any]] = defaultdict(list)

    # for model in ["decoder", "encoder"]:
    #     for task in ["plain_addition", "reversed_addition"]:
    #         results_table["experiment"].append(f"{task}_{model}")
    #         results_table["accuracy_mean"].append(np.mean(results[model][task]))
    #         results_table["accuracy_std"].append(np.std(results[model][task]))
    #         results_table["n_trials"].append(len(results[model][task]))

    results: dict[str, dict[str, list[float]]] = {}

    for use_dollar_signs in ["true", "false"]:
        results[use_dollar_signs] = {}
        for task in ["plain_addition", "reversed_addition"]:
            results[use_dollar_signs][task] = []
            prefix = f"{task}_dollar_signs_{use_dollar_signs}"
            n_seeds = len([f for f in os.listdir(results_dir) if prefix in f])
            for seed in range(n_seeds):
                path = os.path.join(
                    results_dir,
                    f"{task}_dollar_signs_{use_dollar_signs}_{seed}_results.txt",
                )
                with open(path, "r") as f:
                    text = f.read()

                line = text.split("\n")[0]
                n_correct = int(line[: line.find("/")])
                n_total = int(line[line.find("/") + 1 :])
                results[use_dollar_signs][task].append(n_correct / n_total)

    results_table: dict[str, list[Any]] = defaultdict(list)

    for use_dollar_signs in ["true", "false"]:
        for task in ["plain_addition", "reversed_addition"]:
            results_table["experiment"].append(
                f"{task}_dollar_signs_{use_dollar_signs}"
            )
            results_table["accuracy_mean"].append(
                np.mean(results[use_dollar_signs][task])
            )
            results_table["accuracy_std"].append(
                np.std(results[use_dollar_signs][task])
            )
            results_table["n_trials"].append(len(results[use_dollar_signs][task]))

    path = os.path.join(results_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results_table, f)
