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

    results: dict[str, dict[str, list[float]]] = {}

    for model in ["decoder", "encoder"]:
        results[model] = {}
        for task in ["plain_addition", "reversed_addition"]:
            results[model][task] = []
            prefix = f"{task}_{model}"
            n_seeds = len([f for f in os.listdir(results_dir) if prefix in f])
            for seed in range(n_seeds):
                path = os.path.join(results_dir, f"{results_dir}_{task}_{model}_{seed}_results.txt")
                with open(path, "r") as f:
                    text = f.read()

                line = text.split("\n")[0]
                n_correct = int(line[: line.find("/")])
                n_total = int(line[line.find("/") + 1 :])
                results[model][task].append(n_correct / n_total)

    results_table: dict[str, list[Any]] = defaultdict(list)

    for model in ["decoder", "encoder"]:
        for task in ["plain_addition", "reversed_addition"]:
            results_table["experiment"].append(f"{task}_{model}")
            results_table["accuracy_mean"].append(np.mean(results[model][task]))
            results_table["accuracy_std"].append(np.std(results[model][task]))
            results_table["accuracy_median"].append(np.median(results[model][task]))
            results_table["n_trials"].append(len(results[model][task]))
            results_table["trial_accuracies"].append(results[model][task])

    path = os.path.join(results_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results_table, f)
