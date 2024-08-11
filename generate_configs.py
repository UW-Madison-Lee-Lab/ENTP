import copy
import json

BASE_CONFIG: dict[str, bool | int | float | str] = {
    "max_iters": 8000,
    "n_test": 70000,
    "n_val": 10000,
}

if __name__ == "__main__":
    for n_train in [10000]:
        for decoder in [True]:
            for task in ["plain_addition"]:
                for seed in range(0, 5, 2):
                    name = f"{n_train // 1000}k_{task}_{'decoder' if decoder else 'encoder'}_{seed}"
                    config = copy.deepcopy(BASE_CONFIG)
                    config["n_train"] = n_train
                    config["decoder"] = decoder
                    config["task"] = task
                    config["seed"] = seed
                    config["name"] = name
                    config["results_dir"] = f"results/{n_train // 1000}k"

                    if n_train == 5000:
                        config["max_loss_for_early_stopping"] = 1e9

                    config_path = f"configs/{name}.json"
                    with open(config_path, "w") as f:
                        json.dump(config, f)

    for n_train in [20000]:
        for decoder in [True, False]:
            for task in ["reversed_addition"]:
                for seed in range(5, 10):
                    name = f"{n_train // 1000}k_{task}_{'decoder' if decoder else 'encoder'}_{seed}"
                    config = copy.deepcopy(BASE_CONFIG)
                    config["n_train"] = n_train
                    config["decoder"] = decoder
                    config["task"] = task
                    config["seed"] = seed
                    config["name"] = name
                    config["results_dir"] = f"results/{n_train // 1000}k"

                    if n_train == 5000:
                        config["max_loss_for_early_stopping"] = 1e9

                    config_path = f"configs/{name}.json"
                    with open(config_path, "w") as f:
                        json.dump(config, f)
