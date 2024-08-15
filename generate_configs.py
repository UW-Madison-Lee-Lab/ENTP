import copy
import json

BASE_CONFIG: dict[str, bool | int | float | str] = {
    "max_iters": 8000,
    "n_test": 70000,
    "n_val": 10000,
    "max_evals_without_improving": 5,
    "batch_size": 32,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for n_train in [3750]:
        for decoder in [False]:
            for task in ["reversed_addition"]:
                for seed in range(3, 5):
                    name = f"{n_train_str(n_train)}_{task}_{'decoder' if decoder else 'encoder'}_{seed}"
                    config = copy.deepcopy(BASE_CONFIG)
                    config["n_train"] = n_train
                    config["decoder"] = decoder
                    config["task"] = task
                    config["seed"] = seed
                    config["name"] = name
                    config["results_dir"] = f"results/{n_train_str(n_train)}"

                    if n_train <= 5000:
                        config["max_loss_for_early_stopping"] = 1e9

                    config_path = f"configs/{name}.json"
                    with open(config_path, "w") as f:
                        json.dump(config, f)
