import copy
import json

SMALL: dict[str, int] = {
    "n_layer": 3,
    "n_head": 3,
    "n_embd": 192,
}

MEDIUM: dict[str, int] = {
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
}

LARGE: dict[str, int] = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
}

BASE_CONFIG: dict[str, bool | int | float | str] = {
    "data_dir": "data/addition",
    "max_evals_without_improving": 25,
    "max_iters": 10000,
    "n_train": 2500,
    "n_val": 10000,
    "n_test": 70000,
    "test_accuracy_during_training": True,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for size in [SMALL, MEDIUM]:
        for decoder in [True, False]:
            for task in ["plain_addition", "reversed_addition"]:
                for seed in range(1):
                    name = f"{task}_{'small' if size == SMALL else 'medium'}_{'decoder' if decoder else 'encoder'}_{seed}"
                    config = copy.deepcopy(BASE_CONFIG | size)
                    config["decoder"] = decoder
                    config["task"] = task
                    config["seed"] = seed
                    config["name"] = name
                    config["results_dir"] = f"results/{n_train_str(config["n_train"])}"

                    if config["n_train"] <= 5000:
                        config["max_loss_for_early_stopping"] = 1e9

                    config_path = f"configs/{name}.json"
                    with open(config_path, "w") as f:
                        json.dump(config, f)

    # for size in [SMALL, MEDIUM]:
    #     for decoder in [True, False]:
    #         for seed in range(1):
    #             name = f"shakespeare_{'small' if size == SMALL else 'medium'}_{'decoder' if decoder else 'encoder'}_no_wpe_{seed}"
    #             config = copy.deepcopy(BASE_CONFIG | size)
    #             config["decoder"] = decoder
    #             config["seed"] = seed
    #             config["name"] = name

    #             config_path = f"configs/{name}.json"
    #             with open(config_path, "w") as f:
    #                 json.dump(config, f)

