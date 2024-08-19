import copy
import json

BASE_CONFIG: dict[str, bool | int | float | str] = {
    "batch_size": 64,
    "data_dir": "data/shakespeare",
    "max_evals_without_improving": 10,
    "max_iters": 16000,
    "max_loss_for_early_stopping": 1e9,
    "n_test": 70000,
    "n_val": 10000,
    "task": "shakespeare",
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for decoder in [True, False]:
        for small in [True, False]:
            for seed in range(5):
                name = f"shakespeare_{'small' if small else 'standare'}_{'decoder' if decoder else 'encoder'}_{seed}"
                config = copy.deepcopy(BASE_CONFIG)
                config["decoder"] = decoder
                config["seed"] = seed
                config["name"] = name

                if small:
                    config["n_embd"] = 192
                    config["n_head"] = 3
                    config["n_layer"] = 3

                config_path = f"configs/{name}.json"
                with open(config_path, "w") as f:
                    json.dump(config, f)
