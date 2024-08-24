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
    "max_evals_without_improving": 10,
    "max_iters": 10000,
    "test_accuracy_during_training": True,
    "task": "counting",
    "counting_seed_max": 16,
    "counting_seed_size": 16,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for size in [MEDIUM]:
        for decoder in [True, False]:
            for use_wpe in [True, False]:
                for seed in range(1):
                    name = f"counting_{'small' if size == SMALL else 'medium'}_{'decoder' if decoder else 'encoder'}_{'' if use_wpe else 'nope_'}{seed}"
                    config = copy.deepcopy(BASE_CONFIG | size)
                    config["name"] = name
                    config["decoder"] = decoder
                    config["seed"] = seed

                    config_path = f"configs/{name}.json"
                    with open(config_path, "w") as f:
                        json.dump(config, f)
