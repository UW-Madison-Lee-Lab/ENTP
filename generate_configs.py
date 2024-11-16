import copy
import json
from typing import Any

EXTRA_SMALL: dict[str, Any] = {
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 128,
    "size_name": "extra_small",
}

EXTRA_SMALL_DEEP: dict[str, Any] = {
    "n_layer": 8,
    "n_head": 2,
    "n_embd": 128,
    "size_name": "extra_small_deep",
}

SMALL: dict[str, Any] = {
    "n_layer": 3,
    "n_head": 3,
    "n_embd": 192,
    "size_name": "small",
}

SMALL_DEEP: dict[str, Any] = {
    "n_layer": 12,
    "n_head": 3,
    "n_embd": 192,
    "size_name": "small_deep",
}

MEDIUM: dict[str, Any] = {
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "size_name": "medium",
}

MEDIUM_DEEP: dict[str, Any] = {
    "n_layer": 24,
    "n_head": 6,
    "n_embd": 384,
    "size_name": "medium_deep",
}

LARGE: dict[str, Any] = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "size_name": "large",
}

EXTRA_LARGE: dict[str, Any] = {
    "n_layer": 16,
    "n_head": 16,
    "n_embd": 1024,
    "size_name": "extra_large",
}

BASE_CONFIG: dict[str, Any] = {
    "task": "reversed_addition",
    "data_dir": "data/addition",
    "results_dir": "results",
    "n_embd": 128,
    "n_val": 10000,
    "n_test": 70000,
    "n_digits": 3,
    "max_iters": 2500,
    "lr_decay_iters": 2500,
    "warmup_iters": 100,
    "block_size": 64,
    "batch_size": 256,
    "test_batch_size": 2048,
    "eval_interval": 100,
    "use_delimiter": True,
    "test_accuracy_during_training": True,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for n_train in [1250, 2500, 3750, 5000, 10000, 15000, 20000]:
        for seed in range(1):
            name = BASE_CONFIG["task"]
            name += f"_mlp_{n_train}_{seed}"

            config = copy.deepcopy(BASE_CONFIG)
            config["name"] = name
            config["seed"] = seed
            config["n_train"] = n_train

            config_path = f"configs/{name}.json"
            with open(config_path, "w") as f:
                json.dump(config, f)
