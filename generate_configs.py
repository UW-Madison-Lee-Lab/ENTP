import copy
import json
from typing import Any

EXTRA_SMALL: dict[str, Any] = {
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 128,
    "size_name": "extra_small",
}

SMALL: dict[str, Any] = {
    "n_layer": 3,
    "n_head": 3,
    "n_embd": 192,
    "size_name": "small",
}

MEDIUM: dict[str, Any] = {
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "size_name": "medium",
}

MEDIUM_DEEP: dict[str, Any] = {
    "n_layer": 12,
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
    "task": "reversed_addition_len_gen",
    "data_dir": "data/addition",
    "results_dir": "results/len_gen",
    "n_train": 100000,
    "n_val": 10000,
    "n_test": 10000,
    "n_digits": 10,
    "max_iters": 100000,
    "lr_decay_iters": 100000,
    "warmup_iters": 500,
    "block_size": 96,
    "batch_size": 16,
    "test_batch_size": 32,
    "eval_interval": 500,
    "test_accuracy_during_training": True,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for size in [MEDIUM_DEEP]:
        for decoder in [True, False]:
            for seed in range(1):
                name = "reversed_addition_len_gen_v2"
                name += f"_{size['size_name']}"
                name += "_decoder" if decoder else "_encoder"
                name += f"_{seed}"

                config = copy.deepcopy(BASE_CONFIG | size)
                config["name"] = name
                config["decoder"] = decoder
                config["seed"] = seed

                config_path = f"configs/{name}.json"
                with open(config_path, "w") as f:
                    json.dump(config, f)
