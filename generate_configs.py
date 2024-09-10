import copy
import json
from typing import Any

LAYER_2: dict[str, Any] = {
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 128,
    "size_name": "2_layer",
}

LAYER_3: dict[str, Any] = {
    "n_layer": 3,
    "n_head": 3,
    "n_embd": 192,
    "size_name": "small",
}

LAYER_4: dict[str, Any] = {
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 256,
    "size_name": "4_layer",
}

LAYER_6: dict[str, Any] = {
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "size_name": "6_layer",
}

LAYER_8: dict[str, Any] = {
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    "size_name": "8_layer",
}

LAYER_12: dict[str, Any] = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "size_name": "12_layer",
}

LAYER_16: dict[str, Any] = {
    "n_layer": 16,
    "n_head": 16,
    "n_embd": 1024,
    "size_name": "16_layer",
}

LAYER_24: dict[str, Any] = {
    "n_layer": 24,
    "n_head": 16,
    "n_embd": 1024,
    "size_name": "24_layer",
}

BASE_CONFIG: dict[str, Any] = {
    "task": "count3",
    "data_gen_seed_size": 16,
    "data_gen_seed_max": 64,  # block_size
    "max_iters": 150000,
    "lr_decay_iters": 100000,
    "warmup_iters": 500,
    "block_size": 64,
    "batch_size": 64,
    "test_batch_size": 256,
    "max_evals_without_improving": 50,
    "eval_interval": 100,
    "test_accuracy_during_training": True,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for size in [LAYER_3, LAYER_4]:
        for decoder in [False]:
            for seed in range(1):
                name = "count3"
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
