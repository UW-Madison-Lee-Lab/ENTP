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
    "task": "clutrr",
    "warmup_iters": 50,
    "max_iters": 10000,
    "lr_decay_iters": 10000, 
    "max_lr": 1e-5,
    "min_lr": 1e-6,
    "beta1": 0.9, 
    "beta2": 0.95, 
    "weight_decay": 0.1,
    "block_size": 128,
    "batch_size": 64,
    "test_batch_size": 256,
    "eval_interval": 50, 
    "n_layer": 6, 
    "n_head": 6, 
    "n_embd": 384, 
    "seed": 0
}

# BASE_CONFIG: dict[str, Any] = {
#     "task": "ner",
#     "warmup_iters": 50,
#     "lr_decay_iters": 15000, 
#     "block_size": 64,
#     "batch_size": 64,
#     "test_batch_size": 1024,
#     "eval_interval": 50, 
#     "seed": 0
# }

def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for decoder in [True, False]:
        for n_train in [1250, 2500, 5000, 7500]:  # full train set is 11000
            name = BASE_CONFIG["task"]
            name += f"_finetuning_{n_train_str(n_train)}_{'decoder' if decoder else 'encoder'}"

            config = copy.deepcopy(BASE_CONFIG)
            config["name"] = name
            config["decoder"] = decoder
            config["n_train"] = n_train
            config["max_iters"] = n_train

            config_path = f"configs/{name}.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

    # for size in [SMALL]:
    #     for n_train in [250, 500, 1000, 2000, 4000]:
    #         for decoder in [True, False]:
    #             name = BASE_CONFIG["task"]
    #             name += f"_{n_train_str(n_train)}_{size["size_name"]}_{'decoder' if decoder else 'encoder'}"

    #             config = copy.deepcopy(BASE_CONFIG | size)
    #             config["name"] = name
    #             config["n_train"] = n_train
    #             config["decoder"] = decoder
    #             config["max_iters"] = 4 * n_train

    #             config_path = f"configs/{name}.json"
    #             with open(config_path, "w") as f:
    #                 json.dump(config, f)
