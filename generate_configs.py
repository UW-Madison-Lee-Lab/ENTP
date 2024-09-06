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
    "task": "memory_bound",
    "max_iters": 100000,
    "lr_decay_iters": 100000,
    "warmup_iters": 500,
    "min_lr": 1e-4,
    "max_lr": 1e-3,
    "weight_decay": 0.01,
    "block_size": 256,
    "batch_size": 32,
    "max_evals_without_improving": 100,
    "eval_interval": 100,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for size in [EXTRA_SMALL]:
        for decoder in [True, False]:
            for seed in range(1):
                name = "memory_bound"
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

    # for decoder in [True, False]:
    #     for use_wpe in [True, False]:
    #         for permutation_invariant in [True, False]:
    #             for seed in range(1):
    #                 name = "counting_extra_small_rt"
    #                 name += "_decoder" if decoder else "_encoder"
    #                 name += "" if use_wpe else "_nope"
    #                 name += (
    #                     "_perm_invariant" if permutation_invariant else "_perm_variant"
    #                 )
    #                 name += f"_{seed}"

    #                 config = copy.deepcopy(BASE_CONFIG | EXTRA_SMALL)
    #                 config["name"] = name
    #                 config["decoder"] = decoder
    #                 config["use_wpe"] = use_wpe
    #                 config["counting_permutation_invariant"] = permutation_invariant
    #                 config["seed"] = seed

    #                 config_path = f"configs/{name}.json"
    #                 with open(config_path, "w") as f:
    #                     json.dump(config, f)
