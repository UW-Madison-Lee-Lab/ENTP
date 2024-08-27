import copy
import json

EXTRA_SMALL: dict[str, int] = {
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 128,
}

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
    "batch_size": 32,
}

EXTRA_LARGE: dict[str, int] = {
    "n_layer": 16,
    "n_head": 16,
    "n_embd": 1024,
    "batch_size": 32,
}

BASE_CONFIG: dict[str, bool | int | float | str] = {
    "max_iters": 5000,
    "test_accuracy_during_training": True,
    "task": "ortho_vec",
    "block_size": 32,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for decoder in [True]:
        for seed in range(1):
            name = "ortho_vec_extra_small"
            name += "_decoder" if decoder else "_encoder"
            name += f"_{seed}"

            config = copy.deepcopy(BASE_CONFIG | EXTRA_SMALL)
            config["name"] = name
            config["decoder"] = decoder
            config["seed"] = seed

            config_path = f"configs/{name}.json"
            with open(config_path, "w") as f:
                json.dump(config, f)
