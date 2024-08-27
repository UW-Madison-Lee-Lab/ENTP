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
    "task": "superquadratic",
    "data_gen_seed_size": 16,
    "data_gen_seed_max": 64,
    "block_size": 64,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for decoder in [True, False]:
        for seed in range(1):
            name = "superquadratic_extra_small"
            name += "_decoder" if decoder else "_encoder"
            name += f"_{seed}"

            config = copy.deepcopy(BASE_CONFIG | EXTRA_SMALL)
            config["name"] = name
            config["decoder"] = decoder
            config["seed"] = seed

            config_path = f"configs/{name}.json"
            with open(config_path, "w") as f:
                json.dump(config, f)
