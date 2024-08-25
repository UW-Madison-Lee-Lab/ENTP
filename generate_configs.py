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
    "data_dir": "data/addition",
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
    # for decoder in [True]:
    #     for use_wpe in [True]:
    #         for permutation_invariant in [True, False]:
    #             for seed in range(1):
    #                 name = "counting_extra_small"
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

    for decoder in [False]:
        for use_wpe in [False]:
            for permutation_invariant in [False]:
                for seed in range(1):
                    name = "counting_large"
                    name += "_decoder" if decoder else "_encoder"
                    name += "" if use_wpe else "_nope"
                    name += (
                        "_perm_invariant" if permutation_invariant else "_perm_variant"
                    )
                    name += f"_{seed}"

                    config = copy.deepcopy(BASE_CONFIG | LARGE)
                    config["name"] = name
                    config["decoder"] = decoder
                    config["use_wpe"] = use_wpe
                    config["counting_permutation_invariant"] = permutation_invariant
                    config["seed"] = seed

                    config_path = f"configs/{name}.json"
                    with open(config_path, "w") as f:
                        json.dump(config, f)
