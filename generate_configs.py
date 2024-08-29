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
    "max_iters": 50000,
    "lr_decay_iters": 50000,
    "max_evals_without_improving": 20,
    "test_accuracy_during_training": True,
    "task": "superquadratic",
    "data_gen_seed_size": 16,
    "data_gen_seed_max": 64,
    "block_size": 64,
    "test_batch_size": 256,
}


def n_train_str(n_train: int) -> str:
    if n_train % 1000 == 0:
        return f"{n_train // 1000}k"
    else:
        return f"{n_train / 1000}k"


if __name__ == "__main__":
    for decoder in [True]:
        for seed in range(1):
            name = "superquadratic_medium"
            name += "_decoder" if decoder else "_encoder"
            name += f"_{seed}"

            config = copy.deepcopy(BASE_CONFIG | MEDIUM)
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
