import copy

from evaluate import evaluate
from generate_addition_data import generate_addition_data
from train import train
from util import Config, Environment

BASE_CONFIG_DICT: dict[str, bool | int | float | str] = {
    "max_iters": 8000,
    # "n_test": 75000,
    # "n_train": 15000,
    # "n_val": 10000,
    # "results_dir": "results/15k",
    "warmup_iters": 150,
}


if __name__ == "__main__":

    # 20k experiment
    base_config_dict_20k = copy.deepcopy(BASE_CONFIG_DICT)
    base_config_dict_20k["n_test"] = 70000
    base_config_dict_20k["n_train"] = 20000
    base_config_dict_20k["n_val"] = 10000
    base_config_dict_20k["results_dir"] = "results/20k"

    for decoder in [True]:
        for task in ["reversed_addition"]:
            for seed in [1, 2]:
                config_dict = copy.deepcopy(base_config_dict_20k)
                config_dict["decoder"] = decoder
                config_dict["task"] = task
                config_dict["seed"] = seed
                config_dict[
                    "name"
                ] = f"20k_{task}_{'decoder' if decoder else 'encoder'}_{seed}"

                config = Config(config_dict)
                env = Environment()

                generate_addition_data(config)
                train(config, env)
                evaluate(config, env, log_incorrect_examples=True)
