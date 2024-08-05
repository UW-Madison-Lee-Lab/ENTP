import copy

from evaluate import evaluate
from generate_addition_data import generate_addition_data
from train import train
from util import Config

BASE_CONFIG_DICT = {
    "data_dir": "data/addition",
    "results_dir": "results/15k",
    "decoder": "true",
    "n_train": 15000,
    "n_val": 10000,
    "n_test": 75000,
    "use_dollar_signs": "true",
    "resample_data": "true",
    "max_iters": 4000,
    "eval_interval": 100,
    "block_size": 64,
    "batch_size": 64,
    "max_lr": 5e-4,
    "min_lr": 5e-5,
    "dropout": 0.0,
    "lr_decay_iters": 5000,
    "warmup_iters": 150,
    "beta1": 0.9,
    "beta2": 0.99,
}


if __name__ == "__main__":
    for decoder in ["true", "false"]:
        for task in ["plain_addition", "reversed_addition"]:
            for seed in range(5):
                config_dict = copy.deepcopy(BASE_CONFIG_DICT)
                config_dict["decoder"] = decoder
                config_dict["task"] = task
                config_dict["seed"] = seed
                config_dict["name"] = (
                    f"{task}_{'decoder' if decoder == 'true' else 'encoder'}_{seed}"
                )

                config = Config(config_dict)

                generate_addition_data(config)
                train(config)
                evaluate(config)
