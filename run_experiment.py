import copy

from evaluate import evaluate
from generate_addition_data import generate_addition_data
from train import train
from util import Config, Environment

BASE_CONFIG_DICT = {
    "batch_size": 64,
    "beta1": 0.9,
    "beta2": 0.99,
    "block_size": 64,
    "data_dir": "data/addition",
    "decoder": True,
    "dropout": 0.0,
    "eval_interval": 100,
    "lr_decay_iters": 5000,
    "max_iters": 4000,
    "max_lr": 5e-4,
    "min_lr": 5e-5,
    "n_test": 75000,
    "n_train": 15000,
    "n_val": 10000,
    "resample_data": True,
    "results_dir": "results/15k",
    "use_delimiter": True,
    "warmup_iters": 150,
}


if __name__ == "__main__":
    for decoder in [True, False]:
        for task in ["plain_addition", "reversed_addition"]:
            for seed in range(5):
                config_dict = copy.deepcopy(BASE_CONFIG_DICT)
                config_dict["decoder"] = decoder
                config_dict["task"] = task
                config_dict["seed"] = seed
                config_dict[
                    "name"
                ] = f"{task}_{'decoder' if decoder else 'encoder'}_{seed}"

                config = Config(config_dict)
                env = Environment()

                generate_addition_data(config)
                train(config, env)
                evaluate(config, env, log_incorrect_examples=True)
