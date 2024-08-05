import json

if __name__ == "__main__":
    config = {
        "data_dir": "data/addition",
        "results_dir": "results/5k",
        "decoder": "true",
        "n_train": 5000,
        "n_val": 10000,
        "n_test": 75000,
        "use_dollar_signs": "true",
        "resample_data": "true",
        "max_iters": 5000,
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

    i = 0

    for decoder in ["true", "false"]:
        for task in ["plain_addition", "reversed_addition"]:
            for seed in range(5):
                config["decoder"] = decoder
                config["task"] = task
                config["seed"] = seed
                config[
                    "name"
                ] = f"{task}_{'decoder' if decoder == 'true' else 'encoder'}_{seed}"

                with open(f"config{i}.json", "w") as f:
                    json.dump(config, f)

                i += 1
