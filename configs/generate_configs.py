import json

if __name__ == "__main__":
    config = {
        "data_dir": "data/addition",
        "results_dir": "results/15k_resampled",
        "decoder": "true",
        "n_train": 15000,
        "n_val": 10000,
        "n_test": 75000,
        "resample_data": "true",
        "max_iters": 6000,
        "eval_interval": 100,
        "block_size": 1024,
        "batch_size": 12,
        "max_lr": 1e-3,
        "min_lr": 1e-4,
        "dropout": 0.2,
        "lr_decay_iters": 5000,
        "warmup_iters": 100,
        "beta2": 0.99,
    }

    i = 0

    for use_dollar_signs in ["true", "false"]:
        for task in ["plain_addition", "reversed_addition"]:
            for seed in range(5):
                config["use_dollar_signs"] = str(use_dollar_signs)
                config["task"] = str(task)
                config["seed"] = str(seed)
                config["name"] = (
                    f"{task}_dollar_signs_{use_dollar_signs}_{seed}_resampled"
                )

                with open(f"config{i}.json", "w") as f:
                    json.dump(config, f)

                i += 1
