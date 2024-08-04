import json

if __name__ == "__main__":
    config = {
        "data_dir": "data/addition",
        "n_train": "15000",
        "n_val": "10000",
        "n_test": "50000",
        "max_iters": "4000",
        "eval_interval": "100",
    }

    i = 0

    for decoder in ["true", "false"]:
        for task in ["plain_addition", "reversed_addition"]:
            for seed in range(5):
                config["decoder"] = str(decoder)
                config["task"] = str(task)
                config["seed"] = str(seed)

                with open(f"config{i}.json", "w") as f:
                    json.dump(config, f)

                i += 1
