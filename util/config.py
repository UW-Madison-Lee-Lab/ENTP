import json
from typing import Any, Literal


class Config:
    """Configuration for data generation, training, and evaluation."""

    batch_size: int = 64
    beta1: float = 0.9
    beta2: float = 0.99
    block_size: int = 64
    counting_permutation_invariant: bool = True
    custom_optim_groups: list[dict[str, Any]] = []
    data_dir: str = "data/addition"
    data_gen_seed_max: int = 16
    data_gen_seed_size: int = 16
    decoder: bool = True
    dropout: float = 0.0
    eval_interval: int = 100
    log_wpe_norm: bool = False
    lr_decay_iters: int = 5000
    max_evals_without_improving: int = 1000
    max_iters: int = 5000
    max_loss_for_early_stopping: float = 1e9
    max_lr: float = 5e-4
    min_lr: float = 5e-5
    model_dir: str = "models"
    n_digits: int = 3
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    n_test: int = 75000
    n_train: int = 10000
    n_val: int = 10000
    name: str = ""
    resample_data: bool = True
    results_dir: str = "results"
    resume: bool = False
    seed: int = 42
    task: Literal[
        "plain_addition",
        "reversed_addition",
        "shakespeare",
        "counting",
    ] = "plain_addition"
    test_batch_size: int = 2048
    test_accuracy_during_training: bool = False
    use_delimiter: bool = True
    use_wpe: bool = True
    warmup_iters: int = 100
    weight_decay: float = 0.1

    def __init__(self, config: dict[str, bool | int | float | str]) -> None:
        for k, v in config.items():
            setattr(self, k, v)

        assert self.name != ""

    def to_dict(self) -> dict[str, bool | int | float | str]:
        """Returns a `dict` with all configuration information."""
        d = {}
        for k in Config.__dict__.keys():
            if "__" not in k and k not in ("from_json", "to_dict"):
                d[k] = getattr(self, k)

        return d

    @property
    def checkpoint_name(self) -> str:
        return self.name + ".pt"

    @staticmethod
    def from_json(path: str) -> "Config":
        with open(path, "r") as f:
            config = json.load(f)

        return Config(config)
