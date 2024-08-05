import json
from typing import Callable, Literal, Self


def str_to_bool(b: str) -> bool:
    """Converts `str` to `bool`."""
    b = b.lower()
    assert b == "true" or b == "false"
    return b == "true"


class Config:
    """Configuration for data generation, training, and evaluation."""

    task: Literal["plain_addition", "reversed_addition", "shakespeare"] = (
        "plain_addition"
    )
    decoder: bool = True
    data_dir: str = "data/addition"
    model_dir: str = "models"
    results_dir: str = "results"
    resume: bool = False
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    dropout: float = 0.0
    block_size: int = 64
    batch_size: int = 64
    test_batch_size: int = 2048
    max_iters: int = 5000
    eval_interval: int = 100
    min_lr: float = 5e-5
    max_lr: float = 5e-4
    warmup_iters: int = 2000
    lr_decay_iters: int = 5000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    seed: int = 42
    n_digits: int = 3
    n_train: int = 10000
    n_val: int = 10000
    n_test: int = 75000
    use_delimiter: bool = True
    resample_data: bool = True
    name: str = ""

    key_to_type: dict[str, Callable[[str], bool | int | float | str]] = {
        "task": str,
        "decoder": str_to_bool,
        "data_dir": str,
        "model_dir": str,
        "results_dir": str,
        "resume": str_to_bool,
        "n_embd": int,
        "n_layer": int,
        "n_head": int,
        "dropout": float,
        "block_size": int,
        "batch_size": int,
        "test_batch_size": int,
        "max_iters": int,
        "eval_interval": int,
        "min_lr": float,
        "max_lr": float,
        "warmup_iters": int,
        "lr_decay_iters": int,
        "weight_decay": float,
        "beta1": float,
        "beta2": float,
        "seed": int,
        "n_digits": int,
        "n_train": int,
        "n_val": int,
        "n_test": int,
        "use_delimiter": str_to_bool,
        "resample_data": str_to_bool,
        "name": str,
    }

    def __init__(self, config: dict) -> None:
        for k, v in config.items():
            setattr(self, k, v)

        assert self.name != ""

    def to_dict(self) -> dict[str, bool | int | float | str]:
        """Returns a `dict` with all configuration information."""
        d = {}
        for k in self.key_to_type.keys():
            d[k] = getattr(self, k)

        return d

    @property
    def checkpoint_name(self) -> str:
        return self.name + ".pt"

    @staticmethod
    def from_json(path: str) -> Self:
        with open(path, "r") as f:
            config = json.load(f)

        return Config(config)
