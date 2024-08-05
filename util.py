import json
import os
import random
from contextlib import nullcontext
from typing import Callable, ContextManager, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils import data


def str_to_bool(b: str) -> bool:
    """Converts `str` to `bool`."""
    b = b.lower()
    assert b == "true" or b == "false"
    return b == "true"


def encode(text: str | list[str], char2int: dict[str, int]) -> Tensor:
    """Encodes `text` at character level."""
    if isinstance(text, list):
        return torch.tensor([[char2int[c] for c in s if c in char2int] for s in text])
    else:
        return torch.tensor([char2int[c] for c in text if c in char2int])


def decode(y: list[int] | Tensor, int2char: dict[int, str]) -> str:
    """Decodes `text` at character level."""
    return "".join([int2char[int(i)] for i in y if int(i) in int2char])


def seed_everything(seed: int) -> None:
    """Sets seeds for Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config:
    """
    Configuration for data generation, training, and evaluation.
    `Config` can an be constructed from a `json` file.
    """

    task: Literal[
        "plain_addition", "reversed_addition", "shakespeare"
    ] = "plain_addition"
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
    max_iters: int = 4000
    eval_interval: int = 100
    min_lr: float = 6e-5
    max_lr: float = 6e-4
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    seed: int = 42
    n_digits: int = 3
    n_train: int = 15000
    n_val: int = 10000
    n_test: int = 50000
    use_dollar_signs: bool = True
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
        "use_dollar_signs": str_to_bool,
        "resample_data": str_to_bool,
        "name": str,
    }

    def __init__(self, config_path: str | None = None) -> None:
        if config_path is not None:
            self.update(config_path)

        assert self.name != ""

    def update(self, config_path: str) -> None:
        """Updates config from a `json` file."""
        with open(config_path, "r") as f:
            config = json.load(f)

        for k, constructor in self.key_to_type.items():
            if k in config:
                setattr(self, k, constructor(config[k]))

    def to_dict(self) -> dict[str, bool | int | float | str]:
        """Returns a `dict` with all configuration information."""
        d = {}
        for k in self.key_to_type.keys():
            d[k] = getattr(self, k)

        return d

    @property
    def checkpoint_name(self) -> str:
        return self.name + ".pt"


class Environment:
    """Configures the environment based on the backend that is available."""

    def __init__(self) -> None:
        torch.set_float32_matmul_precision("high")
        self.context: ContextManager = nullcontext()
        self.pin_memory = False
        self.pin_memory_device = ""
        self.compile_blocks = False

        if torch.cuda.is_available():
            self.device = "cuda"
            self.context = torch.autocast(self.device, dtype=torch.bfloat16)
            self.pin_memory = True
            self.pin_memory_device = "cuda"
            self.compile_blocks = True
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"


class LRSchedule:
    """Cosine learning rate scheduler with warmup."""

    def __init__(self, config: Config) -> None:
        self.min_lr: float = config.min_lr
        self.max_lr: float = config.max_lr
        self.warmup_iters: int = config.warmup_iters
        self.lr_decay_iters: int = config.lr_decay_iters

    def __call__(self, i: int) -> float:
        if i < self.warmup_iters:
            return self.max_lr * i / self.warmup_iters

        if i > self.lr_decay_iters:
            return self.min_lr

        decay_ratio = (i - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio and decay_ratio <= 1
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


class BlockDataset(data.Dataset):
    """Groups time series data into blocks for autoregressive modeling."""

    def __init__(self, data: Tensor, config: Config):
        self.data = data
        self.block_size = config.block_size
        self.block_idxs = np.random.permutation(len(data) - self.block_size)

    def __len__(self) -> int:
        return len(self.block_idxs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        j = self.block_idxs[i]
        x = self.data[j : j + self.block_size]
        y = self.data[j + 1 : j + self.block_size + 1]
        return x, y


def load_data(
    config: Config,
    split: Literal["train", "val", "test"],
    char2int: dict[str, int] | None = None,
) -> tuple[BlockDataset, dict[str, int]]:
    """Loads data specified by `config`. Return encoded data as `BlockDataset`."""
    data_path = os.path.join(config.data_dir, f"{split}_{config.task}.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    if char2int is None:
        chars = sorted(list(set(text)))
        char2int = {c: i for i, c in enumerate(chars)}

    return BlockDataset(encode(text, char2int), config), char2int
