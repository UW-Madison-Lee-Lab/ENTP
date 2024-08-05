import os
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils import data

from .config import Config


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


def encode(text: str | list[str], char2int: dict[str, int]) -> Tensor:
    """Encodes `text` at character level."""
    if isinstance(text, list):
        return torch.tensor([[char2int[c] for c in s if c in char2int] for s in text])
    else:
        return torch.tensor([char2int[c] for c in text if c in char2int])


def decode(y: list[int] | Tensor, int2char: dict[int, str]) -> str:
    """Decodes `text` at character level."""
    return "".join([int2char[int(i)] for i in y if int(i) in int2char])


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
