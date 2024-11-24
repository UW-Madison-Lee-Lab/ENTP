import os
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils import data
import torch.nn.functional as F

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

    
class SeqAlignmentBlockDataset(data.Dataset):
    """Groups time series data into blocks."""

    def __init__(self, data: Tensor, labels: Tensor, config: Config):
        self.data = data
        self.labels = labels
        self.block_size = config.block_size
        self.block_idxs = np.random.permutation(len(data) - self.block_size)

    def __len__(self) -> int:
        return len(self.block_idxs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        j = self.block_idxs[i]
        x = self.data[j : j + self.block_size]
        y = self.labels[j : j + self.block_size]
        return x, y


class SeqAlignmentBlockDataset(data.Dataset):
    """Groups time series data into blocks."""

    def __init__(self, data: Tensor, labels: Tensor, config: Config):
        self.data = data
        self.labels = labels
        self.block_size = config.block_size
        self.block_idxs = np.random.permutation(len(data) - self.block_size)

    def __len__(self) -> int:
        return len(self.block_idxs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        j = self.block_idxs[i]
        x = self.data[j : j + self.block_size]
        y = self.labels[j : j + self.block_size]
        return x, y


class SequenceDataset(data.Dataset):
    """Groups time series data into blocks."""

    def __init__(self, data: Sequence[Tensor], labels: Sequence[Tensor], _: Config):
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.data[i], self.labels[i]


def left_pad_collate(batch: list[tuple[Tensor, Tensor]], value: int):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    x_batch = torch.stack([F.pad(x, (max_len - len(x), 0), value=value) for x in xs], dim=0)
    y_batch = torch.stack(ys, dim=0)
    return x_batch, y_batch


def left_pad_collate_both(batch: list[tuple[Tensor, Tensor]], value1: int, value2: int):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    x_batch = torch.stack([F.pad(x, (max_len - len(x), 0), value=value1) for x in xs], dim=0)
    y_batch = torch.stack([F.pad(y, (max_len - len(y), 0), value=value2) for y in ys], dim=0)
    return x_batch, y_batch


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
