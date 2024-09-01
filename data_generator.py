import random
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch
from numba import njit  # type: ignore
from torch import Tensor

from util import Config


class DataGenerator:
    def __init__(self, config: Config) -> None:
        self.seed_size = config.data_gen_seed_size
        self.seed_max = config.data_gen_seed_max
        self.block_size = config.block_size
        self.batch_size = config.batch_size
        assert self.seed_max < self.vocab_size

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def f(self, x: list[int]) -> int:
        raise NotImplementedError

    def generate_example(self) -> list[int]:
        seq = [random.randint(0, self.seed_max) for _ in range(self.seed_size)]

        while len(seq) <= self.block_size:
            seq.append(self.f(seq))

        assert len(seq) == self.block_size + 1
        return seq

    def generate_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, list[int]]:
        if batch_size is None:
            batch_size = self.batch_size

        data = torch.tensor([self.generate_example() for _ in range(batch_size)])
        x = data[:, :-1]
        y = data[:, 1:]
        forward_idxs = [i for i in range(self.seed_size, self.block_size)]
        return x, y, forward_idxs


class CountingDataGenerator(DataGenerator):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.permutation_invariant = config.counting_permutation_invariant

    @property
    def vocab_size(self) -> int:
        return self.block_size

    def f(self, x: list[int]) -> int:
        y = x[:-1] if self.permutation_invariant else x[:-2]
        z = x[-1] if self.permutation_invariant else x[-2]
        return Counter(y)[z]


class SuperquadraticDataGenerator(DataGenerator):
    @property
    def vocab_size(self) -> int:
        return self.batch_size**2

    def f(self, x: list[int]) -> int:
        n = len(x)
        mod_counts: dict[int, int] = defaultdict(int)
        count = 0
        for i in range(n):
            mod = x[i] % n
            count += mod_counts[(n - mod) % n]
            mod_counts[mod] += 1

        return count


class NewSuperquadraticDataGenerator(DataGenerator):
    @property
    def vocab_size(self) -> int:
        return self.block_size

    @staticmethod
    @njit
    def __helper(x: np.ndarray, mod_counts: np.ndarray) -> int:
        n = len(x)
        count = 0
        for i in range(n - 2):
            mod_counts &= 0
            for j in range(i + 1, n):
                count += mod_counts[(n - x[j]) % n]
                mod_counts[(x[i] + x[j]) % n] += 1

        return count

    def f(self, x: list[int]) -> int:
        return self.__helper(np.array(x), np.zeros(len(x), dtype=int)) % self.vocab_size
