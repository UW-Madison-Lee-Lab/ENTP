import random
from collections import Counter
from typing import Optional

import torch
from torch import Tensor

from util import Config, Environment


class DataGenerator:
    def __init__(self, config: Config, env: Environment) -> None:
        self.seed_size = config.data_gen_seed_size
        self.seed_max = config.data_gen_seed_max  # exclusive
        self.block_size = config.block_size
        self.batch_size = config.batch_size
        assert self.seed_max <= self.vocab_size

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def f(self, x: list[int]) -> int:
        raise NotImplementedError

    def generate_example(self) -> list[int]:
        seq = [random.randint(0, self.seed_max - 1) for _ in range(self.seed_size)]

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
    def __init__(self, config: Config, env: Environment) -> None:
        super().__init__(config, env)
        self.permutation_invariant = config.counting_permutation_invariant

    @property
    def vocab_size(self) -> int:
        return self.block_size

    def f(self, x: list[int]) -> int:
        y = x[:-1] if self.permutation_invariant else x[:-2]
        z = x[-1] if self.permutation_invariant else x[-2]
        return Counter(y)[z]


class TripletCountingDataGenerator(DataGenerator):
    @property
    def vocab_size(self) -> int:
        return self.block_size

    def f(self, x: list[int]) -> int:
        n = len(x)
        count = 0
        mod_counts = [0] * n
        for i in range(n):
            mod_counts[(x[i] + x[-1]) % n] += 1

        for i in range(n):
            count += mod_counts[(n - x[i]) % n]

        return count % n


DATA_GENERATORS: dict[str, type[DataGenerator]] = {
    "counting": CountingDataGenerator,
    "triplet_counting": TripletCountingDataGenerator,
}
