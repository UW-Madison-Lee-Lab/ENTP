import random
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch
from numba import njit  # type: ignore
from torch import Tensor

from nano_transformer import TransformerConfig, TransformerLMHead
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


class SuperquadraticDataGenerator(DataGenerator):
    @property
    def vocab_size(self) -> int:
        return self.block_size**2

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


class TransformerGenerator(DataGenerator):
    def __init__(self, config: Config, env: Environment) -> None:
        super().__init__(config, env)

        self.env = env

        assert config.task in ["decoder", "encoder"]
        self.decoder = config.task == "decoder"

        model_config = TransformerConfig(
            n_positions=config.block_size,
            vocab_size=self.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            use_wpe=config.use_wpe,
        )

        self.model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)
        self.temperature = config.data_gen_temperature

    @property
    def vocab_size(self) -> int:
        return self.seed_max

    def generate_example(self) -> list[int]:
        raise NotImplementedError

    def generate_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, list[int]]:
        if batch_size is None:
            batch_size = self.batch_size

        x = torch.randint(
            high=self.vocab_size,  # seed_max is vocab_size
            size=(batch_size, self.block_size),
            device=self.env.device,
        )

        with self.env.context:
            logits = self.model(x, decoder=self.decoder)

        y = torch.argmax(logits, dim=2)

        forward_idxs = [i for i in range(self.block_size)]
        return x, y, forward_idxs


class AutoregressiveTransformerGenerator(DataGenerator):
    def __init__(self, config: Config, env: Environment) -> None:
        super().__init__(config, env)

        self.env = env

        assert config.task in ["autoregressive_decoder", "autoregressive_encoder"]
        self.decoder = config.task == "autoregressive_decoder"

        model_config = TransformerConfig(
            n_positions=config.block_size,
            vocab_size=self.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            use_wpe=config.use_wpe,
        )

        self.model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)
        self.temperature = config.data_gen_temperature

    @property
    def vocab_size(self) -> int:
        return self.seed_max

    def generate_example(self) -> list[int]:
        raise NotImplementedError

    def generate_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, list[int]]:
        if batch_size is None:
            batch_size = self.batch_size

        seed = torch.randint(
            high=self.vocab_size,  # seed_max is vocab_size
            size=(batch_size, self.seed_size),
            device=self.env.device,
        )

        with self.env.context:
            data = self.model.generate(
                seed,
                max_new_tokens=self.block_size - self.seed_size + 1,
                decoder=self.decoder,
                deterministic=False,
                temperature=self.temperature,
            )

        x = data[:, :-1]
        y = data[:, 1:]
        forward_idxs = [i for i in range(self.seed_size, self.block_size)]
        return x, y, forward_idxs


class Match3Generator(DataGenerator):
    def __init__(self, config: Config, env: Environment, mod: int = 100) -> None:
        self.mod = mod
        self.false_id = self.mod
        self.true_id = self.mod + 1
        super().__init__(config, env)

    @property
    def vocab_size(self) -> int:
        return self.mod + 2

    def f(self, x: list[int] | Tensor) -> int:
        n = len(x)
        i = n - 1
        for j in range(n - 2):
            for k in range(j + 1, n - 1):
                if (x[i] + x[j] + x[k]) % self.mod == 0:
                    return self.true_id

        return self.false_id

    def generate_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, list[int]]:
        if batch_size is None:
            batch_size = self.batch_size

        x = torch.tensor(
            [
                [random.randint(0, self.mod - 1) for _ in range(self.block_size)]
                for _ in range(batch_size)
            ]
        )

        y = torch.empty_like(x)

        for i in range(batch_size):
            for j in range(self.block_size):
                y[i, j] = self.f(x[i, : j + 1])

        forward_idxs = [i for i in range(self.block_size)]
        return x, y, forward_idxs


DATA_GENERATORS: dict[str, type[DataGenerator]] = {
    "counting": CountingDataGenerator,
    "superquadratic": SuperquadraticDataGenerator,
    "new_superquadratic": NewSuperquadraticDataGenerator,
    "decoder": TransformerGenerator,
    "encoder": TransformerGenerator,
    "autoregressive_decoder": AutoregressiveTransformerGenerator,
    "autoregressive_encoder": AutoregressiveTransformerGenerator,
    "match3": Match3Generator,
}
