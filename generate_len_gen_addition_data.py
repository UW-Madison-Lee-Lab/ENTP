import os
import random
import sys
from typing import TypeVar

from util import Config

T = TypeVar("T")


class AdditionGenerator:
    def __init__(self, config: Config, extra_test_digits=2) -> None:
        self.n_digits_max = config.n_digits
        self.reversed = reversed
        self.char2int = {c: i for i, c in enumerate(sorted("0123456789+=\n"))}
        self.int2char = {i: c for c, i in self.char2int.items()}
        self.extra_test_digits = extra_test_digits

    def generate_number(self, n_digits: int) -> int:
        return random.randint(10 ** (n_digits - 1), 10**n_digits - 1)

    def generate_example(self, train=True) -> tuple[int, int, int]:
        if train:
            x = self.generate_number(random.randint(1, self.n_digits_max))
            y = self.generate_number(random.randint(1, self.n_digits_max))
        else:
            x = self.generate_number(random.randint(1, self.n_digits_max + self.extra_test_digits))
            y = self.generate_number(random.randint(self.n_digits_max + 1, self.n_digits_max + self.extra_test_digits))

        if random.choice((True, False)):
            x, y = y, x

        z = x + y

        return x, y, z


def make_file(
    config: Config,
    data: dict[tuple[int, int], str],
    name: str,
) -> None:
    """Saves data file to `config.data_dir`."""
    if config.use_delimiter:
        text = "".join(f"${i}+{j}={k}$\n" for (i, j), k in data.items())
    else:
        text = "".join(f"{i}+{j}={k}\n" for (i, j), k in data.items())

    file_path = os.path.join(config.data_dir, f"{name}.txt")
    with open(file_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python generate_len_gen_addition_data.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])

    gen = AdditionGenerator(config)

    train_data = {}
    while len(train_data) < config.n_train:
        x, y, z = gen.generate_example(train=True)
        train_data[(x, y)] = str(z)[::-1]

    val_data = {}
    while len(val_data) < config.n_val:
        x, y, z = gen.generate_example(train=True)
        if (x, y) not in train_data:
            val_data[(x, y)] = str(z)[::-1]

    test_data = {}
    while len(test_data) < config.n_test:
        x, y, z = gen.generate_example(train=False)
        test_data[(x, y)] = str(z)[::-1]


    make_file(config, train_data, "train_reversed_addition_len_gen")

    make_file(config, val_data, "val_reversed_addition_len_gen")

    make_file(config, test_data, "test_reversed_addition_len_gen")
