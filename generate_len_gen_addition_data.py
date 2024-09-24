import os
import random
import sys
from typing import TypeVar

from util import Config

T = TypeVar("T")


class AdditionGenerator:
    def __init__(self, config: Config) -> None:
        self.n_digits_max = config.n_digits
        self.reversed = reversed
        self.char2int = {c: i for i, c in enumerate(sorted("0123456789+=\n"))}
        self.int2char = {i: c for c, i in self.char2int.items()}

    def generate_number(self, n_digits: int) -> int:
        return random.randint(10 ** (n_digits - 1), 10**n_digits - 1)

    def generate_example(self, train=True) -> tuple[int, int, int]:
        if train:
            x = self.generate_number(random.randint(1, self.n_digits_max))
            y = self.generate_number(random.randint(1, self.n_digits_max))
        else:
            x = self.generate_number(random.randint(1, 2 * self.n_digits_max))
            y = self.generate_number(random.randint(self.n_digits_max + 1, 2 * self.n_digits_max))

        if random.choice((True, False)):
            x, y = y, x

        z = x + y

        return x, y, z


def make_file(
    config: Config,
    inputs: list[tuple[T, T]],
    outputs: list[T],
    name: str,
) -> None:
    """Saves data file to `config.data_dir`."""
    if config.use_delimiter:
        text = "".join(f"${i}+{j}={k}$\n" for (i, j), k in zip(inputs, outputs))
    else:
        text = "".join(f"{i}+{j}={k}\n" for (i, j), k in zip(inputs, outputs))

    file_path = os.path.join(config.data_dir, f"{name}.txt")
    with open(file_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python generate_len_gen_addition_data.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])

    gen = AdditionGenerator(config)

    train_inputs = []
    train_outputs = []
    for _ in range(config.n_train):
        x, y, z = gen.generate_example(train=True)
        train_inputs.append((x, y))
        train_outputs.append(z)

    val_inputs = []
    val_outputs = []
    for _ in range(config.n_val):
        x, y, z = gen.generate_example(train=True)
        val_inputs.append((x, y))
        val_outputs.append(z)

    test_inputs = []
    test_outputs = []
    for _ in range(config.n_test):
        x, y, z = gen.generate_example(train=False)
        test_inputs.append((x, y))
        test_outputs.append(z)

    train_outputs_reversed = [str(n)[::-1] for n in train_outputs]
    val_outputs_reversed = [str(n)[::-1] for n in val_outputs]
    test_outputs_reversed = [str(n)[::-1] for n in test_outputs]

    make_file(
        config, train_inputs, train_outputs_reversed, "train_reversed_addition_len_gen"
    )

    make_file(config, val_inputs, val_outputs_reversed, "val_reversed_addition_len_gen")

    make_file(
        config, test_inputs, test_outputs_reversed, "test_reversed_addition_len_gen"
    )
