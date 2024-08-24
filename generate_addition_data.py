import os
import sys
from typing import TypeVar

from sklearn.model_selection import train_test_split  # type: ignore

from util import Config

T = TypeVar("T")


def count_digits(n1: int, n2: int) -> int:
    """Returns the number of digits in the largest number."""
    return max(len(str(n1)), len(str(n2)))


def count_carrys(n1: int, n2: int) -> int:
    """Returns the carry operations whan computing `n1 + n2`."""
    digits1 = [int(d) for d in str(n1)][::-1]
    digits2 = [int(d) for d in str(n2)][::-1]

    digits1 += [0] * max(0, len(digits2) - len(digits1))
    digits2 += [0] * max(0, len(digits1) - len(digits2))

    res = 0
    c = 0
    for d1, d2 in zip(digits1, digits2):
        s = d1 + d2 + c
        c = s // 10
        if c > 0:
            res += 1

    return res


def resample(
    config: Config,
    inputs: list[tuple[int, int]],
    outputs: list[int],
    digits: list[int],
    carrys: list[int],
) -> tuple[list[tuple[int, int]], list[int], list[int], list[int]]:
    """Resamples data, removing 90% of the three digit examples."""
    if config.n_digits != 3:
        raise NotImplementedError()

    def split(arr: list) -> tuple[list, list]:
        arr2 = []
        arr3 = []
        for i, d in enumerate(digits):
            assert d == 2 or d == 3
            if d == 2:
                arr2.append(arr[i])
            elif d == 3:
                arr3.append(arr[i])

        return arr2, arr3

    inputs2, inputs3 = split(inputs)
    outputs2, outputs3 = split(outputs)
    carrys2, carrys3 = split(carrys)

    new_inputs3, _, new_outputs3, _, new_carrys3, _ = train_test_split(
        inputs3,
        outputs3,
        carrys3,
        train_size=0.1,
        shuffle=True,
        stratify=carrys3,
        random_state=config.seed,
    )

    new_inputs = inputs2 + new_inputs3
    new_outputs = outputs2 + new_outputs3
    new_carrys = carrys2 + new_carrys3
    new_digits = [2] * len(inputs2) + [3] * len(new_inputs3)

    return new_inputs, new_outputs, new_digits, new_carrys


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


def generate_addition_data(config: Config) -> None:
    """Generates addition data and saves files to `config.data_dir`."""
    max_digits = config.n_digits

    inputs = []
    outputs = []
    digits = []
    carrys = []

    for i in range(10**max_digits):
        for j in range(10**max_digits):
            if count_digits(i, j) > 1:
                inputs.append((i, j))
                outputs.append(i + j)
                digits.append(count_digits(i, j))
                carrys.append(count_carrys(i, j))

    if config.resample_data:
        inputs, outputs, digits, carrys = resample(
            config,
            inputs,
            outputs,
            digits,
            carrys,
        )

    assert config.n_train + config.n_val + config.n_test <= len(inputs)

    (
        train_inputs,
        holdout_inputs,
        train_outputs,
        holdout_outputs,
        train_digits,
        holdout_digits,
        train_carrys,
        holdout_carrys,
    ) = train_test_split(
        inputs,
        outputs,
        digits,
        carrys,
        train_size=config.n_train - 100,
        test_size=config.n_val + config.n_test,
        shuffle=True,
        stratify=[t for t in zip(digits, carrys)],
        random_state=config.seed,
    )

    (
        val_inputs,
        test_inputs,
        val_outputs,
        test_outputs,
        val_digits,
        test_digits,
        val_carrys,
        test_carrys,
    ) = train_test_split(
        holdout_inputs,
        holdout_outputs,
        holdout_digits,
        holdout_carrys,
        train_size=config.n_val,
        test_size=config.n_test,
        shuffle=True,
        stratify=[t for t in zip(holdout_digits, holdout_carrys)],
        random_state=config.seed,
    )

    for i in range(10):
        for j in range(10):
            train_inputs.append((i, j))
            train_outputs.append(i + j)
            train_digits.append(count_digits(i, j))
            train_carrys.append(count_carrys(i, j))

    assert len(train_inputs) == config.n_train and len(train_outputs) == config.n_train
    assert len(val_inputs) == config.n_val and len(val_outputs) == config.n_val
    assert len(test_inputs) == config.n_test and len(test_outputs) == config.n_test

    for i in range(len(train_inputs)):
        n1, n2 = train_inputs[i]
        assert train_outputs[i] == n1 + n2
        assert train_digits[i] == count_digits(n1, n2)
        assert train_carrys[i] == count_carrys(n1, n2)

    for i in range(len(val_inputs)):
        n1, n2 = val_inputs[i]
        assert val_outputs[i] == n1 + n2
        assert val_digits[i] == count_digits(n1, n2)
        assert val_carrys[i] == count_carrys(n1, n2)

    for i in range(len(test_inputs)):
        n1, n2 = test_inputs[i]
        assert test_outputs[i] == n1 + n2
        assert test_digits[i] == count_digits(n1, n2)
        assert test_carrys[i] == count_carrys(n1, n2)

    for d in set(digits):
        for c in set(carrys):
            if c > d:
                continue

            n_train = sum(
                d == nd and c == nc for nd, nc in zip(train_digits, train_carrys)
            )
            n_test = sum(
                d == nd and c == nc for nd, nc in zip(test_digits, test_carrys)
            )

            if d == 1:
                assert n_train == 100
                assert n_test == 0
            else:
                min_ratio = 0.95 * (config.n_test / (config.n_train - 100))
                max_ratio = 1.05 * (config.n_test / (config.n_train - 100))
                ratio = n_test / n_train
                assert min_ratio < ratio and ratio < max_ratio

    train_outputs_reversed = [str(n)[::-1] for n in train_outputs]
    val_outputs_reversed = [str(n)[::-1] for n in val_outputs]
    test_outputs_reversed = [str(n)[::-1] for n in test_outputs]

    make_file(config, train_inputs, train_outputs, "train_plain_addition")
    make_file(config, train_inputs, train_outputs_reversed, "train_reversed_addition")

    make_file(config, val_inputs, val_outputs, "val_plain_addition")
    make_file(config, val_inputs, val_outputs_reversed, "val_reversed_addition")

    make_file(config, test_inputs, test_outputs, "test_plain_addition")
    make_file(config, test_inputs, test_outputs_reversed, "test_reversed_addition")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python generate_addition_data.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    generate_addition_data(config)
