import os
import sys

from sklearn.model_selection import train_test_split  # type: ignore

from util import Config


def count_digits(n1: int, n2: int) -> int:
    return max(len(str(n1)), len(str(n2)))


def count_carrys(n1: int, n2: int) -> int:
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


def make_file(
    config: Config, inputs: list[tuple[int, int]], outputs: list[int], name: str
) -> None:
    file_path = os.path.join(config.data_dir, f"{name}.txt")
    text = "".join(f"${i}+{j}={k}$\n" for (i, j), k in zip(inputs, outputs))
    with open(file_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python generate_data.py config-path.json")
        exit(1)

    config = Config(sys.argv[1])

    max_digits = 3
    assert config.n_train + config.n_val + config.n_test <= 10 ** (2 * max_digits)

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

    train_outputs_reversed = [int(str(n)[::-1]) for n in train_outputs]
    val_outputs_reversed = [int(str(n)[::-1]) for n in val_outputs]
    test_outputs_reversed = [int(str(n)[::-1]) for n in test_outputs]

    make_file(config, train_inputs, train_outputs, "train_plain_addition")
    make_file(config, train_inputs, train_outputs_reversed, "train_reversed_addition")

    make_file(config, val_inputs, val_outputs, "val_plain_addition")
    make_file(config, val_inputs, val_outputs_reversed, "val_reversed_addition")

    make_file(config, test_inputs, test_outputs, "test_plain_addition")
    make_file(config, test_inputs, test_outputs_reversed, "test_reversed_addition")
