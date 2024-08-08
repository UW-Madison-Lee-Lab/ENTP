import ast
from collections import Counter

from generate_addition_data import (
    count_carrys,
    count_digits,
    generate_addition_data,
)
from util import Config


def parse_line(s: str) -> tuple[int, int]:
    if s[0] == "$":
        s = s[1:]

    i = s.find("+")
    j = s.find("=")
    n1 = int(s[:i])
    n2 = int(s[i + 1 : j])
    n_digits = count_digits(n1, n2)
    n_carrys = count_carrys(n1, n2)
    return n_digits, n_carrys


def process_result_file(result_file_path: str, header=False) -> str:
    with open(result_file_path, "r") as f:
        result_line, config_line, *incorrect_examples = f.readlines()

    i = result_line.find("/")
    n_correct = int(result_line[:i])
    n_total = int(result_line[i + 1 : -1])
    accuracy = n_correct / n_total

    config = ast.literal_eval(config_line[:-1])

    assert n_total == config["n_test"]

    row = {"n_correct_test": n_correct, "accuracy_test": accuracy} | config

    incorrect_counts = Counter([parse_line(line) for line in incorrect_examples])

    del config["checkpoint_name"]
    generate_addition_data(Config(config))

    with open(f"{config['data_dir']}/test_plain_addition.txt") as f:
        test_lines = f.readlines()

    test_counts = Counter([parse_line(line) for line in test_lines])

    for k in incorrect_counts.keys():
        assert k in test_counts

    assert sum(test_counts.values()) == n_total
    assert n_total - sum(incorrect_counts.values()) == n_correct

    for key in test_counts.keys():
        n_total_key = test_counts[key]
        n_incorrect_key = incorrect_counts[key] if key in incorrect_counts else 0
        n_correct_key = n_total_key - n_incorrect_key
        key_str = f"{key[0]}d_{key[1]}c"

        row[f"n_{key_str}"] = n_total_key
        row[f"n_correct_{key_str}"] = n_correct_key
        row[f"accuracy_{key_str}"] = n_correct_key / n_total_key

    row_keys = sorted(row.keys())

    if header:
        return ",".join(row_keys) + "\n"
    else:
        return ",".join(str(row[k]) for k in row_keys) + "\n"
