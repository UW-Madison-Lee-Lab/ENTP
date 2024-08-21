import sys
from collections import defaultdict
from os import path
from typing import Literal

import torch
from torch import Tensor
from tqdm import tqdm  # type: ignore

from nano_transformer import TransformerConfig, TransformerLMHead
from util import Config, Environment, decode, encode


def evaluate_split_with_model(
    model: TransformerLMHead,
    config: Config,
    env: Environment,
    split: Literal["train", "val", "test"],
) -> float:
    data_path = path.join(config.data_dir, f"{split}_{config.task}.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        test_text = f.read()

    chars = sorted(list(set(test_text)))
    vocab_size = len(chars)
    if config.use_delimiter:
        assert vocab_size == 14
    else:
        assert vocab_size == 13

    char2int = {c: i for i, c in enumerate(chars)}

    model.eval()

    lines = test_text.split("\n")[:-1]
    line_lens = [len(line) for line in lines]
    line_eq_idxs = [line.find("=") for line in lines]
    lines_data = [t for t in zip(line_lens, line_eq_idxs)]

    line_groups = defaultdict(list)

    for line, line_data in zip(lines, lines_data):
        line_groups[line_data].append(line)

    batches: list[Tensor] = []
    batch_eq_idxs: list[int] = []

    for (_, eq_idx), grouped_lines in line_groups.items():
        for i in range(0, len(grouped_lines), config.test_batch_size):
            unencoded_batch = grouped_lines[i : i + config.test_batch_size]
            batches.append(encode(unencoded_batch, char2int))
            batch_eq_idxs.append(eq_idx)
            assert torch.all(batches[-1][:, batch_eq_idxs[-1]] == char2int["="])

    n_correct = 0
    n_total = 0

    for batch, eq_idx in zip(batches, batch_eq_idxs):
        input_ids = batch[:, : eq_idx + 1].to(env.device)

        with env.context:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=5,
                decoder=config.decoder,
            )

        output_ids = output_ids[:, eq_idx + 1 : batch.shape[1]].cpu()
        target_ids = batch[:, eq_idx + 1 :]

        correct = torch.all(output_ids == target_ids, dim=1)

        n_correct += int(torch.sum(correct.int()))
        n_total += len(correct)

    return n_correct / n_total


def evaluate_split(
    config: Config,
    env: Environment,
    split: Literal["train", "test"],
) -> tuple[int, int, list[tuple[Tensor, Tensor, Tensor]], dict[int, str]]:
    """
    Evaluates model on `split` dataset. Assumes model is in `config.model_dir`.
    Assumes data is in `config.data_dir`. Saves results in `config.results_dir`.
    Returns `(n_correct, n_total, incorrect_examples, int2char)`.
    """

    test_data_path = path.join(config.data_dir, f"{split}_{config.task}.txt")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_text = f.read()

    chars = sorted(list(set(test_text)))
    vocab_size = len(chars)
    if config.use_delimiter:
        assert vocab_size == 14
    else:
        assert vocab_size == 13

    char2int = {c: i for i, c in enumerate(chars)}
    int2char = {i: c for i, c in enumerate(chars)}

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        use_wpe=config.use_wpe,
    )

    model = TransformerLMHead(model_config).to(env.device)

    model_path = path.join(config.model_dir, config.checkpoint_name)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    lines = test_text.split("\n")[:-1]
    line_lens = [len(line) for line in lines]
    line_eq_idxs = [line.find("=") for line in lines]
    lines_data = [t for t in zip(line_lens, line_eq_idxs)]

    line_groups = defaultdict(list)

    for line, line_data in zip(lines, lines_data):
        line_groups[line_data].append(line)

    batches: list[Tensor] = []
    batch_eq_idxs: list[int] = []

    for (_, eq_idx), grouped_lines in line_groups.items():
        for i in range(0, len(grouped_lines), config.test_batch_size):
            unencoded_batch = grouped_lines[i : i + config.test_batch_size]
            batches.append(encode(unencoded_batch, char2int))
            batch_eq_idxs.append(eq_idx)
            assert torch.all(batches[-1][:, batch_eq_idxs[-1]] == char2int["="])

    n_correct = 0
    n_total = 0

    progress_bar = tqdm(
        zip(batches, batch_eq_idxs),
        desc=f"[{0:.2f}% {split}ing accuracy]",
        total=len(batches),
    )

    incorrect_examples = []

    for batch, eq_idx in progress_bar:
        input_ids = batch[:, : eq_idx + 1].to(env.device)

        with env.context:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=5,
                decoder=config.decoder,
            )

        output_ids = output_ids[:, eq_idx + 1 : batch.shape[1]].cpu()
        target_ids = batch[:, eq_idx + 1 :]

        correct = torch.all(output_ids == target_ids, dim=1)

        for i, c in enumerate(correct):
            if not c:
                example = (
                    batch[i, : eq_idx + 1],
                    output_ids[i],
                    target_ids[i],
                )
                incorrect_examples.append(example)

        n_correct += int(torch.sum(correct.int()))
        n_total += len(output_ids)

        progress_bar.set_description(
            f"[{100 * n_correct / n_total:.2f}% {split}ing accuracy]"
        )

    print(f"{split} dataset: {n_correct}/{n_total} correct")

    return n_correct, n_total, incorrect_examples, int2char


def evaluate(
    config: Config,
    env: Environment,
    log_incorrect_examples=False,
):
    n_correct_test, n_total_test, incorrect_examples_test, int2char = evaluate_split(
        config,
        env,
        split="test",
    )

    n_correct_train, n_total_train, *_ = evaluate_split(
        config,
        env,
        split="train",
    )

    results_str = f"{n_correct_test}/{n_total_test} (test)\n"
    results_str += f"{n_correct_train}/{n_total_train} (train)\n"
    results_str += f"{str(config.to_dict())}\n"

    if log_incorrect_examples:
        for input_ids, output_ids, target_ids in incorrect_examples_test:
            input_str = decode(input_ids, int2char)
            output_str = decode(output_ids, int2char).removesuffix("\n")
            target_str = decode(target_ids, int2char).removesuffix("\n")
            results_str += f"{input_str}{output_str},{input_str}{target_str}\n"

    f_name = f"{config.name}_results.txt"
    with open(path.join(config.results_dir, f_name), "w") as f:
        f.write(results_str)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python eval-addition.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    env = Environment()
    evaluate(config, env, log_incorrect_examples=True)
