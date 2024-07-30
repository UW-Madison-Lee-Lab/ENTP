from collections import defaultdict
from contextlib import nullcontext
from os import path
from typing import ContextManager

import torch
from torch import Tensor
from tqdm import tqdm  # type: ignore

from nano_transformer import TransformerConfig, TransformerLMHead

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

context: ContextManager = (
    nullcontext() if device == "mps" else torch.autocast(device, dtype=torch.bfloat16)  # type: ignore
)
pin_memory = device == "cuda"
pin_memory_device = device if device == "cuda" else ""

ADDITION_TYPE = "reversed"

DATA_DIR = "data"
OUT_DIR = "out"
MODEL_DIR = "models"
MODEL_NAME = f"{ADDITION_TYPE}_decoder.pt"

N_EMBD = 384
N_LAYER = 6
N_HEAD = 6

BLOCK_SIZE = 128
BATCH_SIZE = 2500


def encode(text: str | list[str], char2int: dict[str, int]) -> Tensor:
    if isinstance(text, list):
        return torch.tensor([[char2int[c] for c in s if c in char2int] for s in text])
    else:
        return torch.tensor([char2int[c] for c in text if c in char2int])


def decode(y: list[int] | Tensor, int2char: dict[int, str]) -> str:
    return "".join([int2char[int(i)] for i in y if int(i) in int2char])


if __name__ == "__main__":
    test_data_path = path.join(DATA_DIR, f"test_{ADDITION_TYPE}.txt")
    with open("data/test_reversed.txt", "r", encoding="utf-8") as f:
        test_text = f.read()

    chars = sorted(list(set(test_text)))
    vocab_size = len(chars)
    assert vocab_size == 14

    char2int = {c: i for i, c in enumerate(chars)}
    int2char = {i: c for i, c in enumerate(chars)}

    config = TransformerConfig(
        n_positions=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
    )

    model = TransformerLMHead(config).to(device)

    model_path = path.join(MODEL_DIR, MODEL_NAME)
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
        for i in range(0, len(grouped_lines), BATCH_SIZE):
            unencoded_batch = grouped_lines[i : i + BATCH_SIZE]
            batches.append(encode(unencoded_batch, char2int))
            batch_eq_idxs.append(eq_idx)
            assert torch.all(batches[-1][:, batch_eq_idxs[-1]] == char2int["="])

    n_correct = 0
    n_total = 0

    progress_bar = tqdm(
        zip(batches, batch_eq_idxs),
        desc=f"[{0:.2f}% accuracy]",
        total=len(batches),
    )

    incorrect_examples = []

    for batch, eq_idx in progress_bar:
        input_ids = batch[:, : eq_idx + 1].to(device)

        with context:
            output_ids = model.generate(input_ids, max_new_tokens=5)

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

        progress_bar.set_description(f"[{100 * n_correct / n_total:.2f}% accuracy]")

    print(f"{n_correct}/{n_total} correct")

    incorrect_examples_text = ""
    for input_ids, output_ids, target_ids in incorrect_examples:
        input_str = decode(input_ids, int2char)
        output_str = decode(output_ids, int2char).removesuffix("\n")
        target_str = decode(target_ids, int2char).removesuffix("\n")
        incorrect_examples_text += f"{input_str}{output_str},{input_str}{target_str}\n"

    with open(path.join(OUT_DIR, "incorrect_examples.txt"), "w") as f:
        f.write(incorrect_examples_text)
