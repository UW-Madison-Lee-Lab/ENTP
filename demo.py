import sys
from contextlib import nullcontext

import torch
from torch import Tensor

from nano_model import TransformerConfig, TransformerLMHead

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

context = nullcontext() if device == "mps" else torch.autocast(device)
pin_memory = device == "cuda"
pin_memory_device = device if device == "cuda" else ""

MODEL_PATH = "models/plain_decoder.pt"

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
    if len(sys.argv) != 3:
        print("usage: python demo.py n1 n2")
        exit(1)

    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
    n_digits = max(len(str(n1)), len(str(n2)))

    with open("data/test_plain.txt", "r", encoding="utf-8") as f:
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

    checkpoint = torch.load("models/plain_decoder.pt", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt = f"${n1}+{n2}="
    encoded_prompt = encode(prompt, char2int)[None].to(device)

    with context:
        output = model.generate(encoded_prompt, max_new_tokens=n_digits + 1)

    decoded_output = decode(output[0], int2char)
    print(decoded_output.replace("$", "").removesuffix("\n"))
