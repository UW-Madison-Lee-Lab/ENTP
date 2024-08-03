import random
from contextlib import nullcontext
from os import path
from typing import ContextManager, Literal

import numpy as np
import torch
import wandb
from torch import Tensor, optim
from torch.utils import data

from nano_transformer import TransformerConfig, TransformerLMHead, flat_cross_entropy

torch.set_float32_matmul_precision("high")

context: ContextManager = nullcontext()
pin_memory = False
pin_memory_device = ""
compile_blocks = False

if torch.cuda.is_available():
    device = "cuda"
    context = torch.autocast(device, dtype=torch.bfloat16)
    pin_memory = True
    pin_memory_device = "cuda"
    compile_blocks = True
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(
    f"{device=}, {type(context)=}, {pin_memory=}, {pin_memory_device=}, {compile_blocks=}"
)

TASK: Literal["plain_addition", "reversed_addition", "shakespeare"] = "shakespeare"
DECODER: bool = True

DATA_DIR: str = "data/shakespeare"
OUT_DIR: str = "out"
CHECKPOINT_NAME: str = f"{TASK}_{'decoder' if DECODER else 'encoder'}.pt"
RESUME: bool = False

N_EMBD: int = 384
N_LAYER: int = 6
N_HEAD: int = 6

EVAL_INTERVAL: int = 100

BLOCK_SIZE: int = 64
BATCH_SIZE: int = 64

MAX_ITERS: int = 5000

MIN_LR: float = 6e-5
MAX_LR: float = 6e-4
WARMUP_ITERS: int = 2000
LR_DECAY_ITERS: int = 600000

WEIGHT_DECAY: float = 0.1
BETAS: tuple[float, float] = (0.9, 0.95)

SEED: int = 42


class BlockDataset(data.Dataset):
    def __init__(self, data: Tensor, block_size=BLOCK_SIZE):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        x = self.data[self.block_size * i : self.block_size * (i + 1)]
        y = self.data[self.block_size * i + 1 : self.block_size * (i + 1) + 1]
        return x, y


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(s: str, char2int: dict[str, int]) -> Tensor:
    return torch.tensor([char2int[c] for c in s if c in char2int])


def decode(y: list[int] | Tensor, int2char: dict[int, str]) -> str:
    return "".join([int2char[int(i)] for i in y if int(i) in int2char])


def get_lr(iter_num: int) -> float:
    if iter_num < WARMUP_ITERS:
        return MAX_LR * iter_num / WARMUP_ITERS

    if iter_num > LR_DECAY_ITERS:
        return MIN_LR

    decay_ratio = (iter_num - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio and decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


@torch.no_grad()
def evaluate_loss(
    model: TransformerLMHead,
    dataset: data.Dataset,
    max_iters=100,
) -> float:
    data_loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )

    loss_sum = 0.0
    cnt = 0
    for i, (x, y) in enumerate(data_loader):
        if i >= max_iters:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = flat_cross_entropy(logits, y).cpu()
        loss_sum += loss.cpu().item() * len(x)
        cnt += len(x)

    return loss_sum / cnt


def train(
    model: TransformerLMHead,
    optimizer: optim.Optimizer,
    train_dataset: data.Dataset,
    test_dataset: data.Dataset,
    load_checkpoint_name: str | None = None,
    save_checkpoint_name: str | None = None,
) -> None:
    i = 0
    best_test_loss = float("inf")

    if load_checkpoint_name is not None:
        load_checkpoint_path = path.join(OUT_DIR, load_checkpoint_name)
        checkpoint = torch.load(load_checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        i = checkpoint["i"]
        best_test_loss = checkpoint["best_test_loss"]

    while i < MAX_ITERS:
        train_data_loader = data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device,
        )

        for x, y in train_data_loader:
            if i >= MAX_ITERS:
                break

            lr = get_lr(i)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x = x.to(device)
            y = y.to(device)

            with context:
                logits = model(x, decoder=DECODER)
                loss = flat_cross_entropy(logits, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            wandb.log({"train_loss": loss.item()}, step=i)

            if (i + 1) % EVAL_INTERVAL == 0:
                test_loss = evaluate_loss(model, test_dataset)
                wandb.log({"test_loss": test_loss}, step=i)

                if test_loss < best_test_loss and save_checkpoint_name is not None:
                    best_test_loss = test_loss
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "i": i,
                        "best_test_loss": best_test_loss,
                    }
                    save_checkpoint_path = path.join(OUT_DIR, save_checkpoint_name)
                    torch.save(checkpoint, save_checkpoint_path)

            i += 1


if __name__ == "__main__":
    wandb.init(
        dir=OUT_DIR,
        project="encoder-addition",
        config={
            "n_embd": N_EMBD,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "max_iters": MAX_ITERS,
            "min_lr": MIN_LR,
            "max_lr": MAX_LR,
            "warmup_iters": WARMUP_ITERS,
            "lr_decay_iters": LR_DECAY_ITERS,
            "weight_decay": WEIGHT_DECAY,
            "betas": BETAS,
            "seed": SEED,
        },
        name=CHECKPOINT_NAME[:-3],
        resume=RESUME,
    )

    seed_everything(SEED)

    train_data_path = path.join(DATA_DIR, f"train_{TASK}.txt")
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    test_data_path = path.join(DATA_DIR, f"test_{TASK}.txt")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_text = f.read()

    chars = sorted(list(set(train_text)))
    vocab_size = len(chars)

    char2int = {c: i for i, c in enumerate(chars)}
    int2char = {i: c for i, c in enumerate(chars)}

    train_dataset = BlockDataset(encode(train_text, char2int))
    test_dataset = BlockDataset(encode(test_text, char2int))

    config = TransformerConfig(
        n_positions=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
    )

    model = TransformerLMHead(config, compile_blocks).to(device)

    optimizer = model.configure_optimizers(
        lr=MIN_LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
        device=device,
    )

    train(
        model,
        optimizer,
        train_dataset,
        test_dataset,
        load_checkpoint_name=CHECKPOINT_NAME if RESUME else None,
        save_checkpoint_name=CHECKPOINT_NAME,
    )
