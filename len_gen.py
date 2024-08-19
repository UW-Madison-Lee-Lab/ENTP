import os
import random
import sys
from pprint import pprint

import numpy as np
import torch
import wandb
from torch import Tensor

from nano_transformer import TransformerConfig, TransformerLMHead, flat_cross_entropy
from util import Config, Environment, LRSchedule


class AdditionGenerator:
    def __init__(
        self,
        n_digits_min: int,
        n_digits_max: int,
        block_size: int,
        batch_size: int,
    ) -> None:
        self.n_digits_min = n_digits_min
        self.n_digits_max = n_digits_max
        self.block_size = block_size
        self.batch_size = batch_size
        self.char2int = {c: i for i, c in enumerate(sorted("0123456789+=\n"))}
        self.int2char = {i: c for c, i in self.char2int.items()}

    def generate_number(self, use_n_digits_min: bool) -> int:
        if use_n_digits_min:
            n_digits = random.randint(self.n_digits_min, self.n_digits_max)
        else:
            n_digits = random.randint(1, self.n_digits_max)

        return random.randint(10 ** (n_digits - 1), 10**n_digits)

    def generate_example(self) -> tuple[list[int], list[int]]:
        x = self.generate_number(use_n_digits_min=True)
        y = self.generate_number(use_n_digits_min=False)

        if random.choice((True, False)):
            x, y = y, x

        z = x + y

        addition = [self.char2int[c] for c in f"{x}+{y}={z}\n"]
        abacus_embd = (
            list(range(1, len(str(x)) + 1))
            + [0]
            + list(range(1, len(str(y)) + 1))
            + [0]
            + list(range(1, len(str(z)) + 1))
            + [0]
        )

        assert len(addition) == len(abacus_embd)

        return addition, abacus_embd

    def generate_block(self) -> tuple[list[int], list[int]]:
        addition_block: list[int] = []
        abacus_embd_block: list[int] = []

        while len(addition_block) <= self.block_size:
            addition, abacus_embd = self.generate_example()
            addition_block += addition
            abacus_embd_block += abacus_embd

        extra = len(addition_block) - self.block_size
        start_idx = random.randint(0, extra)
        end_idx = len(addition_block) + start_idx - extra

        addition_block = addition_block[start_idx:end_idx]
        abacus_embd_block = abacus_embd_block[start_idx:end_idx]

        assert len(addition_block) == self.block_size
        assert len(abacus_embd_block) == self.block_size

        return addition_block, abacus_embd_block

    def generate_batch(self) -> tuple[Tensor, Tensor]:
        addition_batch: list[list[int]] = []
        abacus_embd_batch: list[list[int]] = []

        for _ in range(self.batch_size):
            addition_block, abacus_embd_block = self.generate_block()
            addition_batch.append(addition_block)
            abacus_embd_batch.append(abacus_embd_block)

        return torch.tensor(addition_batch), torch.tensor(abacus_embd_batch)


def train(config: Config, env: Environment, resume: bool = False) -> None:
    print(f"{env.device=}, env.context={str(type(env.context))[8 : -2]}", end=", ")
    print(f"{env.pin_memory=}, {env.pin_memory_device=}, {env.compile_blocks=}")
    pprint(config.to_dict())

    run = wandb.init(
        dir=config.results_dir,
        project="encoder-addition",
        config=config.to_dict(),
        name=config.name,
        resume=resume,
    )

    env.seed_everything(config.seed)

    train_data_generator = AdditionGenerator(
        n_digits_min=1,
        n_digits_max=config.n_digits_train,
        block_size=config.block_size + 1,
        batch_size=config.batch_size,
    )

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=len(train_data_generator.char2int),
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )

    model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)

    optimizer = model.configure_optimizers(
        lr=config.min_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        device=env.device,
    )

    lr_schedule = LRSchedule(config)
    i = 0
    best_loss = float("inf")
    losses = []
    n_evals_without_improving = 0

    if resume:
        load_path = os.path.join(config.model_dir, config.checkpoint_name)
        checkpoint = torch.load(load_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        i = checkpoint["i"]
        best_loss = checkpoint["best_loss"]

    while True:
        i += 1

        model.train()

        lr = lr_schedule(i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        addition_batch, abacus_embd_batch = train_data_generator.generate_batch()

        x = addition_batch[:, :-1].to(env.device)
        y = addition_batch[:, 1:].to(env.device)
        # embd = abacus_embd_batch[:, :-1].to(env.device)

        with env.context:
            logits = model(x, decoder=config.decoder)
            loss = flat_cross_entropy(logits, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

        if i % config.eval_interval == 0:
            train_loss = float(np.mean(losses[-config.eval_interval :]))
            wandb.log({"train_loss": train_loss}, step=i)

            if train_loss < best_loss:
                n_evals_without_improving = 0
                print(f"saved checkpoint    {f'{i=}':8}  {train_loss=:.3f}")
                best_loss = train_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "i": i,
                    "best_loss": best_loss,
                }
                save_path = os.path.join(config.model_dir, config.checkpoint_name)
                torch.save(checkpoint, save_path)
            else:
                n_evals_without_improving += 1

        if i >= config.max_iters or (
            n_evals_without_improving >= config.max_evals_without_improving
            and best_loss < config.max_loss_for_early_stopping
        ):
            run.finish()
            return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python train.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    env = Environment()

    train(config, env)
