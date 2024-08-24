import os
import random
import sys
from collections import Counter
from pprint import pprint

import numpy as np
import torch
import wandb
from torch import Tensor

from nano_transformer import TransformerConfig, TransformerLMHead, flat_cross_entropy
from util import Config, Environment, LRSchedule


class CountingDataGenerator:
    """Groups time series data into blocks for autoregressive modeling."""

    def __init__(self, config: Config):
        self.seed_size = config.counting_seed_size
        self.seed_max = config.counting_seed_max
        self.block_size = config.block_size
        self.batch_size = config.batch_size

    @staticmethod
    def f(x: list[int]) -> int:
        y = x[:-1]
        z = x[-1]
        return Counter(y)[z]

    def generate_example(self) -> list[int]:
        seq = [random.randint(0, self.seed_max) for _ in range(self.seed_size)]

        while len(seq) <= self.block_size:
            seq.append(self.f(seq))

        assert len(seq) == self.block_size + 1
        return seq

    def generate_batch(self) -> tuple[Tensor, Tensor, list[int]]:
        data = torch.tensor([self.generate_example() for _ in range(self.batch_size)])
        x = data[:, :-1]
        y = data[:, 1:]
        forward_idxs = [i for i in range(self.seed_size, self.batch_size)]
        return x, y, forward_idxs

    @property
    def vocab_size(self) -> int:
        seq = [0] * self.seed_size

        while len(seq) <= self.block_size:
            seq.append(self.f(seq))

        return max(seq) + 1


@torch.no_grad()
def test_accuracy(
    model: TransformerLMHead,
    data_generator: CountingDataGenerator,
    config: Config,
    env: Environment,
    n_iters=100,
) -> float:
    accuracies = []

    for _ in range(n_iters):
        x, y, forward_idxs = data_generator.generate_batch()

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            logits = model(x, decoder=config.decoder, forward_idxs=forward_idxs)

        y_pred = torch.argmax(logits, dim=2)

        y = y[:, forward_idxs]
        y_pred = y_pred[:, forward_idxs]
        accuracies.append(torch.mean((y == y_pred).float()).item())

    return float(np.mean(accuracies))


def train(config: Config, env: Environment) -> None:
    print(f"{env.device=}, env.context={str(type(env.context))[8 : -2]}", end=", ")
    print(f"{env.pin_memory=}, {env.pin_memory_device=}, {env.compile_blocks=}")
    pprint(config.to_dict())

    run = wandb.init(
        dir=config.results_dir,
        project="encoder-addition",
        config=config.to_dict(),
        name=config.name,
        resume=config.resume,
    )

    env.seed_everything(config.seed)

    data_generator = CountingDataGenerator(config)

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=data_generator.vocab_size,
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

    if config.resume:
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

        x, y, forward_idxs = data_generator.generate_batch()

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            logits = model(x, decoder=config.decoder, forward_idxs=forward_idxs)
            loss = flat_cross_entropy(logits[:, forward_idxs], y[:, forward_idxs])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

        if i % config.eval_interval == 0:
            eval_loss = float(np.mean(losses[-config.eval_interval :]))
            wandb.log({"loss": eval_loss}, step=i)

            if config.test_accuracy_during_training:
                eval_accuracy = test_accuracy(model, data_generator, config, env)
                wandb.log({"accuracy": eval_accuracy}, step=i)

            if eval_loss < best_loss:
                n_evals_without_improving = 0
                print(f"saved checkpoint    {f'{i=}':8}  {eval_loss=:.3f}")
                best_loss = eval_loss
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
