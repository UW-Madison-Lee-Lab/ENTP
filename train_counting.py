import os
import random
import sys
from collections import Counter
from pprint import pprint
from typing import Optional

import numpy as np
import torch
import wandb
from torch import Tensor

from nano_transformer import TransformerConfig, TransformerLMHead, flat_cross_entropy
from util import Config, Environment, LRSchedule


class CountingDataGenerator:
    def __init__(self, config: Config):
        self.seed_size = config.data_gen_seed_size
        self.seed_max = config.data_gen_seed_max
        self.permutation_invariant = config.counting_permutation_invariant
        self.block_size = config.block_size
        self.batch_size = config.batch_size

    def f(self, x: list[int]) -> int:
        y = x[:-1] if self.permutation_invariant else x[:-2]
        z = x[-1] if self.permutation_invariant else x[-2]
        return Counter(y)[z]

    def generate_example(self) -> list[int]:
        seq = [random.randint(0, self.seed_max) for _ in range(self.seed_size)]

        while len(seq) <= self.block_size:
            seq.append(self.f(seq))

        assert len(seq) == self.block_size + 1
        return seq

    def generate_batch(self, batch_size: Optional[int] = None) -> tuple[Tensor, Tensor, list[int]]:
        if batch_size is None:
            batch_size = self.batch_size

        data = torch.tensor([self.generate_example() for _ in range(batch_size)])
        x = data[:, :-1]
        y = data[:, 1:]
        forward_idxs = [i for i in range(self.seed_size, batch_size)]
        return x, y, forward_idxs


@torch.no_grad()
def test_accuracy(
    model: TransformerLMHead,
    data_generator: CountingDataGenerator,
    config: Config,
    env: Environment,
    n_iters=50,
) -> float:
    model.eval()
    accuracies = []

    for _ in range(n_iters):
        x = data_generator.generate_batch(config.test_batch_size)[0].to(env.device)
        x_seed = x[:, :config.data_gen_seed_size]

        with env.context:
            x_pred = model.generate(
                x_seed, 
                max_new_tokens=config.block_size - config.data_gen_seed_size, 
                decoder=config.decoder,
            )

        accuracies.append(torch.mean((x == x_pred).float()).item())

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
        vocab_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        use_wpe=config.use_wpe,
    )

    model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)

    optimizer = model.configure_optimizer(
        lr=config.min_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        custom_optim_groups=config.custom_optim_groups,
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

            if config.log_wpe_norm:
                wpe_fro_norm = torch.norm(model.transformer.wpe.weight).item()
                wandb.log({"wpe_fro_norm": wpe_fro_norm}, step=i)

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
