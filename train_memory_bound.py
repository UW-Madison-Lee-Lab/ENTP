import os
import sys
from pprint import pprint
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import wandb
from nano_transformer import (
    Transformer,
    TransformerConfig,
    configure_optimizer,
)
from util import Config, Environment, LRSchedule


class MemoryBoundSeqDataGenerator:
    def __init__(self, config: Config) -> None:
        self.dim = config.n_embd
        self.block_size = config.block_size
        self.batch_size = config.batch_size

    def generate_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        if batch_size is None:
            batch_size = self.batch_size

        x = torch.randn(batch_size, self.block_size, self.dim)

        y = torch.zeros_like(x)
        for i in range(1, self.block_size):
            a = x[:, i] / self.dim**0.5
            b = x[:, :i]
            y[:, i, 0] = torch.einsum("ik,ijk->i", a, b)

        return x, y


def train(config: Config, env: Environment) -> None:
    print(f"{env.device=}, env.context={str(type(env.context))[8 : -2]}", end=", ")
    print(f"{env.pin_memory=}, {env.pin_memory_device=}, {env.compile_blocks=}")
    pprint(config.to_dict())

    run = wandb.init(
        dir=config.results_dir,
        project="memory-bound-sequence",
        config=config.to_dict(),
        name=config.name + ("_resumed" if config.resume else ""),
        resume=False,
    )

    env.seed_everything(config.seed)

    data_generator = MemoryBoundSeqDataGenerator(config)

    model_config = TransformerConfig(
        n_positions=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        use_wpe=config.use_wpe,
    )

    model = Transformer(model_config, env.compile_blocks).to(env.device)

    optimizer = configure_optimizer(
        model,
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

        x, y = data_generator.generate_batch()

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            y_pred = model(input_embds=x, decoder=config.decoder)
            loss = F.mse_loss(y_pred[:, :, 0], y[:, :, 0])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

        if i % config.eval_interval == 0:
            eval_loss = float(np.mean(losses[-config.eval_interval :]))
            wandb.log({"loss": eval_loss}, step=i)

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
