import os
import sys
from pprint import pprint

import numpy as np
import torch
import wandb
from torch import Tensor

from nano_transformer import TransformerConfig, TransformerLMHead, flat_cross_entropy
from util import Config, Environment, LRSchedule


class OrthoVecGenerator:
    def __init__(self, config: Config, true_rate_low=0.025, true_rate_high=0.05):
        self.true_rate_low = true_rate_low
        self.true_rate_high = true_rate_high
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.batch_size = config.batch_size

    def generate_batch(self) -> tuple[Tensor, Tensor]:
        true_rate = (
            torch.rand(()) * (self.true_rate_high - self.true_rate_low)
            + self.true_rate_low
        )
        x = (
            torch.rand(self.batch_size, self.n_embd, self.block_size) < true_rate
        ).float()
        y = torch.zeros(self.batch_size, self.block_size, dtype=torch.bool)
        for m in range(2, self.block_size):
            mask = x[:, :, m]
            for i in range(m - 1):
                for j in range(i + 1, m):
                    y[:, m] |= (
                        torch.diag((x[:, :, i] * mask) @ (x[:, :, j] * mask).T) > 0
                    )

        return x, y.long()


@torch.no_grad()
def test_accuracy(
    model: TransformerLMHead,
    data_generator: OrthoVecGenerator,
    config: Config,
    env: Environment,
    n_iters=100,
) -> float:
    accuracies = []

    for _ in range(n_iters):
        x, y = data_generator.generate_batch()

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            logits = model(input_embds=x, decoder=config.decoder)

        y_pred = torch.argmax(logits, dim=2)

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

    data_generator = OrthoVecGenerator(config)

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

        x, y = data_generator.generate_batch()

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            logits = model(input_embds=x, decoder=config.decoder)
            loss = flat_cross_entropy(logits, y)

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
