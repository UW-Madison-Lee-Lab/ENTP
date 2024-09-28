import os
import random
import sys
from pprint import pprint

import numpy as np
import torch
from torch import Tensor

import wandb
from nano_transformer import (
    TransformerConfig,
    TransformerLMHead,
    configure_optimizer,
    flat_cross_entropy,
)
from util import Config, Environment, LRSchedule


class LenGenCountingGenerator:
    def __init__(self, config: Config) -> None:
        self.train_len_max = config.train_len_max
        self.test_len_max = config.test_len_max
        self.block_size = config.block_size
        self.batch_size = config.batch_size

        self.max_num = 2 * max(self.train_len_max, self.test_len_max) - 1
        self.to = self.max_num + 1
        self.bos = self.max_num + 2
        self.eos = self.max_num + 3

    @property
    def vocab_size(self) -> int:
        return self.eos + 1

    def generate_sequence(self, seq_len) -> list[int]:
        start = random.randint(0, self.max_num - seq_len + 1)
        end = start + seq_len - 1
        seq = list(range(start, end + 1))
        return [self.bos, start, end, self.to] + seq + [self.eos]

    def generate_train_block(self) -> list[int]:
        block: list[int] = []
        while len(block) <= self.block_size:
            block += self.generate_sequence(random.randint(1, self.train_len_max))

        extra = len(block) - self.block_size - 1
        i = random.randint(0, extra)
        block = block[i : i + self.block_size + 1]
        assert len(block) == self.block_size + 1
        return block

    def generate_train_batch(self) -> tuple[Tensor, Tensor]:
        data = torch.tensor(
            [self.generate_train_block() for _ in range(self.batch_size)]
        )
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    def generate_test_batch(
        self, seq_len, batch_size
    ) -> tuple[Tensor, Tensor, list[int]]:
        data = torch.tensor(
            [self.generate_sequence(seq_len) for _ in range(batch_size)]
        )
        x = data[:, :-1]
        y = data[:, 1:]
        forward_idxs = list(range(3, x.shape[1]))
        return x, y, forward_idxs


@torch.no_grad()
def log_accuracy(
    model: TransformerLMHead,
    data_generator: LenGenCountingGenerator,
    step: int,
    config: Config,
    env: Environment,
    n_iters=4,
) -> None:
    model.eval()
    id_acc = []
    ood_acc = []

    for seq_len in range(1, config.test_len_max + 1):
        acc = []

        for _ in range(n_iters):
            x, y, forward_idxs = data_generator.generate_test_batch(
                seq_len, config.test_batch_size
            )

            x = x.to(env.device)
            y = y.to(env.device)

            with env.context:
                logits = model(x, decoder=config.decoder, forward_idxs=forward_idxs)

            y_pred = torch.argmax(logits, dim=2)

            y = y[:, forward_idxs]
            y_pred = y_pred[:, forward_idxs]
            acc.append(torch.mean(torch.all(y == y_pred, dim=1).float()).item())

        wandb.log({f"len_{seq_len}_acc": np.mean(acc)}, step=step)

        if seq_len > config.train_len_max:
            ood_acc.append(np.mean(acc))
        else:
            id_acc.append(np.mean(acc))

    wandb.log({"id_acc": np.mean(id_acc)}, step=step)
    wandb.log({"ood_acc": np.mean(ood_acc)}, step=step)


def train(config: Config, env: Environment) -> None:
    print(f"{env.device=}, env.context={str(type(env.context))[8 : -2]}", end=", ")
    print(f"{env.pin_memory=}, {env.pin_memory_device=}, {env.compile_blocks=}")
    pprint(config.to_dict())

    run = wandb.init(
        dir=config.results_dir,
        project="encoder-len-gen-counting",
        config=config.to_dict(),
        name=config.name + ("_resumed" if config.resume else ""),
        resume=config.resume,
    )

    env.seed_everything(config.seed)

    data_generator = LenGenCountingGenerator(config)

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=data_generator.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        use_wpe=config.use_wpe,
    )

    model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)

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

        x, y = data_generator.generate_train_batch()

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            logits = model(x, decoder=config.decoder)
            loss = flat_cross_entropy(logits, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

        if i % config.eval_interval == 0:
            eval_loss = float(np.mean(losses[-config.eval_interval :]))
            wandb.log({"loss": eval_loss}, step=i)

            if config.test_accuracy_during_training:
                log_accuracy(model, data_generator, i, config, env)

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
        print("usage: python train_len_gen_counting.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    env = Environment()

    assert config.task == "len_gen_counting"

    train(config, env)
