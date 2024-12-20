import os
import sys
from pprint import pprint
from typing import Literal

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


def get_batch(
    config: Config,
    env: Environment,
    split: Literal["train", "val"],
) -> tuple[Tensor, Tensor]:
    if split == "train":
        data = np.memmap(
            os.path.join(config.data_dir, "train.bin"),
            dtype=np.uint16,
            mode="r",
        )
    else:
        data = np.memmap(
            os.path.join(config.data_dir, "val.bin"),
            dtype=np.uint16,
            mode="r",
        )

    def make_block(i: int) -> Tensor:
        return torch.from_numpy(data[i : i + config.block_size].astype(np.int64))

    idxs = torch.randint(
        len(data) - config.block_size,
        (config.batch_size,),
    )

    x = torch.stack([make_block(i) for i in idxs])  # type: ignore
    y = torch.stack([make_block(i + 1) for i in idxs])  # type: ignore

    if env.pin_memory:
        x = x.pin_memory().to(env.pin_memory_device, non_blocking=True)
        y = y.pin_memory().to(env.pin_memory_device, non_blocking=True)
    else:
        x = x.to(env.device)
        y = y.to(env.device)

    return x, y


@torch.no_grad()
def evaluate_loss(
    config: Config,
    env: Environment,
    model: TransformerLMHead,
    decoder: bool,
    max_iters=50,
) -> float:
    """Evaluates `model` loss on `dataset`."""
    model.eval()

    loss_sum = 0.0
    cnt = 0
    for _ in range(max_iters):
        x, y = get_batch(config, env, split="val")

        with env.context:
            logits = model(x, decoder=decoder)
            loss = flat_cross_entropy(logits, y)

        loss_sum += loss.cpu().item() * len(x)
        cnt += len(x)

    return loss_sum / cnt


def train(config: Config, env: Environment) -> None:
    print(f"{env.device=}, env.context={str(type(env.context))[8 : -2]}", end=", ")
    print(f"{env.pin_memory=}, {env.pin_memory_device=}, {env.compile_blocks=}")
    pprint(config.to_dict())

    wandb.init(
        dir=config.results_dir,
        project="encoder-openwebtext",
        config=config.to_dict(),
        name=config.name + ("_resumed" if config.resume else ""),
        resume=False,
    )

    env.seed_everything(config.seed)

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=50304,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        use_wpe=config.use_wpe,
    )

    decoder_model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)
    encoder_model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)

    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(
            decoder_model.named_parameters(),
            encoder_model.named_parameters(),
        ):
            assert n1 == n2
            p1.copy_(p2)

    decoder_optimizer = configure_optimizer(
        decoder_model,
        lr=config.min_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        custom_optim_groups=config.custom_optim_groups,
        device=env.device,
    )

    encoder_optimizer = configure_optimizer(
        encoder_model,
        lr=config.min_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        custom_optim_groups=config.custom_optim_groups,
        device=env.device,
    )

    lr_schedule = LRSchedule(config)

    i = 0

    decoder_best_loss = float("inf")
    encoder_best_loss = float("inf")

    decoder_losses = []
    encoder_losses = []

    if config.resume:
        decoder_load_path = os.path.join(
            config.model_dir, "decoder_" + config.checkpoint_name
        )
        decoder_checkpoint = torch.load(decoder_load_path, weights_only=False)
        decoder_model.load_state_dict(decoder_checkpoint["model"])
        decoder_optimizer.load_state_dict(decoder_checkpoint["optimizer"])
        decoder_i = decoder_checkpoint["i"]
        decoder_best_loss = decoder_checkpoint["best_loss"]

        encoder_load_path = os.path.join(
            config.model_dir, "encoder_" + config.checkpoint_name
        )
        encoder_checkpoint = torch.load(encoder_load_path, weights_only=False)
        encoder_model.load_state_dict(encoder_checkpoint["model"])
        encoder_optimizer.load_state_dict(encoder_checkpoint["optimizer"])
        encoder_i = encoder_checkpoint["i"]
        encoder_best_loss = encoder_checkpoint["best_loss"]

        i = max(encoder_i, decoder_i)

    while True:
        i += 1

        decoder_model.train()
        encoder_model.train()

        lr = lr_schedule(i)

        for param_group in decoder_optimizer.param_groups:
            param_group["lr"] = lr

        for param_group in encoder_optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(config, env, split="train")

        if config.sample_forward_idxs is not None:
            n_idxs = int(np.ceil(config.sample_forward_idxs * config.block_size))
            forward_idxs = sorted(int(i) for i in np.random.permutation(config.block_size)[:n_idxs])
        else:
            forward_idxs = None

        # train step for decoder
        with env.context:
            logits = decoder_model(x, forward_idxs=forward_idxs, decoder=True)
            loss = flat_cross_entropy(logits[:, forward_idxs], y[:, forward_idxs])

        loss.backward()
        decoder_optimizer.step()
        decoder_optimizer.zero_grad(set_to_none=True)

        decoder_losses.append(loss.item())

        # train step for encoder
        with env.context:
            logits = encoder_model(x, forward_idxs=forward_idxs, decoder=False)
            loss = flat_cross_entropy(logits[:, forward_idxs], y[:, forward_idxs])

        loss.backward()
        encoder_optimizer.step()
        encoder_optimizer.zero_grad(set_to_none=True)

        encoder_losses.append(loss.item())

        if i % config.eval_interval == 0:
            # evaluate and save checkpoint for decoder
            decoder_train_loss = float(np.mean(decoder_losses[-config.eval_interval :]))
            decoder_val_loss = evaluate_loss(config, env, decoder_model, decoder=True)
            wandb.log(
                {
                    "decoder_train_loss": decoder_train_loss,
                    "decoder_val_loss": decoder_val_loss,
                },
                step=i,
            )
            decoder_best_loss = min(decoder_best_loss, decoder_val_loss)
            decoder_checkpoint = {
                "model": decoder_model.state_dict(),
                "optimizer": decoder_optimizer.state_dict(),
                "i": i,
                "best_loss": decoder_best_loss,
            }
            save_path = os.path.join(
                config.model_dir, "decoder_" + config.checkpoint_name
            )
            torch.save(decoder_checkpoint, save_path)

            # evaluate and save checkpoint for encoder
            encoder_train_loss = float(np.mean(encoder_losses[-config.eval_interval :]))
            encoder_val_loss = evaluate_loss(config, env, encoder_model, decoder=False)
            wandb.log(
                {
                    "encoder_train_loss": encoder_train_loss,
                    "encoder_val_loss": encoder_val_loss,
                },
                step=i,
            )
            encoder_best_loss = min(encoder_best_loss, encoder_val_loss)
            encoder_checkpoint = {
                "model": encoder_model.state_dict(),
                "optimizer": encoder_optimizer.state_dict(),
                "i": i,
                "best_loss": encoder_best_loss,
            }
            save_path = os.path.join(
                config.model_dir, "encoder_" + config.checkpoint_name
            )
            torch.save(encoder_checkpoint, save_path)

            print(
                f"saved checkpoint    {f'{i=}':8}  {decoder_val_loss=:.3f}    {encoder_val_loss=:.3f}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python train_openwebtext.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    env = Environment()

    assert config.task == "openwebtext"

    train(config, env)
