import os
import sys
from pprint import pprint

import torch
import wandb
from torch.utils import data

from eval_addition import eval_model
from nano_transformer import TransformerConfig, TransformerLMHead, flat_cross_entropy
from util import Config, Environment, LRSchedule, load_data, seed_everything


@torch.no_grad()
def evaluate_loss(
    config: Config,
    env: Environment,
    model: TransformerLMHead,
    dataset: data.Dataset,
    max_iters=100,
) -> float:
    """Evaluates `model` loss on `dataset`."""
    model.eval()
    data_loader = data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=env.pin_memory,
        pin_memory_device=env.pin_memory_device,
    )

    loss_sum = 0.0
    cnt = 0
    for i, (x, y) in enumerate(data_loader):
        if i >= max_iters:
            break

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            logits = model(x, decoder=config.decoder)
            loss = flat_cross_entropy(logits, y)

        loss_sum += loss.cpu().item() * len(x)
        cnt += len(x)

    return loss_sum / cnt


def train(config: Config, resume: bool = False) -> None:
    """
    Trains model using config parameters. Assumes data is in `config.data_dir`.
    Saves model in `config.model_dir`. Evaluates model after training.
    """

    env = Environment()

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

    seed_everything(config.seed)

    train_dataset, char2int = load_data(config, split="train")
    val_dataset, _ = load_data(config, split="val", char2int=char2int)

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=len(char2int),
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
    i = 1
    best_val_loss = float("inf")

    if resume:
        load_path = os.path.join(config.model_dir, config.checkpoint_name)
        checkpoint = torch.load(load_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        i = checkpoint["i"]
        best_val_loss = checkpoint["best_val_loss"]

    while i <= config.max_iters:
        train_data_loader = data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=env.pin_memory,
            pin_memory_device=env.pin_memory_device,
        )

        for x, y in train_data_loader:
            if i > config.max_iters:
                break

            model.train()

            lr = lr_schedule(i)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x = x.to(env.device)
            y = y.to(env.device)

            with env.context:
                logits = model(x, decoder=config.decoder)
                loss = flat_cross_entropy(logits, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            wandb.log({"train_loss": loss.item()}, step=i)

            if i % config.eval_interval == 0:
                val_loss = evaluate_loss(config, env, model, val_dataset)
                wandb.log({"val_loss": val_loss}, step=i)

                if val_loss < best_val_loss:
                    print(f"saved checkpoint at {i=}, {val_loss=:.2f}")
                    best_val_loss = val_loss
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "i": i,
                        "best_val_loss": best_val_loss,
                    }
                    save_path = os.path.join(config.model_dir, config.checkpoint_name)
                    torch.save(checkpoint, save_path)

            i += 1

    run.finish()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python train.py config-path.json")
        exit(1)

    config = Config(sys.argv[1])

    train(config)
    eval_model(config, log_incorrect_examples=True)
