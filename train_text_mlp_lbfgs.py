import os
import sys
from math import pi, sqrt
from pprint import pprint
from typing import Optional, Sequence

import torch
from torch import Tensor, nn, optim
from torch.utils import data

import wandb
from evaluate_addition import evaluate_split_with_model
from nano_transformer import TransformerConfig, flat_cross_entropy
from util import Config, Environment, load_data


class LanguageMLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.n_features = config.n_positions * config.n_embd
        self.fc1 = nn.Linear(self.n_features, self.n_features)
        self.fc2 = nn.Linear(self.n_features, config.vocab_size)

    @staticmethod
    def gelu(x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * x**3)))

    def forward(
        self,
        input_ids: Optional[Tensor],
        forward_idxs: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> Tensor:
        x = self.wte(input_ids)
        y = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.config.vocab_size,
            dtype=x.dtype,
            device=x.device,
        )

        for t in range(x.shape[1]):
            if forward_idxs is None or t in forward_idxs:
                x_t = torch.zeros(
                    x.shape[0],
                    self.n_features,
                    dtype=x.dtype,
                    device=x.device,
                )
                x_t[:, -(1 + t) * self.config.n_embd :] = x[:, : t + 1].view(
                    x.shape[0], -1
                )
                y[:, t] = self.fc2(self.gelu(self.fc1(x_t)))

        return y


@torch.no_grad()
def evaluate_loss(
    config: Config,
    env: Environment,
    model: LanguageMLP,
    dataset: data.Dataset,
    max_iters=1,
) -> float:
    """Evaluates `model` loss on `dataset`."""
    model.eval()
    data_loader = data.DataLoader(
        dataset,
        batch_size=config.test_batch_size,
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


def train(config: Config, env: Environment) -> None:
    """
    Trains model using config parameters. Assumes data is in `config.data_dir`.
    Saves model in `config.model_dir`.
    """

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

    train_dataset, char2int = load_data(config, split="train")
    val_dataset, _ = load_data(config, split="val", char2int=char2int)

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=len(char2int),
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        use_wpe=config.use_wpe,
    )

    model = LanguageMLP(model_config).to(env.device)

    optimizer = optim.LBFGS(model.parameters(), lr=1.0)  # type: ignore

    i = 0
    best_val_loss = float("inf")
    n_evals_without_improving = 0

    if config.resume:
        load_path = os.path.join(config.model_dir, config.checkpoint_name)
        checkpoint = torch.load(load_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        i = checkpoint["i"]
        best_val_loss = checkpoint["best_val_loss"]

    while True:
        train_data_loader = data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=env.pin_memory,
            pin_memory_device=env.pin_memory_device,
        )

        for x, y in train_data_loader:
            i += 1

            model.train()

            x = x.to(env.device)
            y = y.to(env.device)

            def closure():
                with env.context:
                    optimizer.zero_grad()
                    logits = model(x, decoder=config.decoder)
                    loss = flat_cross_entropy(logits, y)
                    loss.backward()
                    return loss

            loss = optimizer.step(closure)

            wandb.log({"train_loss": loss.item()}, step=i)

            if i % config.eval_interval == 0:
                val_loss = evaluate_loss(config, env, model, val_dataset)
                wandb.log({"val_loss": val_loss}, step=i)

                if val_loss < best_val_loss:
                    n_evals_without_improving = 0
                    print(f"saved checkpoint    {f'{i=}':8}  {val_loss=:.3f}")
                    best_val_loss = val_loss
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "i": i,
                        "best_val_loss": best_val_loss,
                    }
                    save_path = os.path.join(config.model_dir, config.checkpoint_name)
                    torch.save(checkpoint, save_path)
                else:
                    n_evals_without_improving += 1

                if config.test_accuracy_during_training:
                    evaluate_split_with_model(model, config, env, split="train", step=i)  # type: ignore
                    evaluate_split_with_model(model, config, env, split="val", step=i)  # type: ignore
                    evaluate_split_with_model(model, config, env, split="test", step=i)  # type: ignore

            if i >= config.max_iters or (
                n_evals_without_improving >= config.max_evals_without_improving
                and best_val_loss < config.max_loss_for_early_stopping
            ):
                run.finish()
                return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python train_text.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    env = Environment()

    train(config, env)
