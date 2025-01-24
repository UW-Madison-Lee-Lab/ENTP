import ast
import sys
from collections import defaultdict
from functools import partial
from pprint import pprint

import tiktoken
import torch
from datasets import load_dataset
from torch.utils import data

import wandb
from nano_transformer import (
    TransformerConfig,
    TransformerLMHead,
    configure_optimizer,
    flat_cross_entropy,
)
from util import Config, Environment, LRSchedule, SequenceDataset, left_pad_collate


def load_and_split_data(
    tokenizer: tiktoken.Encoding,
    config: Config,
) -> tuple[SequenceDataset, SequenceDataset, dict[int, SequenceDataset]]:
    ds = load_dataset("CLUTRR/v1", "gen_train234_test2to10")

    train_ids = []
    train_labels = []
    for row in ds["train"]:  # type: ignore
        if len(train_ids) >= config.n_train:
            break

        if row["target"] not in [1, 9, 12, 13, 20]:  # type: ignore
            q = ast.literal_eval(row["query"])  # type: ignore
            prompt = tokenizer.encode_ordinary(
                row["story"].replace("[", "").replace("]", "") + f" {q[1]} is {q[0]}'s"  # type: ignore
            )
            ans = tokenizer.encode_ordinary(" " + row["target_text"])  # type: ignore
            if len(prompt) <= config.block_size and len(ans) == 1:
                train_ids.append(torch.tensor(prompt))
                train_labels.append(torch.tensor(ans[0]))

    val_ids = []
    val_labels = []
    for row in ds["validation"]:  # type: ignore
        if row["target"] not in [1, 9, 12, 13, 20]:  # type: ignore
            q = ast.literal_eval(row["query"])  # type: ignore
            prompt = tokenizer.encode_ordinary(
                row["story"].replace("[", "").replace("]", "") + f" {q[1]} is {q[0]}'s"  # type: ignore
            )
            ans = tokenizer.encode_ordinary(" " + row["target_text"])  # type: ignore
            if len(prompt) <= config.block_size and len(ans) == 1:
                val_ids.append(torch.tensor(prompt))
                val_labels.append(torch.tensor(ans[0]))

    test_ids = defaultdict(list)
    test_labels = defaultdict(list)
    for row in ds["test"]:  # type: ignore
        if row["target"] not in [1, 9, 12, 13, 20]:  # type: ignore
            level = int(row["task_name"].split(".")[-1])  # type: ignore
            q = ast.literal_eval(row["query"])  # type: ignore
            prompt = tokenizer.encode_ordinary(
                row["story"].replace("[", "").replace("]", "") + f" {q[1]} is {q[0]}'s"  # type: ignore
            )
            ans = tokenizer.encode_ordinary(" " + row["target_text"])  # type: ignore
            if len(prompt) <= config.block_size and len(ans) == 1:
                test_ids[level].append(torch.tensor(prompt))
                test_labels[level].append(torch.tensor(ans[0]))

    test_datasets = {}
    for level in test_ids:
        test_datasets[level] = SequenceDataset(
            test_ids[level], test_labels[level], config
        )

    return (
        SequenceDataset(train_ids, train_labels, config),
        SequenceDataset(val_ids, val_labels, config),
        test_datasets,
    )


@torch.no_grad()
def evaluate_accuracy_and_loss(
    config: Config,
    env: Environment,
    model: TransformerLMHead,
    tokenizer: tiktoken.Encoding,
    dataset: data.Dataset,
    max_iters=1000,
) -> tuple[float, float]:
    """Evaluates `model` loss on `dataset`."""
    model.eval()
    data_loader = data.DataLoader(
        dataset,
        batch_size=config.test_batch_size,
        shuffle=True,
        pin_memory=env.pin_memory,
        pin_memory_device=env.pin_memory_device,
        collate_fn=partial(left_pad_collate, value=tokenizer.eot_token),
    )

    loss_sum = 0.0
    n_correct = 0
    cnt = 0
    for i, (x, y) in enumerate(data_loader):
        if i >= max_iters:
            break

        x = x.to(env.device)
        y = y.to(env.device)

        with env.context:
            logits = model(x, forward_idxs=[x.shape[1] - 1], decoder=config.decoder)[
                :, -1
            ]
            loss = flat_cross_entropy(logits, y)

        loss_sum += loss.cpu().item() * len(x)
        n_correct += torch.mean(
            (torch.argmax(logits, dim=1) == y).float()
        ).item() * len(x)
        cnt += len(x)

    return n_correct / cnt, loss_sum / cnt


@torch.no_grad()
def test_model(
    config: Config,
    env: Environment,
    model: TransformerLMHead,
    tokenizer: tiktoken.Encoding,
    test_datasets: dict[int, SequenceDataset],
    step: int,
) -> None:
    """Evaluates `model` loss on `dataset`."""
    model.eval()

    for level, dataset in test_datasets.items():
        data_loader = data.DataLoader(
            dataset,
            batch_size=config.test_batch_size,
            shuffle=True,
            pin_memory=env.pin_memory,
            pin_memory_device=env.pin_memory_device,
            collate_fn=partial(left_pad_collate, value=tokenizer.eot_token),
        )

        n_correct = 0
        cnt = 0
        for x, y in data_loader:
            x = x.to(env.device)
            y = y.to(env.device)

            with env.context:
                logits = model(
                    x, forward_idxs=[x.shape[1] - 1], decoder=config.decoder
                )[:, -1]

            n_correct += torch.mean(
                (torch.argmax(logits, dim=1) == y).float()
            ).item() * len(x)
            cnt += len(x)

        wandb.log({f"test_accuracy_{level}": n_correct / cnt}, step=step)


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

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset, val_dataset, test_datasets = load_and_split_data(tokenizer, config)

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=50304,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        use_wpe=config.use_wpe,
    )

    model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)
    checkpoint_path = f"models/seperate/{'decoder' if config.decoder else 'encoder'}_medium_openwebtext.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    optimizer = configure_optimizer(
        model,
        lr=config.min_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        custom_optim_groups=config.custom_optim_groups,
        device=env.device,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])

    lr_schedule = LRSchedule(config)
    i = 0

    while True:
        train_data_loader = data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=env.pin_memory,
            pin_memory_device=env.pin_memory_device,
            collate_fn=partial(left_pad_collate, value=tokenizer.eot_token),
        )

        for x, y in train_data_loader:
            i += 1

            model.train()

            lr = lr_schedule(i)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x = x.to(env.device)
            y = y.to(env.device)

            with env.context:
                logits = model(
                    x, forward_idxs=[x.shape[1] - 1], decoder=config.decoder
                )[:, -1]
                loss = flat_cross_entropy(logits, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            wandb.log({"train_loss": loss.item()}, step=i)

            if i % config.eval_interval == 0:
                val_accuracy, val_loss = evaluate_accuracy_and_loss(
                    config, env, model, tokenizer, val_dataset
                )
                print(f"{i=}, {val_accuracy=:.4f}, {val_loss=:.4f}")
                wandb.log({"val_accuracy": val_accuracy, "val_loss": val_loss}, step=i)
                # test_model(config, env, model, tokenizer, test_datasets, step=i)

            if i >= config.max_iters:
                run.finish()
                return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python train_text.py <config-path>")
        exit(1)

    config = Config.from_json(sys.argv[1])
    env = Environment()

    train(config, env)


# def load_and_split_data(tokenizer: tiktoken.Encoding, config: Config) -> tuple[SequenceDataset, SequenceDataset, int]:
#     df = pd.read_csv(f"{config.data_dir}/data.csv")
#     train_df, test_df = train_test_split(df, test_size=0.1, random_state=config.seed)

#     labels_set = set()
#     train_ids = []
#     train_labels = []
#     for _, row in train_df.iterrows():
#         x = tokenizer.encode_ordinary(row["text"])
#         if len(x) <= config.block_size:
#             train_ids.append(torch.tensor(x))
#             train_labels.append(torch.tensor(row["target"]))
#             labels_set.add(row["target"])

#     test_ids = []
#     test_labels = []
#     for _, row in test_df.iterrows():
#         x = tokenizer.encode_ordinary(row["text"])
#         if len(x) <= config.block_size and row["target"] in labels_set:
#             test_ids.append(torch.tensor(x))
#             test_labels.append(torch.tensor(row["target"]))

#     return SequenceDataset(train_ids, train_labels, config), SequenceDataset(test_ids, test_labels, config), len(labels_set)
