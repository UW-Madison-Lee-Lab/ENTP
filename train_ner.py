import sys
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
from util import Config, Environment, LRSchedule, SequenceDataset, left_pad_collate_both


def get_label(tag: int) -> int:
    table = {
        0: 0,
        96: 1,
        44: 2,
        95: 3,
        43: 4,
        31: 5,
        64: 6,
        32: 7,
        26: 8,
        101: 9,
        54: 10,
        39: 11,
        102: 12,
        25: 13,
        42: 14,
        63: 15,
        20: 16,
        88: 17,
        41: 18,
        19: 19,
        105: 20,
    }
    if tag in table:
        return table[tag]

    return 21


def load_and_split_data(
    config: Config,
) -> tuple[SequenceDataset, SequenceDataset, tiktoken.Encoding]:
    tokenizer = tiktoken.get_encoding("gpt2")
    ds = load_dataset("YurtsAI/named_entity_recognition_document_context")

    train_data = []
    train_labels = []

    for example in ds["train"]:  # type: ignore
        if len(train_data) >= config.n_train:
            break

        tokens = []
        labels = []
        for word, tag in zip(example["tokens"], example["ner_tags"]):  # type: ignore
            try:
                x = tokenizer.encode_ordinary(word)
            except Exception:
                continue

            tokens += x
            labels += [bool(tag)] * len(x)

        tokens.append(tokenizer.eot_token)
        labels.append(0)
        if len(tokens) <= config.batch_size:
            train_data.append(torch.tensor(tokens))
            train_labels.append(torch.tensor(labels))

    test_data = []
    test_labels = []

    for example in ds["test"]:  # type: ignore
        tokens = []
        labels = []
        for word, tag in zip(example["tokens"], example["ner_tags"]):  # type: ignore
            try:
                x = tokenizer.encode_ordinary(word)
            except Exception:
                continue

            tokens += x
            labels += [bool(tag)] * len(x)

        tokens.append(tokenizer.eot_token)
        labels.append(0)

        if len(tokens) <= config.batch_size:
            test_data.append(torch.tensor(tokens))
            test_labels.append(torch.tensor(labels))

    train_dataset = SequenceDataset(train_data, train_labels, config)
    test_dataset = SequenceDataset(test_data, test_labels, config)

    return train_dataset, test_dataset, tokenizer


@torch.no_grad()
def evaluate_accuracy_and_loss(
    config: Config,
    env: Environment,
    model: TransformerLMHead,
    tokenizer: tiktoken.Encoding,
    dataset: data.Dataset,
    max_iters=400,
) -> tuple[float, float]:
    """Evaluates `model` loss on `dataset`."""
    tokenizer = tiktoken.get_encoding("gpt2")

    model.eval()
    data_loader = data.DataLoader(
        dataset,
        batch_size=config.test_batch_size,
        shuffle=True,
        pin_memory=env.pin_memory,
        pin_memory_device=env.pin_memory_device,
        collate_fn=partial(
            left_pad_collate_both, value1=tokenizer.eot_token, value2=-1
        ),
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
            logits = model(x, decoder=config.decoder)
            loss = flat_cross_entropy(logits, y, ignore_index=-1)

        loss_sum += loss.cpu().item() * len(x)
        for i in range(len(x)):
            idxs = y[i] != -1
            n_correct += torch.sum(
                (torch.argmax(logits[i, idxs], dim=1) == y[i, idxs]).float()
            ).item() / len(y[i, idxs])

        cnt += len(x)

    return n_correct / cnt, loss_sum / cnt


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

    train_dataset, test_dataset, tokenizer = load_and_split_data(config)

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=tokenizer.n_vocab,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        use_wpe=config.use_wpe,
    )

    model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)
    model.lm_head = torch.nn.Linear(model.lm_head.in_features, 22, device=env.device)

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

    test_accuracy, test_loss = evaluate_accuracy_and_loss(
        config, env, model, tokenizer, test_dataset
    )
    print(f"{i=}, {test_accuracy=:.4f}, {test_loss=:.4f}")
    wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss}, step=i)

    while True:
        train_data_loader = data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=env.pin_memory,
            pin_memory_device=env.pin_memory_device,
            collate_fn=partial(
                left_pad_collate_both, value1=tokenizer.eot_token, value2=-1
            ),
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
                logits = model(x, decoder=config.decoder)
                loss = flat_cross_entropy(logits, y, ignore_index=-1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            wandb.log({"train_loss": loss.item()}, step=i)

            if i % config.eval_interval == 0:
                test_accuracy, test_loss = evaluate_accuracy_and_loss(
                    config, env, model, tokenizer, test_dataset
                )
                print(f"{i=}, {test_accuracy=:.4f}, {test_loss=:.4f}")
                wandb.log(
                    {"test_accuracy": test_accuracy, "test_loss": test_loss}, step=i
                )

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
