import os
import sys
from pprint import pprint

import torch
from torch.utils import data
import tiktoken
import wandb
from datasets import load_dataset
from nano_transformer import (
    TransformerConfig,
    TransformerLMHead,
    configure_optimizer,
    flat_cross_entropy,
)
from util import Config, Environment, LRSchedule, SeqAlignmentBlockDataset

def load_and_split_data(config: Config) -> tuple[SeqAlignmentBlockDataset, SeqAlignmentBlockDataset, int]:
    tokenizer = tiktoken.get_encoding("gpt2")
    ds = load_dataset("YurtsAI/named_entity_recognition_document_context")

    train_data = []
    train_labels = []

    for example in ds["train"]:
        for word, tag in zip(example["tokens"], example["ner_tags"]):
            try:
                x = tokenizer.encode_ordinary(word)
            except Exception:
                continue

            train_data += x
            train_labels += [bool(tag)] * len(x)

        train_data.append(tokenizer.eot_token)
        train_labels.append(0)
    
    test_data = []
    test_labels = []

    for example in ds["test"]:
        for word, tag in zip(example["tokens"], example["ner_tags"]):
            try:
                x = tokenizer.encode_ordinary(word)
            except Exception:
                continue

            test_data += x
            test_labels += [bool(tag)] * len(x)

        test_data.append(tokenizer.eot_token)
        test_labels.append(0)
        
        test_data.append(tokenizer.eot_token)
        test_labels.append(0)

    
    train_dataset = SeqAlignmentBlockDataset(
        torch.tensor(train_data, dtype=torch.int64),
        torch.tensor(train_labels, dtype=torch.int64),
        config,
    )

    test_dataset = SeqAlignmentBlockDataset(
        torch.tensor(test_data, dtype=torch.int64),
        torch.tensor(test_labels, dtype=torch.int64),
        config,
    )

    return train_dataset, test_dataset, tokenizer.n_vocab


@torch.no_grad()
def evaluate_accuracy_and_loss(
    config: Config,
    env: Environment,
    model: TransformerLMHead,
    dataset: data.Dataset,
    max_iters=100,
) -> tuple[float, float]:
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
    n_correct = 0
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
        n_correct += torch.mean((torch.argmax(logits, dim=2) == y).float()).item() * len(x)
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

    train_dataset, test_dataset, _ = load_and_split_data(config)

    model_config = TransformerConfig(
        n_positions=48,
        vocab_size=50304,
        n_layer=24,
        n_head=6,
        n_embd=384,
        dropout=0,
        use_wpe=True,
    )
    
    model = TransformerLMHead(model_config, env.compile_blocks).to(env.device)
    checkpoint_path = f"models/seperate/{'decoder' if config.decoder else 'encoder'}_medium_deep_openwebtext.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.lm_head = torch.nn.Linear(model.lm_head.in_features, 2, device=env.device)

    # train probe
    optimizer = configure_optimizer(
        model.lm_head,
        lr=config.min_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        custom_optim_groups=config.custom_optim_groups,
        device=env.device,
    )

    lr_schedule = LRSchedule(config)
    i = 0

    test_accuracy, test_loss = evaluate_accuracy_and_loss(config, env, model, test_dataset)
    print(f"{i=}, {test_accuracy=:.4f}, {test_loss=:.4f}")
    wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss}, step=i)
    
    while i < config.linear_probe_training_iters:
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
                test_accuracy, test_loss = evaluate_accuracy_and_loss(config, env, model, test_dataset)
                print(f"{i=}, {test_accuracy=:.4f}, {test_loss=:.4f}")
                wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss}, step=i)

            if i >= config.linear_probe_training_iters:
                break
    
    # train model
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
                test_accuracy, test_loss = evaluate_accuracy_and_loss(config, env, model, test_dataset)
                print(f"{i=}, {test_accuracy=:.4f}, {test_loss=:.4f}")
                wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss}, step=i)

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
