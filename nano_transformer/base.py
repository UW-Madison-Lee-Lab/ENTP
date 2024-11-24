from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    n_positions: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    use_wpe: bool = True


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        size_out = x.shape[:-1] + (self.weight.shape[-1],)
        x = torch.addmm(self.bias, x.view(-1, x.shape[-1]), self.weight)
        x = x.view(size_out)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.c_attn = Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = Linear(self.n_embd, self.n_embd)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x: Tensor, is_causal: bool) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        dropout_p = self.dropout if self.training else 0
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def gelu(x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * x**3)))

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, is_causal: bool) -> Tensor:
        x = x + self.attn(self.ln_1(x), is_causal)
        x = x + self.mlp(self.ln_2(x))
        return x


def flat_cross_entropy(logits: Tensor, target: Tensor, ignore_index=-100) -> Tensor:
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1), ignore_index=ignore_index)


def configure_optimizer(
    model: nn.Module,
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.01,
    custom_optim_groups: list[dict[str, Any]] = [],
    device: str = "cpu",
) -> optim.AdamW:  # type: ignore
    """Configures AdamW optimizer."""
    custom_params = set(sum([d["params"] for d in custom_optim_groups], []))
    filtered_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n not in custom_params
    ]

    decay_params = [p for p in filtered_params if p.dim() >= 2]
    nodecay_params = [p for p in filtered_params if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    name2param = dict(model.named_parameters())
    for d in custom_optim_groups:
        for i in range(len(d["params"])):
            d["params"][i] = name2param[d["params"][i]]

    optim_groups += custom_optim_groups

    return optim.AdamW(optim_groups, lr=lr, betas=betas, fused=device == "cuda")  # type: ignore
