import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import Block, Linear, TransformerConfig


class Transformer(nn.Module):
    """Transformer model without lm-head, with optional causal encoder."""

    def __init__(self, config: TransformerConfig, compile_blocks=False) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.apply(self.__init_weights)
        for p_name, p in self.named_parameters():
            if p_name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        if compile_blocks:
            for block in self.h:
                block.compile()

    @staticmethod
    def __init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def __decoder_forward(self, x: Tensor, is_causal=True) -> Tensor:
        """Helper method for forward. Decoder forward is O(N^2)."""
        for block in self.h:
            x = block(x, is_causal)
        return self.ln_f(x)

    def __encoder_forward(
        self,
        x: Tensor,
        forward_idxs: Optional[Sequence[int]],
        is_causal=False,
    ) -> Tensor:
        """Helper method for forward. Encoder forward is O(N^3)."""
        y = torch.zeros_like(x)
        for t in range(x.shape[1]):
            if forward_idxs is None or t in forward_idxs:
                x_t = x[:, : t + 1]
                for block in self.h:  # type: ignore
                    x_t = block(x_t, is_causal)
                y[:, t] = self.ln_f(x_t[:, t])
        return y

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        input_embds: Optional[Tensor] = None,
        decoder: bool = True,
        forward_idxs: Optional[Sequence[int]] = None,
    ) -> Tensor:
        if input_ids is not None:
            x = self.wte(input_ids)
        elif input_embds is not None:
            x = input_embds
        else:
            raise ValueError("`input_ids` and `input_embds` cannot both be `None`.")

        T = x.shape[1]
        assert T <= self.config.n_positions

        if self.config.use_wpe:
            position_ids = torch.arange(T, device=x.device)
            x += self.wpe(position_ids)

        if decoder:
            return self.__decoder_forward(x)
        else:
            return self.__encoder_forward(x, forward_idxs)


class TransformerLMHead(nn.Module):
    """Transformer language model, with optional causal encoder."""

    def __init__(self, config: TransformerConfig, compile_blocks=False) -> None:
        super().__init__()
        self.n_positions = config.n_positions
        self.transformer = Transformer(config, compile_blocks)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        input_embds: Optional[Tensor] = None,
        decoder: bool = True,
        forward_idxs: Optional[Sequence[int]] = None,
    ) -> Tensor:
        x = self.transformer(input_ids, input_embds, decoder, forward_idxs)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        decoder: bool = True,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.n_positions:
                input_ids = input_ids[:, -self.n_positions :]

            logits = self(
                input_ids,
                decoder=decoder,
                forward_idxs=(input_ids.shape[1] - 1,),
            )

            if deterministic:
                next_id = torch.argmax(logits[:, -1:], dim=2)
            else:
                distribution = F.softmax(logits[:, -1] / temperature, dim=1)
                next_id = torch.multinomial(distribution, num_samples=1)

            input_ids = torch.cat((input_ids, next_id), dim=1)

        return input_ids
