import numpy as np

from .config import Config


class LRSchedule:
    """Cosine learning rate scheduler with warmup."""

    def __init__(self, config: Config) -> None:
        self.min_lr: float = config.min_lr
        self.max_lr: float = config.max_lr
        self.warmup_iters: int = config.warmup_iters
        self.lr_decay_iters: int = config.lr_decay_iters

    def __call__(self, i: int) -> float:
        if i < self.warmup_iters:
            return self.max_lr * i / self.warmup_iters

        if i > self.lr_decay_iters:
            return self.min_lr

        decay_ratio = (i - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio and decay_ratio <= 1
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
