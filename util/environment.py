import random
from contextlib import nullcontext
from typing import ContextManager

import numpy as np
import torch


class Environment:
    """Configures the environment based on the backend that is available."""

    def __init__(self) -> None:
        torch.set_float32_matmul_precision("high")
        self.context: ContextManager = nullcontext()
        self.pin_memory = False
        self.pin_memory_device = ""
        self.compile_blocks = False

        if torch.cuda.is_available():
            self.device = "cuda"
            self.context = torch.autocast(self.device, dtype=torch.bfloat16)
            self.pin_memory = True
            self.pin_memory_device = "cuda"
            self.compile_blocks = torch.__version__ >= "2.4.0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    @staticmethod
    def seed_everything(seed: int) -> None:
        """Sets seeds for Python, NumPy, and Torch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
