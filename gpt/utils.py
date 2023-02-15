import random

import torch
import numpy as np


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def device(disable_cuda: bool = False) -> torch.device:
    if not disable_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
