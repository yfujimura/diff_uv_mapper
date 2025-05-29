import torch
from dataclasses import dataclass

@dataclass
class OptimConfig:
    train_iters: int
    lr: dict
    weight: float

def inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))