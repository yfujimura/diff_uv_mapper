import torch
from dataclasses import dataclass

@dataclass
class OptimConfig:
    train_iters: int
    lr: dict
    weight: float
    optimize_only_texture: bool

def get_optim_config(args):
    return OptimConfig(
        args.train_iters,
        {
            "albedo": 1e-2,
            "offset": 1e-4,
        },
        args.weight,
        args.optimize_only_texture,
    )

def inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))