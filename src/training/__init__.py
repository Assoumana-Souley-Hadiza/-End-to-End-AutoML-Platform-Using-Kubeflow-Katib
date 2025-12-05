"""
Training scripts for AutoML platform
"""
from .train import train_model, build_model
from .train_pbt import train_model as train_pbt_model
from .train_nas import train_nas_model
from .train_darts import train_darts_model

__all__ = [
    "train_model",
    "build_model",
    "train_pbt_model",
    "train_nas_model",
    "train_darts_model",
]



