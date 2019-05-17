from dataclasses import dataclass

from torch.utils.data import Dataset
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from models.custom_module import CustomModule


@dataclass
class TrainingConfig:
    dataset: Dataset
    model: CustomModule  # TODO type, not instance
    model_args: dict
    batch_size: int
    epochs: int
    val_size: float
