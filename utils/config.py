from dataclasses import dataclass
from typing import Type

from torch.utils.data import Dataset
from models.custom_module import CustomModule


@dataclass
class TrainingConfig:
    dataset: Type[Dataset]
    dataset_args: dict
    model: Type[CustomModule]
    model_args: dict
    batch_size: int
    epochs: int
    val_size: float
    log_path: str
    save_every: int  # save model every save_every epochs
